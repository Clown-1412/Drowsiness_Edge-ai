"""
Script kiểm thử đối soát tính chính xác (Parity Benchmark) giữa:
  1. Baseline ONNX Runtime (tiền xử lý CPU float32 NCHW)
  2. RKNN NPU (tiền xử lý phần cứng uint8 NHWC)

Đo lường:
  - Cosine Similarity trên feature vector 256 chiều (yêu cầu >= 0.999)
  - Mean Absolute Error (MAE) trên attention weights 3 chiều
  - Thời gian suy luận (latency) và FPS tương đối
"""
import os
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import EXTRACTOR_ONNX, EXTRACTOR_RKNN
from src.inference.preprocessor import preprocess_crop_uint8, to_onnx_input


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    dot = np.dot(v1, v2)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm == 0:
        return 0.0
    return float(dot / norm)


def main():
    print("=" * 65)
    print("  PARITY BENCHMARK: ONNX (float32 NCHW) vs RKNN (uint8 NHWC)")
    print("=" * 65)

    if not os.path.exists(EXTRACTOR_ONNX):
        print(f"❌ Không tìm thấy file ONNX: {EXTRACTOR_ONNX}")
        sys.exit(1)
    if not os.path.exists(EXTRACTOR_RKNN):
        print(f"❌ Không tìm thấy file RKNN: {EXTRACTOR_RKNN}")
        print("   Vui lòng chạy scripts/convert_onnx_to_rknn_uint8.py trước.")
        sys.exit(1)

    # 1. Khởi tạo ONNX Runtime
    import onnxruntime as ort
    print("🧠 Nạp ONNX Runtime (CPU Baseline)...")
    onnx_sess = ort.InferenceSession(EXTRACTOR_ONNX, providers=["CPUExecutionProvider"])

    # 2. Khởi tạo RKNN
    print("🚀 Nạp RKNN Engine...")
    rknn = None
    try:
        from rknnlite.api import RKNNLite
        rknn = RKNNLite()
        ret = rknn.load_rknn(EXTRACTOR_RKNN)
        if ret != 0:
            raise RuntimeError(f"load_rknn failed code={ret}")
        rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
        print("   -> Sử dụng rknnlite.api.RKNNLite (NPU_CORE_0)")
    except ImportError:
        try:
            from rknn.api import RKNN
            rknn = RKNN()
            ret = rknn.load_rknn(EXTRACTOR_RKNN)
            if ret != 0:
                raise RuntimeError(f"load_rknn failed code={ret}")
            if hasattr(rknn, "init_runtime"):
                rknn.init_runtime()
            print("   -> Sử dụng rknn.api.RKNN (Toolkit runtime)")
        except ImportError:
            print("❌ Cả 'rknnlite' lẫn 'rknn' đều không khả dụng.")
            print("   Vui lòng chạy script này trên RK3588 hoặc máy có rknn-toolkit2.")
            sys.exit(1)

    # 3. Tạo 10 mẫu kiểm thử (kích thước động mô phỏng bounding box MediaPipe)
    np.random.seed(42)
    num_samples = 10
    cosine_scores = []
    attn_maes = []
    onnx_latencies = []
    rknn_latencies = []

    print(f"\n🔬 Đang chạy đối soát trên {num_samples} mẫu ảnh thực nghiệm...")

    for i in range(num_samples):
        # Tạo crop với kích thước ngẫu nhiên (vd: 150x180, 200x210, ...)
        h = np.random.randint(120, 300)
        w = np.random.randint(120, 300)
        sample_face = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
        sample_eye = np.random.randint(0, 256, (h // 2, w // 2, 3), dtype=np.uint8)
        sample_mouth = np.random.randint(0, 256, (h // 2, w // 2, 3), dtype=np.uint8)

        # Tiền xử lý uint8
        f_u8 = preprocess_crop_uint8(sample_face)
        e_u8 = preprocess_crop_uint8(sample_eye)
        m_u8 = preprocess_crop_uint8(sample_mouth)

        # Tiền xử lý ONNX (từ mảng uint8 chuyển sang float32 NCHW)
        f_onnx = to_onnx_input(f_u8)
        e_onnx = to_onnx_input(e_u8)
        m_onnx = to_onnx_input(m_u8)

        # Run ONNX
        t0 = time.perf_counter()
        onnx_out = onnx_sess.run(
            None,
            {"face_input": f_onnx, "eye_input": e_onnx, "mouth_input": m_onnx}
        )
        onnx_latencies.append((time.perf_counter() - t0) * 1000)

        feat_onnx = onnx_out[0][0]  # (256,)
        attn_onnx = onnx_out[1][0]  # (3,)

        # Run RKNN (truyền thẳng uint8 NHWC)
        t0 = time.perf_counter()
        rknn_out = rknn.inference(inputs=[f_u8, e_u8, m_u8])
        rknn_latencies.append((time.perf_counter() - t0) * 1000)

        feat_rknn = rknn_out[0][0]  # (256,)
        attn_rknn = rknn_out[1][0]  # (3,)

        # Đo độ tương đồng
        sim = cosine_similarity(feat_onnx, feat_rknn)
        mae = float(np.mean(np.abs(attn_onnx - attn_rknn)))

        cosine_scores.append(sim)
        attn_maes.append(mae)

        print(f"   [Mẫu {i+1:02d}] Cosine Sim: {sim:.5f} | Attention MAE: {mae:.5f}")

    # 4. Tổng kết
    avg_sim = np.mean(cosine_scores)
    min_sim = np.min(cosine_scores)
    avg_mae = np.mean(attn_maes)
    avg_onnx_ms = np.mean(onnx_latencies)
    avg_rknn_ms = np.mean(rknn_latencies)

    print("\n" + "=" * 65)
    print("                   KẾT QUẢ ĐỐI SOÁT TỔNG HỢP")
    print("=" * 65)
    print(f"  Cosine Similarity Trung bình : {avg_sim:.5f} (Min: {min_sim:.5f})")
    print(f"  Attention Weights MAE        : {avg_mae:.5f}")
    print(f"  Độ trễ ONNX CPU Trung bình   : {avg_onnx_ms:.2f} ms")
    print(f"  Độ trễ RKNN NPU Trung bình   : {avg_rknn_ms:.2f} ms")
    print("-" * 65)

    if avg_sim >= 0.999:
        print("  ✅ ĐẠT YÊU CẦU: Mô hình RKNN uint8 khớp hoàn hảo với ONNX baseline!")
    elif avg_sim >= 0.990:
        print("  ⚠️ CHẤP NHẬN ĐƯỢC: Có độ lệch số học nhỏ do làm tròn phần cứng.")
    else:
        print("  ❌ KHÔNG ĐẠT: Sai lệch số học lớn, cần kiểm tra lại mean/std.")

    print("=" * 65)

    if hasattr(rknn, "release"):
        rknn.release()


if __name__ == "__main__":
    main()
