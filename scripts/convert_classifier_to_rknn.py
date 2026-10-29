"""
Biên dịch TCN Classifier từ ONNX sang RKNN.
Target Platform: Rockchip RK3588 (NPU).

Tính năng:
- Model nhận input là chuỗi vector đặc trưng shape (1, 16, 256) kiểu float32.
- Không lượng tử hóa (do_quantization=False, chế độ FP16) để giữ độ chính xác phân loại tối đa.
- Output: probabilities shape (1, 2) [prob_drowsy, prob_alert].
"""
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ONNX_MODEL = str(PROJECT_ROOT / "models" / "dms_tcn_classifier.onnx")
RKNN_MODEL = str(PROJECT_ROOT / "models" / "dms_tcn_classifier.rknn")


def main():
    if not os.path.exists(ONNX_MODEL):
        print(f"❌ Không tìm thấy file ONNX nguồn: {ONNX_MODEL}")
        sys.exit(1)

    print("=" * 65)
    print("  RKNN Exporter: dms_tcn_classifier (NPU RK3588 - float32 FP16)")
    print("=" * 65)

    try:
        from rknn.api import RKNN
    except ImportError:
        print("❌ Chưa cài đặt thư viện 'rknn-toolkit2'.")
        print("   Vui lòng chạy script này trên môi trường x86_64 Ubuntu có rknn-toolkit2.")
        sys.exit(1)

    rknn = RKNN(verbose=True)

    print("\n--> 1. Cấu hình NPU RK3588...")
    rknn.config(
        target_platform="rk3588",
    )

    print("--> 2. Nạp ONNX TCN Classifier Model...")
    ret = rknn.load_onnx(
        model=ONNX_MODEL,
        inputs=["sequence_features"],
        input_size_list=[[1, 16, 256]],
    )
    if ret != 0:
        print(f"❌ Nạp ONNX thất bại! Mã lỗi: {ret}")
        sys.exit(ret)

    print("--> 3. Biên dịch sang RKNN (Chế độ FP16, không lượng tử hóa)...")
    ret = rknn.build(do_quantization=False)
    if ret != 0:
        print(f"❌ Biên dịch RKNN thất bại! Mã lỗi: {ret}")
        sys.exit(ret)

    print(f"--> 4. Xuất file RKNN: {RKNN_MODEL}...")
    ret = rknn.export_rknn(RKNN_MODEL)
    if ret != 0:
        print(f"❌ Xuất RKNN thất bại! Mã lỗi: {ret}")
        sys.exit(ret)

    print("\n✅ [THÀNH CÔNG] Đã xuất file RKNN TCN Classifier sẵn sàng cho NPU RK3588!")
    print(f"   Đường dẫn: {RKNN_MODEL}")
    rknn.release()


if __name__ == "__main__":
    main()
