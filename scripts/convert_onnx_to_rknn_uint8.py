"""
Biên dịch Feature Extractor từ ONNX sang RKNN với Subgraph Tiền Xử Lý Phần Cứng.
Target Platform: Rockchip RK3588 (NPU).

Tính năng:
- Bakes ImageNet mean_values và std_values trực tiếp vào NPU.
- Model nhận input là ảnh raw RGB uint8 có shape (1, 224, 224, 3) (NHWC).
- Không lượng tử hóa (do_quantization=False, chạy chế độ FP16) để giữ độ chính xác tuyệt đối.
"""
import os
import sys
from pathlib import Path

# Tìm đường dẫn gốc project
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ONNX_MODEL = str(PROJECT_ROOT / "models" / "dms_feature_extractor.onnx")
RKNN_MODEL = str(PROJECT_ROOT / "models" / "dms_feature_extractor.rknn")


def main():
    if not os.path.exists(ONNX_MODEL):
        print(f"❌ Không tìm thấy file ONNX nguồn: {ONNX_MODEL}")
        sys.exit(1)

    print("=" * 65)
    print("  RKNN Exporter: dms_feature_extractor (NPU RK3588 - uint8 NHWC)")
    print("=" * 65)

    try:
        from rknn.api import RKNN
    except ImportError:
        print("❌ Chưa cài đặt thư viện 'rknn-toolkit2'.")
        print("   Vui lòng chạy script này trên môi trường có cài rknn-toolkit2.")
        sys.exit(1)

    rknn = RKNN(verbose=True)

    print("\n--> 1. Cấu hình NPU Subgraph (ImageNet Normalization baked-in)...")
    # Cấu hình mean/std cho cả 3 nhánh đầu vào: face_input, eye_input, mouth_input
    # ImageNet RGB: mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]
    # Khi cấu hình mean/std, RKNN sẽ nhận input uint8 [0..255] và tự động chuẩn hóa ở phần cứng NPU
    rknn.config(
        mean_values=[
            [123.675, 116.28, 103.53],  # face_input
            [123.675, 116.28, 103.53],  # eye_input
            [123.675, 116.28, 103.53],  # mouth_input
        ],
        std_values=[
            [58.395, 57.12, 57.375],
            [58.395, 57.12, 57.375],
            [58.395, 57.12, 57.375],
        ],
        target_platform="rk3588",
    )

    print("--> 2. Nạp ONNX Model và cố định shape tĩnh...")
    ret = rknn.load_onnx(
        model=ONNX_MODEL,
        inputs=["face_input", "eye_input", "mouth_input"],
        input_size_list=[[1, 3, 224, 224], [1, 3, 224, 224], [1, 3, 224, 224]],
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

    print("\n✅ [THÀNH CÔNG] Đã xuất file RKNN mới sẵn sàng cho NPU RK3588!")
    print("   Input kỳ vọng: raw RGB uint8 shape (1, 224, 224, 3) NHWC.")
    print("   Toàn bộ tiền xử lý float32 đã được NPU đảm nhiệm ở phần cứng.")
    rknn.release()


if __name__ == "__main__":
    main()
