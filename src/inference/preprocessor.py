"""
Edge-optimized Preprocessing cho DMS Pipeline.
Hỗ trợ 2 luồng:
  1. RKNN NPU: Chỉ resize và thêm batch dimension -> (1, 224, 224, 3) uint8 (NHWC).
     NPU tự động thực hiện trừ mean và chia std ở phần cứng.
  2. ONNX Fallback: Chuyển mảng uint8 NHWC -> (1, 3, 224, 224) float32 NCHW đã chuẩn hóa.
"""
import cv2
import numpy as np

from src.config import IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE


def preprocess_crop_uint8(img_rgb: np.ndarray, img_size: int = IMG_SIZE) -> np.ndarray:
    """
    Dành riêng cho RKNN NPU (tận dụng subgraph phần cứng):
    Chỉ resize và thêm batch dimension.
    Output shape: (1, 224, 224, 3) kiểu uint8 (NHWC chuẩn).
    Tối ưu tối đa vì không tốn tính toán float trên CPU.
    """
    resized = cv2.resize(img_rgb, (img_size, img_size))
    return resized[np.newaxis]  # (1, 224, 224, 3) uint8


def to_onnx_input(crop_uint8_nhwc: np.ndarray) -> np.ndarray:
    """
    Dành cho nhánh ONNX Fallback:
    Nhận mảng (1, 224, 224, 3) uint8 -> (1, 3, 224, 224) float32 NCHW đã chuẩn hóa ImageNet.
    """
    img = crop_uint8_nhwc.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return img.transpose(0, 3, 1, 2).astype(np.float32)  # (1, 3, H, W)


def preprocess(img_rgb: np.ndarray, img_size: int = IMG_SIZE) -> np.ndarray:
    """
    Hàm tương thích ngược 100% với code ONNX cũ:
    Chuyển crop RGB -> ONNX input (1, 3, 224, 224) float32 NCHW.
    """
    return to_onnx_input(preprocess_crop_uint8(img_rgb, img_size))

