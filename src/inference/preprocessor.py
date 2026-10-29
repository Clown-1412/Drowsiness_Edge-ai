"""
Edge-optimized Preprocessing cho DMS Pipeline trên RK3588 NPU.

RKNN NPU: Chỉ resize và thêm batch dimension -> (1, 224, 224, 3) uint8 (NHWC).
NPU tự động thực hiện trừ mean và chia std ở phần cứng thông qua cấu hình subgraph.
"""
import cv2
import numpy as np

from src.config import IMG_SIZE


def preprocess_crop_uint8(img_rgb: np.ndarray, img_size: int = IMG_SIZE) -> np.ndarray:
    """
    Dành riêng cho RKNN NPU (tận dụng subgraph phần cứng):
    Chỉ resize và thêm batch dimension.
    Output shape: (1, 224, 224, 3) kiểu uint8 (NHWC chuẩn).
    Tối ưu tối đa vì không tốn tính toán float trên CPU.
    """
    resized = cv2.resize(img_rgb, (img_size, img_size))
    return resized[np.newaxis]  # (1, 224, 224, 3) uint8
