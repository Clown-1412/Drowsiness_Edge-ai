"""
Pure-numpy Preprocessing — Edge-optimized.
Clone 100% từ run_webcam_onnx.py hàm preprocess() (L172-180).
Tương đương: A.Resize + A.Normalize(ImageNet mean/std) + channel-first.
Không cần torch / Albumentations.
"""
import cv2
import numpy as np

from src.config import IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE


def preprocess(img_rgb: np.ndarray, img_size: int = IMG_SIZE) -> np.ndarray:
    """
    Chuyển crop RGB (bất kỳ kích thước) → ONNX input (1, 3, 224, 224) float32.

    Pipeline:
      1. Resize → (224, 224, 3) uint8
      2. Scale  → [0.0, 1.0] float32
      3. Normalize → ImageNet mean/std
      4. Transpose → channel-first (1, 3, H, W)

    Clone 100% từ run_webcam_onnx.py — không thay đổi bất kỳ tham số nào.
    """
    img = cv2.resize(img_rgb, (img_size, img_size))            # (224, 224, 3) uint8
    img = img.astype(np.float32) / 255.0                        # [0.0, 1.0]
    img = (img - IMAGENET_MEAN) / IMAGENET_STD                  # ImageNet normalize
    return img.transpose(2, 0, 1)[np.newaxis].astype(np.float32)  # (1, 3, H, W)
