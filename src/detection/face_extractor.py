"""
Face Detection & Region Extraction — MediaPipe FaceLandmarker.
Logic clone 100% từ run_webcam_onnx.py class FaceRegionExtractor (L88-161).
Không thay đổi bất kỳ landmark index, pad value, hay công thức MAR nào.
"""
import math
import os
import urllib.request
from typing import Optional, Tuple

import mediapipe as mp_lib
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions


class FaceRegionExtractor:
    """
    Phát hiện khuôn mặt và trích xuất 3 vùng: Face, Eyes, Mouth.
    Tính toán MAR (Mouth Aspect Ratio).

    CLONE 100% từ run_webcam_onnx.py — không thay đổi bất kỳ tham số nào.
    """

    # ── Landmark indices — giữ nguyên từ demo gốc ──
    LEFT_EYE  = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE = [362, 385, 387, 263, 373, 380]
    MOUTH     = [61, 291, 39, 181, 0, 17, 269, 405]

    # Landmark tính MAR chuẩn: mép trái (61), mép phải (291), môi trên/dưới
    MOUTH_MAR = {
        'left': 61, 'right': 291,
        'top_1': 39,  'bot_1': 181,
        'top_2': 0,   'bot_2': 17,
        'top_3': 269, 'bot_3': 405,
    }

    def __init__(self, landmark_path: str, landmark_url: str,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        """
        Args:
            landmark_path: Đường dẫn file face_landmarker.task
            landmark_url:  URL download nếu file chưa tồn tại
            min_detection_confidence: Ngưỡng detection (giữ 0.5 như demo gốc)
            min_tracking_confidence:  Ngưỡng tracking  (giữ 0.5 như demo gốc)
        """
        # Auto-download nếu chưa có — clone từ demo gốc L53-58
        if not os.path.exists(landmark_path):
            print("⏳ Đang tải mô hình face_landmarker.task...")
            os.makedirs(os.path.dirname(landmark_path), exist_ok=True)
            urllib.request.urlretrieve(landmark_url, landmark_path)
            print("✅ Đã tải xong face_landmarker.task")

        base_options = BaseOptions(model_asset_path=landmark_path)
        options = FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=1,
            min_face_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.detector = FaceLandmarker.create_from_options(options)

    def _crop_region(self, img: np.ndarray, landmarks,
                     indices: list, pad: float = 0.3) -> np.ndarray:
        """
        Crop vùng ảnh dựa trên landmark indices với padding.
        Clone 100% từ demo gốc L111-119.
        """
        h, w = img.shape[:2]
        pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h))
               for i in indices]
        x1 = max(0, min(p[0] for p in pts) - int(w * pad))
        x2 = min(w, max(p[0] for p in pts) + int(w * pad))
        y1 = max(0, min(p[1] for p in pts) - int(h * pad))
        y2 = min(h, max(p[1] for p in pts) + int(h * pad))
        crop = img[y1:y2, x1:x2]
        return crop if crop.size > 0 else img

    def _calc_mar(self, lms, w: int, h: int) -> float:
        """
        Mouth Aspect Ratio — đo độ mở miệng chuẩn hóa theo chiều rộng.
        Clone 100% từ demo gốc L121-135.

        Công thức: MAR = (d_vert1 + d_vert2 + d_vert3) / (3.0 * d_horiz)
        """
        def dist(idx1: int, idx2: int) -> float:
            p1 = (lms[idx1].x * w, lms[idx1].y * h)
            p2 = (lms[idx2].x * w, lms[idx2].y * h)
            return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

        d_vert1 = dist(self.MOUTH_MAR['top_1'], self.MOUTH_MAR['bot_1'])
        d_vert2 = dist(self.MOUTH_MAR['top_2'], self.MOUTH_MAR['bot_2'])
        d_vert3 = dist(self.MOUTH_MAR['top_3'], self.MOUTH_MAR['bot_3'])
        d_horiz = dist(self.MOUTH_MAR['left'],  self.MOUTH_MAR['right'])

        if d_horiz == 0:
            return 0.0
        return (d_vert1 + d_vert2 + d_vert3) / (3.0 * d_horiz)

    def extract(self, img_rgb: np.ndarray) -> Tuple[
        Optional[np.ndarray], Optional[np.ndarray],
        Optional[np.ndarray], float
    ]:
        """
        Trả về: (face_crop, eye_crop, mouth_crop, mar)
        Nếu không phát hiện mặt: (None, None, None, 0.0)

        Clone 100% từ demo gốc L137-161.
        """
        mp_image = mp_lib.Image(
            image_format=mp_lib.ImageFormat.SRGB, data=img_rgb
        )
        results = self.detector.detect(mp_image)
        if not results.face_landmarks:
            return None, None, None, 0.0

        lms = results.face_landmarks[0]
        h, w = img_rgb.shape[:2]

        mar = self._calc_mar(lms, w, h)

        # Crop face — giữ nguyên padding 20px từ demo gốc
        xs = [lm.x * w for lm in lms]
        ys = [lm.y * h for lm in lms]
        x1 = max(0, int(min(xs)) - 20)
        x2 = min(w, int(max(xs)) + 20)
        y1 = max(0, int(min(ys)) - 20)
        y2 = min(h, int(max(ys)) + 20)
        face = img_rgb[y1:y2, x1:x2] if (y2 > y1 and x2 > x1) else img_rgb

        # Crop eyes & mouth — giữ nguyên pad values từ demo gốc
        eyes  = self._crop_region(img_rgb, lms,
                                  self.LEFT_EYE + self.RIGHT_EYE, pad=0.15)
        mouth = self._crop_region(img_rgb, lms, self.MOUTH, pad=0.2)

        return face, eyes, mouth, mar
