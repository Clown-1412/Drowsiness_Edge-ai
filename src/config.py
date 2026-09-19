"""
Cấu hình tập trung cho DMS Pipeline.
Tất cả hằng số, ngưỡng, đường dẫn được định nghĩa tại đây.
Giá trị giữ nguyên 100% so với demo gốc (run_webcam_onnx.py).
"""
import os
import sys
from pathlib import Path

import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Đường dẫn Project
# ──────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent   # d:\Python_pj\Drowsiness_Edge-ai
MODEL_DIR    = PROJECT_ROOT / 'models'

EXTRACTOR_ONNX = str(MODEL_DIR / 'dms_feature_extractor.onnx')
EXTRACTOR_RKNN = str(MODEL_DIR / 'dms_feature_extractor.rknn')
CLASSIFIER_ONNX = str(MODEL_DIR / 'dms_tcn_classifier.onnx')
LANDMARK_TASK   = str(MODEL_DIR / 'face_landmarker.task')
LANDMARK_URL    = (
    'https://storage.googleapis.com/mediapipe-models/'
    'face_landmarker/face_landmarker/float16/1/face_landmarker.task'
)

# ──────────────────────────────────────────────────────────────────────────────
# Tham số Model — KHÔNG ĐƯỢC THAY ĐỔI (đồng bộ 100% với run_webcam_onnx.py)
# ──────────────────────────────────────────────────────────────────────────────
SEQ_LEN         = 16            # Số frame cho TCN sequence
IMG_SIZE        = 224           # Kích thước ảnh input CNN
TARGET_FPS      = 5             # Tần suất inference (5 lần/giây)
SAMPLE_INTERVAL = 1.0 / TARGET_FPS   # 0.2 s / sample

# ──────────────────────────────────────────────────────────────────────────────
# Ngưỡng phân loại — đồng bộ 100% với run_webcam_onnx.py
# ──────────────────────────────────────────────────────────────────────────────
DROWSY_THRESHOLD  = 0.65        # Ngưỡng kích hoạt Drowsy
MAR_THRESHOLD     = 0.70        # Ngưỡng ngáp: MAR >= 0.70
ALPHA             = 0.65        # Hệ số EMA smoothing
YAWN_COOLDOWN_SEC = 4.0         # Cooldown sau khi ngáp (giây)

# ──────────────────────────────────────────────────────────────────────────────
# Camera
# ──────────────────────────────────────────────────────────────────────────────
CAMERA_INDEX  = 0
CAMERA_WIDTH  = 640
CAMERA_HEIGHT = 480

# ──────────────────────────────────────────────────────────────────────────────
# MediaPipe
# ──────────────────────────────────────────────────────────────────────────────
MIN_FACE_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE       = 0.5

# ──────────────────────────────────────────────────────────────────────────────
# Preprocessing — ImageNet normalization
# ──────────────────────────────────────────────────────────────────────────────
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# ──────────────────────────────────────────────────────────────────────────────
# Cấu hình Serial / Debug ESP32
# ──────────────────────────────────────────────────────────────────────────────
SERIAL_ENABLED            = True        # Bật/tắt truyền Serial (False khi test không phần cứng)
SERIAL_PORT               = 'COM1'      # Cổng Serial ESP32 (Linux: '/dev/ttyUSB0')
SERIAL_BAUDRATE           = 115200      # Tốc độ baud (khớp DEBUG_SERIAL_BAUD trên ESP32)
SERIAL_HEARTBEAT_INTERVAL = 3.0         # Chu kỳ heartbeat lặp lại (giây)


# ──────────────────────────────────────────────────────────────────────────────
# Hàm hỗ trợ
# ──────────────────────────────────────────────────────────────────────────────
def setup_cuda_dlls():
    """
    Tự động nạp DLL CUDA trên Windows (ONNX Runtime CUDA EP).
    Clone từ run_webcam_onnx.py L18-24.
    Phải gọi TRƯỚC khi import onnxruntime.
    """
    if sys.platform == 'win32':
        site_packages = os.path.join(sys.prefix, 'Lib', 'site-packages')
        for sub in ['nvidia/cublas/bin', 'nvidia/cudnn/bin',
                     'nvidia/cuda_runtime/bin']:
            dll_path = os.path.join(site_packages, sub.replace('/', os.sep))
            if os.path.exists(dll_path):
                os.add_dll_directory(dll_path)
                os.environ['PATH'] = dll_path + os.pathsep + os.environ['PATH']
