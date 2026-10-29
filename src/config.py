"""
Cấu hình tập trung cho DMS Pipeline.
Tất cả hằng số, ngưỡng, đường dẫn được định nghĩa tại đây.
Chạy thuần RKNN NPU trên RK3588, giao tiếp CAN Bus và Dashboard Bridge.
"""
import sys
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# Đường dẫn Project & Mô hình RKNN
# ──────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent   # d:\Python_pj\Drowsiness_Edge-ai
MODEL_DIR    = PROJECT_ROOT / 'models'

EXTRACTOR_RKNN  = str(MODEL_DIR / 'dms_feature_extractor.rknn')
CLASSIFIER_RKNN = str(MODEL_DIR / 'dms_tcn_classifier.rknn')
LANDMARK_TASK   = str(MODEL_DIR / 'face_landmarker.task')
LANDMARK_URL    = (
    'https://storage.googleapis.com/mediapipe-models/'
    'face_landmarker/face_landmarker/float16/1/face_landmarker.task'
)

# ──────────────────────────────────────────────────────────────────────────────
# Tham số Model
# ──────────────────────────────────────────────────────────────────────────────
SEQ_LEN         = 16            # Số frame cho TCN sequence
IMG_SIZE        = 224           # Kích thước ảnh input CNN
TARGET_FPS      = 5             # Tần suất inference (5 lần/giây)
SAMPLE_INTERVAL = 1.0 / TARGET_FPS   # 0.2 s / sample

# ──────────────────────────────────────────────────────────────────────────────
# Ngưỡng phân loại trạng thái tài xế
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
# MediaPipe Face Landmark
# ──────────────────────────────────────────────────────────────────────────────
MIN_FACE_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE       = 0.5

# ──────────────────────────────────────────────────────────────────────────────
# Cấu hình CAN Bus (RK3588 Native SocketCAN can0 / PC dev test)
# ──────────────────────────────────────────────────────────────────────────────
CAN_ENABLED        = False           # Bật/tắt CAN Bus (True khi chạy với phần cứng CAN)
CAN_INTERFACE      = 'socketcan' if sys.platform.startswith('linux') else 'virtual'
CAN_CHANNEL        = 'can0' if sys.platform.startswith('linux') else 'vcan0'
CAN_BAUDRATE       = 500000         # CAN bus baudrate (500 kbps)
CAN_ARBITRATION_ID = 0x200          # CAN message ID (11-bit standard)
CAN_SEND_INTERVAL  = 0.2            # Chu kỳ gửi CAN (giây) — 5Hz đồng bộ với Inference
CAN_FEEDBACK_ID    = 0x201          # CAN message ID phản hồi từ ECU Alert ESP32
CAN_WATCHDOG_TIMEOUT = 1.5          # Thời gian tối đa (giây) chờ phản hồi trước khi coi ECU Alert offline

# ──────────────────────────────────────────────────────────────────────────────
# Cấu hình Hiển thị (Vẽ lên Video Frame phục vụ Dashboard Stream)
# ──────────────────────────────────────────────────────────────────────────────
HUD_OVERLAY_ENABLED   = False        # Bật/tắt bảng HUD Overlay thông số trên video stream
FACE_BBOX_ENABLED     = True         # Bật/tắt Bounding Box tracking khuôn mặt kèm nhãn trạng thái

# ──────────────────────────────────────────────────────────────────────────────
# Cấu hình Dashboard Bridge (Streaming Video & Telemetry lên Web Dashboard)
# ──────────────────────────────────────────────────────────────────────────────
BRIDGE_ENABLED       = True           # Bật/tắt Dashboard Bridge (False = không khởi động server)
BRIDGE_HOST          = '0.0.0.0'      # Bind address (0.0.0.0 = mọi interface mạng)
BRIDGE_PORT          = 8000           # Port FastAPI server
STREAM_TARGET_FPS    = 20             # FPS tối đa cho video stream (20 FPS, tiết kiệm CPU)
STREAM_JPEG_QUALITY  = 70             # Chất lượng nén JPEG (0–100, 70 = cân bằng chất lượng/dung lượng)
SHM_VIDEO_NAME       = 'dms_video_shm'  # Tên POSIX Shared Memory cho video frames
TELEMETRY_PUSH_HZ    = 10             # Tần suất đẩy telemetry JSON (Hz)
