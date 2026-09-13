import math
import os
import sys
import time
import urllib.request
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions

# ── 1. Tự động nạp DLL CUDA trên Windows (ONNX Runtime CUDA EP) ──
# Cần thiết khi dùng CUDAExecutionProvider trên Windows để ort tìm được
# các thư viện nvidia (cublas, cudnn...) trong site-packages.
if sys.platform == 'win32':
    _site_packages = os.path.join(sys.prefix, 'Lib', 'site-packages')
    for _sub in ['nvidia/cublas/bin', 'nvidia/cudnn/bin', 'nvidia/cuda_runtime/bin']:
        _dll_path = os.path.join(_site_packages, _sub.replace('/', os.sep))
        if os.path.exists(_dll_path):
            os.add_dll_directory(_dll_path)
            os.environ['PATH'] = _dll_path + os.pathsep + os.environ['PATH']

import onnxruntime as ort

# ──────────────────────────────────────────────────────────────────────────────
# 2. Cấu hình & Đường dẫn
# ──────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR       = Path(__file__).resolve().parent
# Thư mục models/ ở gốc dự án (3 cấp trên preview_result/)
MODEL_DIR        = SCRIPT_DIR.parent.parent.parent / 'Drowsiness_Edge-ai/models'
EXTRACTOR_ONNX   = str(MODEL_DIR / 'dms_feature_extractor.onnx')
CLASSIFIER_ONNX  = str(MODEL_DIR / 'dms_tcn_classifier.onnx')
LANDMARK_PATH    = str(SCRIPT_DIR / 'face_landmarker.task')

SEQ_LEN           = 16
IMG_SIZE          = 224
TARGET_FPS        = 5
SAMPLE_INTERVAL   = 1.0 / TARGET_FPS   # 0.2s / sample

# ── Ngưỡng & tham số — đồng bộ 100% với run_webcam_pytorch.py ──
DROWSY_THRESHOLD  = 0.55   # Ngưỡng kích hoạt Drowsy của Model
MAR_THRESHOLD     = 0.50   # Ngưỡng ngáp: MAR >= 0.50 (miệng mở lớn)
ALPHA             = 0.65   # Tốc độ đáp ứng bộ lọc EMA
YAWN_COOLDOWN_SEC = 3.0    # Thời gian duy trì trạng thái ngáp sau khi ngậm miệng (giây)
last_yawn_time    = 0.0

# ──────────────────────────────────────────────────────────────────────────────
# 3. Tải face_landmarker.task nếu chưa có
# ──────────────────────────────────────────────────────────────────────────────
if not os.path.exists(LANDMARK_PATH):
    print("⏳ Đang tải mô hình face_landmarker.task...")
    urllib.request.urlretrieve(
        'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
        LANDMARK_PATH
    )

# ──────────────────────────────────────────────────────────────────────────────
# 4. Khởi tạo ONNX Runtime Engine (tự chọn provider tối ưu nhất có sẵn)
# ──────────────────────────────────────────────────────────────────────────────
for _onnx_path in [EXTRACTOR_ONNX, CLASSIFIER_ONNX]:
    if not os.path.exists(_onnx_path):
        raise FileNotFoundError(
            f"❌ Không tìm thấy file ONNX: {_onnx_path}\n"
            "Vui lòng chạy export_to_onnx.py trước để xuất mô hình."
        )

_available = ort.get_available_providers()
if 'CUDAExecutionProvider' in _available:
    _providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    print("🚀 ONNX Runtime Provider: CUDAExecutionProvider (GPU NVIDIA)")
elif 'DmlExecutionProvider' in _available:
    _providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
    print("🚀 ONNX Runtime Provider: DmlExecutionProvider (DirectML / AMD / Intel GPU)")
else:
    _providers = ['CPUExecutionProvider']
    print("⚠️  ONNX Runtime Provider: CPUExecutionProvider")

session_extractor  = ort.InferenceSession(EXTRACTOR_ONNX,  providers=_providers)
session_classifier = ort.InferenceSession(CLASSIFIER_ONNX, providers=_providers)

# ──────────────────────────────────────────────────────────────────────────────
# 5. MediaPipe Extractor & MAR Calculator
#    Logic clone 100% từ run_webcam_pytorch.py
# ──────────────────────────────────────────────────────────────────────────────
class FaceRegionExtractor:
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

    def __init__(self):
        base_options = BaseOptions(model_asset_path=LANDMARK_PATH)
        options = FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.detector = FaceLandmarker.create_from_options(options)

    def _crop_region(self, img: np.ndarray, landmarks, indices: list, pad: float = 0.3) -> np.ndarray:
        h, w = img.shape[:2]
        pts  = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices]
        x1   = max(0, min(p[0] for p in pts) - int(w * pad))
        x2   = min(w, max(p[0] for p in pts) + int(w * pad))
        y1   = max(0, min(p[1] for p in pts) - int(h * pad))
        y2   = min(h, max(p[1] for p in pts) + int(h * pad))
        crop = img[y1:y2, x1:x2]
        return crop if crop.size > 0 else img

    def _calc_mar(self, lms, w: int, h: int) -> float:
        """Mouth Aspect Ratio — đo độ mở miệng chuẩn hóa theo chiều rộng."""
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

    def extract(self, img_rgb: np.ndarray):
        """
        Trả về: (face_crop, eye_crop, mouth_crop, mar)
        Nếu không phát hiện mặt: (None, None, None, 0.0)
        """
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        results  = self.detector.detect(mp_image)
        if not results.face_landmarks:
            return None, None, None, 0.0

        lms  = results.face_landmarks[0]
        h, w = img_rgb.shape[:2]

        mar = self._calc_mar(lms, w, h)

        xs = [lm.x * w for lm in lms]
        ys = [lm.y * h for lm in lms]
        x1 = max(0, int(min(xs)) - 20)
        x2 = min(w, int(max(xs)) + 20)
        y1 = max(0, int(min(ys)) - 20)
        y2 = min(h, int(max(ys)) + 20)
        face  = img_rgb[y1:y2, x1:x2] if (y2 > y1 and x2 > x1) else img_rgb
        eyes  = self._crop_region(img_rgb, lms, self.LEFT_EYE + self.RIGHT_EYE, pad=0.15)
        mouth = self._crop_region(img_rgb, lms, self.MOUTH, pad=0.2)
        return face, eyes, mouth, mar

extractor = FaceRegionExtractor()

# ──────────────────────────────────────────────────────────────────────────────
# 6. Pure-numpy Preprocessing (Edge-optimized — không cần torch / Albumentations)
#    Kết quả tương đương: A.Resize + A.Normalize(ImageNet mean/std) + channel-first
# ──────────────────────────────────────────────────────────────────────────────
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def preprocess(img_rgb: np.ndarray) -> np.ndarray:
    """
    Chuyển crop RGB (bất kỳ kích thước) → ONNX input (1, 3, 224, 224) float32.
    Chỉ dùng OpenCV + numpy; không cần PyTorch hay Albumentations.
    """
    img = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))                       # (224, 224, 3) uint8
    img = img.astype(np.float32) / 255.0                                   # [0.0, 1.0]
    img = (img - _IMAGENET_MEAN) / _IMAGENET_STD                           # ImageNet normalize
    return img.transpose(2, 0, 1)[np.newaxis].astype(np.float32)           # (1, 3, H, W)

# ──────────────────────────────────────────────────────────────────────────────
# 7. Luồng Camera & Phân Loại 3 Cấp Độ
#    Logic phân loại clone 100% từ run_webcam_pytorch.py
# ──────────────────────────────────────────────────────────────────────────────
# Thử CAP_DSHOW trước (Windows), fallback về backend mặc định (Linux/Jetson)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("❌ Không thể kết nối với Webcam (index 0). Vui lòng kiểm tra lại camera.")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

feature_buffer       = deque(maxlen=SEQ_LEN)
last_sample_time     = time.time()
smoothed_drowsy_prob = 0.0
raw_drowsy_prob      = 0.0
current_mar          = 0.0
level_id             = 1
region_weights       = [0.33, 0.33, 0.33]

label_text = "Buffering (3.2s)..."
color      = (255, 255, 255)
prev_time  = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()
    frame_rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_crop, eye_crop, mouth_crop, mar = extractor.extract(frame_rgb)

    if face_crop is not None:
        current_mar = mar

        # ── Cập nhật yawn timer ở mọi frame (không bị miss khi miệng mở nhanh) ──
        if current_mar >= MAR_THRESHOLD:
            last_yawn_time = current_time

        if current_time - last_sample_time >= SAMPLE_INTERVAL:
            last_sample_time = current_time

            # ── Tiền xử lý: pure numpy, không cần torch ──
            f_np = preprocess(face_crop)
            e_np = preprocess(eye_crop)
            m_np = preprocess(mouth_crop)

            # ── Bước 1: Trích xuất đặc trưng qua ONNX Feature Extractor ──
            frame_feature, attn_w = session_extractor.run(
                None,
                {'face_input': f_np, 'eye_input': e_np, 'mouth_input': m_np}
            )
            region_weights = attn_w[0].tolist()
            feature_buffer.append(frame_feature[0])   # shape: (256,)

            # ── Bước 2: Phân loại chuỗi thời gian khi đủ 16 frames ──
            if len(feature_buffer) == SEQ_LEN:
                # (1, 16, 256) — giống hệt feat_seq trong PyTorch
                feat_seq = np.stack(list(feature_buffer), axis=0)[np.newaxis].astype(np.float32)
                probs    = session_classifier.run(
                    None, {'sequence_features': feat_seq}
                )[0][0]   # shape: (2,) — [prob_drowsy, prob_alert]

                raw_drowsy_prob      = float(probs[0])
                smoothed_drowsy_prob = ALPHA * raw_drowsy_prob + (1.0 - ALPHA) * smoothed_drowsy_prob

                # ── Phân loại 3 Levels — đồng bộ 100% với PyTorch version ──
                is_recently_yawning = (current_time - last_yawn_time) <= YAWN_COOLDOWN_SEC

                if smoothed_drowsy_prob >= DROWSY_THRESHOLD:
                    if current_mar >= MAR_THRESHOLD or is_recently_yawning:
                        # Đang ngáp HOẶC vừa ngáp xong đang chờ Model xả bộ đệm
                        level_id   = 2
                        label_text = f"LEVEL 2: BUON NGU ({smoothed_drowsy_prob*100:.1f}% | MAR: {current_mar:.2f})"
                        color      = (0, 165, 255)   # Cam
                    else:
                        # Model báo buồn ngủ và KHÔNG có hành vi ngáp gần đây → Ngủ gật thực sự
                        level_id   = 0
                        label_text = f"LEVEL 0: NGU GAT ({smoothed_drowsy_prob*100:.1f}%)"
                        color      = (0, 0, 255)     # Đỏ
                else:
                    # Model đã hạ về dưới ngưỡng an toàn → Tỉnh táo hoàn toàn
                    level_id            = 1
                    smoothed_alert_prob = 1.0 - smoothed_drowsy_prob
                    label_text          = f"LEVEL 1: ALERT ({smoothed_alert_prob*100:.1f}%)"
                    color               = (0, 255, 0)   # Xanh lá
    else:
        label_text = "No Face Detected"
        color      = (0, 255, 255)

    fps       = 1.0 / (current_time - prev_time)
    prev_time = current_time

    # ── Hiển thị Trạng thái lên Frame (đồng bộ layout với PyTorch version) ──
    h, w    = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, h - 120), (450, h - 10), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    cv2.putText(frame, label_text,
                (20, h - 85), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
    cv2.putText(frame, f"FPS: {fps:.1f} | Model Drowsy: {smoothed_drowsy_prob*100:.1f}% | MAR: {current_mar:.2f}",
                (20, h - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f"Attn -> Face: {region_weights[0]:.2f} | Eye: {region_weights[1]:.2f} | Mouth: {region_weights[2]:.2f}",
                (20, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1)

    cv2.imshow('DMS Real-Time Demo (ONNX Runtime)', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()