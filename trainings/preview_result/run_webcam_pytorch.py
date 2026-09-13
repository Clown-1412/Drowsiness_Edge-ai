import os
import time
import math
from collections import deque
from pathlib import Path
import urllib.request
import cv2
import numpy as np
import torch
import torch.nn as nn
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions

# ── Cấu hình & Tối ưu GPU ──
SCRIPT_DIR       = Path(__file__).resolve().parent
MODEL_PATH       = str(SCRIPT_DIR / 'best_model.pth')
LANDMARK_PATH    = str(SCRIPT_DIR / 'face_landmarker.task')
SEQ_LEN          = 16
IMG_SIZE         = 224
TARGET_FPS       = 5
SAMPLE_INTERVAL  = 1.0 / TARGET_FPS  # 0.2s / sample
DROWSY_THRESHOLD = 0.56              # Ngưỡng kích hoạt Drowsy của Model
MAR_THRESHOLD    = 0.50              # Ngưỡng ngáp: MAR >= 0.50 (miệng mở lớn)
ALPHA            = 0.65              # Tốc độ đáp ứng bộ lọc EMA
# Thời gian duy trì trạng thái ngáp sau khi ngậm miệng (giây)
YAWN_COOLDOWN_SEC = 3.0  
last_yawn_time = 0.0

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if DEVICE.type == 'cuda':
    torch.backends.cudnn.benchmark = True
    print(f"🚀 Đang chạy trên GPU: {torch.cuda.get_device_name(0)}")
else:
    torch.set_num_threads(4)
    print("⚠️ Không tìm thấy GPU, đang chạy trên CPU.")

if not os.path.exists(LANDMARK_PATH):
    print("⏳ Đang tải mô hình face_landmarker.task...")
    urllib.request.urlretrieve(
        'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
        LANDMARK_PATH
    )

# ── 1. Kiến trúc Model ──
class TemporalAttention(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden, hidden // 4),
            nn.Tanh(),
            nn.Linear(hidden // 4, 1)
        )

    def forward(self, x):
        weights = torch.softmax(self.attn(x), dim=1)
        return (x * weights).sum(dim=1)

class MultiRegionCNNTCN(nn.Module):
    def __init__(self, backbone='tf_efficientnet_lite2', hidden=256, num_classes=2, seq_len=16):
        super().__init__()
        self.seq_len = seq_len
        self.cnn_face  = timm.create_model(backbone, pretrained=False, num_classes=0)
        self.cnn_eye   = timm.create_model(backbone, pretrained=False, num_classes=0)
        self.cnn_mouth = timm.create_model(backbone, pretrained=False, num_classes=0)
        feat_dim = self.cnn_face.num_features

        self.region_attn = nn.Sequential(
            nn.Linear(feat_dim * 3, 3),
            nn.Softmax(dim=1)
        )

        self.fusion = nn.Sequential(
            nn.Linear(feat_dim * 3, hidden),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.tcn = nn.Sequential(
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(0.2),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(0.2),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=4, dilation=4),
            nn.BatchNorm1d(hidden), nn.ReLU(),
        )

        self.temporal_attn = TemporalAttention(hidden)
        # Giữ lại image_head để tương thích với checkpoint gốc khi load_state_dict(strict=True)
        self.image_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.classifier = nn.Linear(hidden, num_classes)

    def extract_regions(self, face, eye, mouth):
        f_feat = self.cnn_face(face)
        e_feat = self.cnn_eye(eye)
        m_feat = self.cnn_mouth(mouth)
        combined = torch.cat([f_feat, e_feat, m_feat], dim=1)
        weights  = self.region_attn(combined)
        weighted = torch.cat([
            f_feat * weights[:, 0:1],
            e_feat * weights[:, 1:2],
            m_feat * weights[:, 2:3],
        ], dim=1)
        return self.fusion(weighted), weights

    def forward_sequence_from_features(self, feat_seq):
        feats = feat_seq.permute(0, 2, 1)
        feats = self.tcn(feats)
        feats = feats.permute(0, 2, 1)
        feats = self.temporal_attn(feats)
        return self.classifier(feats)

# ── 2. Nạp Checkpoint ──
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"❌ Không tìm thấy checkpoint tại: {MODEL_PATH}\n"
        "Vui lòng đặt file weights 'best_model.pth' vào cùng thư mục với script trước khi chạy."
    )

model = MultiRegionCNNTCN(seq_len=SEQ_LEN).to(DEVICE)
state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
if 'model_state_dict' in state_dict:
    model.load_state_dict(state_dict['model_state_dict'], strict=True)
else:
    model.load_state_dict(state_dict, strict=True)
model.eval()

# ── 3. MediaPipe Extractor & MAR Calculator ──
class FaceRegionExtractor:
    LEFT_EYE  = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE = [362, 385, 387, 263, 373, 380]
    MOUTH     = [61, 291, 39, 181, 0, 17, 269, 405]
    
    # Landmark tính MAR chuẩn: mép trái (61), mép phải (291), môi trên/dưới
    MOUTH_MAR = {
        'left': 61, 'right': 291,
        'top_1': 39, 'bot_1': 181,
        'top_2': 0,  'bot_2': 17,
        'top_3': 269, 'bot_3': 405
    }

    def __init__(self):
        base_options = BaseOptions(model_asset_path=LANDMARK_PATH)
        options = FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.detector = FaceLandmarker.create_from_options(options)

    def _crop_region(self, img, landmarks, indices, pad=0.3):
        h, w = img.shape[:2]
        pts  = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices]
        x1   = max(0, min(p[0] for p in pts) - int(w * pad))
        x2   = min(w, max(p[0] for p in pts) + int(w * pad))
        y1   = max(0, min(p[1] for p in pts) - int(h * pad))
        y2   = min(h, max(p[1] for p in pts) + int(h * pad))
        crop = img[y1:y2, x1:x2]
        return crop if crop.size > 0 else img

    def _calc_mar(self, lms, w, h):
        def dist(idx1, idx2):
            p1 = (lms[idx1].x * w, lms[idx1].y * h)
            p2 = (lms[idx2].x * w, lms[idx2].y * h)
            return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

        d_vert1 = dist(self.MOUTH_MAR['top_1'], self.MOUTH_MAR['bot_1'])
        d_vert2 = dist(self.MOUTH_MAR['top_2'], self.MOUTH_MAR['bot_2'])
        d_vert3 = dist(self.MOUTH_MAR['top_3'], self.MOUTH_MAR['bot_3'])
        d_horiz = dist(self.MOUTH_MAR['left'], self.MOUTH_MAR['right'])

        if d_horiz == 0:
            return 0.0
        return (d_vert1 + d_vert2 + d_vert3) / (3.0 * d_horiz)

    def extract(self, img_rgb):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        results  = self.detector.detect(mp_image)
        if not results.face_landmarks:
            return None, None, None, 0.0

        lms  = results.face_landmarks[0]
        h, w = img_rgb.shape[:2]
        
        mar = self._calc_mar(lms, w, h)

        xs   = [lm.x * w for lm in lms]
        ys   = [lm.y * h for lm in lms]
        x1   = max(0, int(min(xs)) - 20)
        x2   = min(w, int(max(xs)) + 20)
        y1   = max(0, int(min(ys)) - 20)
        y2   = min(h, int(max(ys)) + 20)
        face = img_rgb[y1:y2, x1:x2] if (y2 > y1 and x2 > x1) else img_rgb
        eyes  = self._crop_region(img_rgb, lms, self.LEFT_EYE + self.RIGHT_EYE, pad=0.15)
        mouth = self._crop_region(img_rgb, lms, self.MOUTH, pad=0.2)
        return face, eyes, mouth, mar

extractor = FaceRegionExtractor()

transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

# ── 4. Luồng Camera & Phân Loại 3 Cấp Độ ──
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("❌ Không thể kết nối với Webcam (index 0). Vui lòng kiểm tra lại camera.")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

feature_buffer = deque(maxlen=SEQ_LEN)
last_sample_time = time.time()
smoothed_drowsy_prob = 0.0
raw_drowsy_prob = 0.0
current_mar = 0.0
level_id = 1
region_weights = [0.33, 0.33, 0.33]

label_text = "Buffering (3.2s)..."
color = (255, 255, 255)
prev_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_crop, eye_crop, mouth_crop, mar = extractor.extract(frame_rgb)

    if face_crop is not None:
        current_mar = mar
        if current_time - last_sample_time >= SAMPLE_INTERVAL:
            last_sample_time = current_time

            f_t = transform(image=face_crop)['image'].unsqueeze(0).to(DEVICE)
            e_t = transform(image=eye_crop)['image'].unsqueeze(0).to(DEVICE)
            m_t = transform(image=mouth_crop)['image'].unsqueeze(0).to(DEVICE)

            with torch.inference_mode():
                with torch.amp.autocast(device_type=DEVICE.type, enabled=(DEVICE.type == 'cuda')):
                    frame_feature, attn_w = model.extract_regions(f_t, e_t, m_t)

                region_weights = attn_w[0].cpu().tolist()
                feature_buffer.append(frame_feature)

                if len(feature_buffer) == SEQ_LEN:
                    feat_seq = torch.cat(list(feature_buffer), dim=0).unsqueeze(0)
                    with torch.amp.autocast(device_type=DEVICE.type, enabled=(DEVICE.type == 'cuda')):
                        logits = model.forward_sequence_from_features(feat_seq)
                        probs  = torch.softmax(logits, dim=1)[0]

                    raw_drowsy_prob = probs[0].item()
                    smoothed_drowsy_prob = ALPHA * raw_drowsy_prob + (1 - ALPHA) * smoothed_drowsy_prob

                    # ── Phân loại 3 Levels Chuẩn Logic ──
                    if current_mar >= MAR_THRESHOLD:
                        last_yawn_time = current_time

                    # Kiểm tra xem có đang trong giai đoạn vừa ngáp xong hay không
                    is_recently_yawning = (current_time - last_yawn_time) <= YAWN_COOLDOWN_SEC

                    # ── Phân loại 3 Levels chống nhảy cờ ảo ──
                    if smoothed_drowsy_prob >= DROWSY_THRESHOLD:
                        if current_mar >= MAR_THRESHOLD or is_recently_yawning:
                            # Đang ngáp HOẶC vừa ngáp xong đang chờ Model xả bộ đệm
                            level_id = 2
                            label_text = f"LEVEL 2: BUON NGU ({smoothed_drowsy_prob*100:.1f}% | MAR: {current_mar:.2f})"
                            color = (0, 165, 255)  # Cam
                        else:
                            # Model báo buồn ngủ và KHÔNG hề có hành vi ngáp gần đây -> Ngủ gật thực sự
                            level_id = 0
                            label_text = f"LEVEL 0: NGU GAT ({smoothed_drowsy_prob*100:.1f}%)"
                            color = (0, 0, 255)    # Đỏ
                    else:
                        # Model đã hạ về dưới ngưỡng an toàn -> Tỉnh táo hoàn toàn
                        level_id = 1
                        smoothed_alert_prob = 1.0 - smoothed_drowsy_prob
                        label_text = f"LEVEL 1: ALERT ({smoothed_alert_prob*100:.1f}%)"
                        color = (0, 255, 0)        # Xanh lá
    else:
        label_text = "No Face Detected"
        color = (0, 255, 255)

    fps = 1.0 / (current_time - prev_time)
    prev_time = current_time

    # ── Hiển thị Trạng thái lên Frame ──
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, h - 120), (450, h - 10), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    cv2.putText(frame, label_text, (20, h - 85), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
    cv2.putText(frame, f"FPS: {fps:.1f} | Model Drowsy: {smoothed_drowsy_prob*100:.1f}% | MAR: {current_mar:.2f}",
                (20, h - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f"Attn -> Face: {region_weights[0]:.2f} | Eye: {region_weights[1]:.2f} | Mouth: {region_weights[2]:.2f}",
                (20, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1)

    cv2.imshow('DMS Real-Time Demo (GPU)', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()