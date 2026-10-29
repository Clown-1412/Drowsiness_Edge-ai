"""
Thread 3: AI Inference — Thuần RKNN NPU (Feature Extractor + TCN Classifier).
Chạy ở tần suất TARGET_FPS (5 Hz = mỗi 200ms).
Đọc detection results từ SharedState, chạy inference trên phần cứng NPU RK3588, cập nhật kết quả.

- Feature Extractor: RKNN NPU (RK3588) nhận uint8 NHWC, chuẩn hóa ở phần cứng.
- TCN Classifier: RKNN NPU (RK3588) nhận sequence features (1, 16, 256) float32.
- 100% RKNN NPU, không fallback ONNX, không phụ thuộc CUDA/GPU.
"""
import os
import sys
import threading
import time
from collections import deque

if sys.platform == 'win32':
    try:
        if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
            sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import numpy as np

from src.config import (
    EXTRACTOR_RKNN, CLASSIFIER_RKNN,
    SEQ_LEN, SAMPLE_INTERVAL,
    DROWSY_THRESHOLD, MAR_THRESHOLD, ALPHA, YAWN_COOLDOWN_SEC,
)
from src.inference.preprocessor import preprocess_crop_uint8
from src.classifier.state_machine import DriverStateMachine
from src.shared_state import SharedState

try:
    from rknnlite.api import RKNNLite
except ImportError:
    RKNNLite = None


class InferenceEngine(threading.Thread):
    """
    Thread chạy inference trên NPU RK3588 ở tần suất 5 FPS.

    Pipeline:
      1. Đọc face_crop, eye_crop, mouth_crop từ SharedState
      2. Preprocess uint8 NHWC → (1, 224, 224, 3)
      3. Feature Extractor RKNN NPU → feature vector (256,), attention weights (3,)
      4. Thêm vào feature_buffer (deque maxlen=16)
      5. Khi đủ 16 frames → TCN Classifier RKNN NPU → probs (2,)
      6. EMA smoothing + 3-Level classification
      7. Cập nhật kết quả vào SharedState
    """

    def __init__(self, shared_state: SharedState):
        super().__init__(daemon=True, name="InferenceThread")
        self.shared_state = shared_state
        self._stopped = False

        # Feature buffer (deque maxlen=16)
        self.feature_buffer = deque(maxlen=SEQ_LEN)

        # State machine (3 Levels + EMA smoothing)
        self.state_machine = DriverStateMachine(
            drowsy_threshold=DROWSY_THRESHOLD,
            mar_threshold=MAR_THRESHOLD,
            alpha=ALPHA,
            yawn_cooldown_sec=YAWN_COOLDOWN_SEC,
        )

        self.rknn_extractor = None
        self.rknn_classifier = None

        # Khởi tạo RKNN NPU sessions
        self._init_rknn_sessions()

    def _init_rknn_sessions(self):
        """
        Khởi tạo 2 mô hình RKNN trên phần cứng NPU RK3588:
          - Feature Extractor (NPU Core 0)
          - TCN Classifier (NPU Core 1)
        """
        if RKNNLite is None:
            raise RuntimeError(
                "❌ Thư viện 'rknnlite' không khả dụng!\n"
                "Pipeline yêu cầu chạy trực tiếp trên RK3588 với rknn-toolkit-lite2."
            )

        # ── 1. Khởi tạo Feature Extractor RKNN ──
        if not os.path.exists(EXTRACTOR_RKNN):
            raise FileNotFoundError(
                f"❌ Không tìm thấy model RKNN Feature Extractor: {EXTRACTOR_RKNN}\n"
                "Vui lòng chạy script scripts/convert_onnx_to_rknn_uint8.py để biên dịch."
            )

        self.rknn_extractor = RKNNLite()
        ret = self.rknn_extractor.load_rknn(EXTRACTOR_RKNN)
        if ret != 0:
            raise RuntimeError(f"load_rknn Extractor thất bại (mã lỗi: {ret})")

        core_0 = getattr(RKNNLite, 'NPU_CORE_0', 1)
        ret = self.rknn_extractor.init_runtime(core_mask=core_0)
        if ret != 0:
            raise RuntimeError(f"init_runtime Extractor thất bại (mã lỗi: {ret})")
        print("🚀 Feature Extractor: RKNN NPU (NPU_CORE_0) sẵn sàng")

        # ── 2. Khởi tạo TCN Classifier RKNN ──
        if not os.path.exists(CLASSIFIER_RKNN):
            raise FileNotFoundError(
                f"❌ Không tìm thấy model RKNN TCN Classifier: {CLASSIFIER_RKNN}\n"
                "Vui lòng chạy script scripts/convert_classifier_to_rknn.py để biên dịch."
            )

        self.rknn_classifier = RKNNLite()
        ret = self.rknn_classifier.load_rknn(CLASSIFIER_RKNN)
        if ret != 0:
            raise RuntimeError(f"load_rknn Classifier thất bại (mã lỗi: {ret})")

        core_1 = getattr(RKNNLite, 'NPU_CORE_1', getattr(RKNNLite, 'NPU_CORE_AUTO', 0))
        ret = self.rknn_classifier.init_runtime(core_mask=core_1)
        if ret != 0:
            # Fallback về Core 0 nếu core 1 bận
            ret = self.rknn_classifier.init_runtime(core_mask=core_0)
            if ret != 0:
                raise RuntimeError(f"init_runtime Classifier thất bại (mã lỗi: {ret})")

        print("🧠 TCN Classifier: RKNN NPU sẵn sàng")

    def run(self):
        """Vòng lặp inference — chạy trên thread riêng."""
        print("🧠 RKNN Inference thread started (100% NPU)")
        last_inference_time = time.time()

        while not self._stopped:
            current_time = time.time()
            elapsed = current_time - last_inference_time

            # ── Đợi đủ SAMPLE_INTERVAL (200ms = 5Hz) ──
            if elapsed < SAMPLE_INTERVAL:
                time.sleep(0.01)
                continue

            last_inference_time = current_time

            # ── Đọc crops từ SharedState ──
            with self.shared_state.lock:
                face      = self.shared_state.face_crop
                eye       = self.shared_state.eye_crop
                mouth     = self.shared_state.mouth_crop
                mar       = self.shared_state.current_mar
                detected  = self.shared_state.face_detected
                last_yawn = self.shared_state.last_yawn_time

            # Bỏ qua nếu không có mặt
            if not detected or face is None or eye is None or mouth is None:
                continue

            # ── Tiền xử lý uint8 NHWC (1, 224, 224, 3) ──
            f_uint8 = preprocess_crop_uint8(face)
            e_uint8 = preprocess_crop_uint8(eye)
            m_uint8 = preprocess_crop_uint8(mouth)

            # ── Feature Extraction trên RKNN NPU ──
            outputs = self.rknn_extractor.inference(inputs=[f_uint8, e_uint8, m_uint8])
            frame_feature = outputs[0]    # [1, 256] float32
            attn_w = outputs[1]           # [1, 3]  float32
            region_weights = attn_w[0].tolist()

            self.feature_buffer.append(frame_feature[0])   # shape: (256,)

            # ── TCN Classification khi đủ 16 frames ──
            if len(self.feature_buffer) == SEQ_LEN:
                feat_seq = np.stack(
                    list(self.feature_buffer), axis=0
                )[np.newaxis].astype(np.float32)  # (1, 16, 256)

                cls_outputs = self.rknn_classifier.inference(inputs=[feat_seq])
                probs = cls_outputs[0][0]   # shape: (2,) — [prob_drowsy, prob_alert]

                raw_drowsy_prob = float(probs[0])

                # Tính is_recently_yawning dựa trên last_yawn_time
                is_recently_yawning = (
                    (current_time - last_yawn) <= YAWN_COOLDOWN_SEC
                )

                # ── State machine: EMA + 3-Level classification ──
                state = self.state_machine.update(
                    raw_drowsy_prob, mar, is_recently_yawning
                )

                # ── Cập nhật SharedState ──
                with self.shared_state.lock:
                    self.shared_state.level_id             = state.level_id
                    self.shared_state.smoothed_drowsy_prob  = state.smoothed_drowsy_prob
                    self.shared_state.raw_drowsy_prob       = raw_drowsy_prob
                    self.shared_state.region_weights        = region_weights
                    self.shared_state.is_recently_yawning   = is_recently_yawning
                    self.shared_state.buffer_ready          = True
                    self.shared_state.label_text            = state.label_text
                    self.shared_state.label_color           = state.color

            else:
                # Chưa đủ buffer — cập nhật region weights
                with self.shared_state.lock:
                    self.shared_state.region_weights = region_weights

    def stop(self):
        """Dừng thread và giải phóng toàn bộ tài nguyên NPU."""
        self._stopped = True
        if self.rknn_extractor is not None:
            try:
                self.rknn_extractor.release()
                print("🧹 RKNN Extractor runtime đã giải phóng")
            except Exception:
                pass
            self.rknn_extractor = None

        if self.rknn_classifier is not None:
            try:
                self.rknn_classifier.release()
                print("🧹 RKNN Classifier runtime đã giải phóng")
            except Exception:
                pass
            self.rknn_classifier = None
