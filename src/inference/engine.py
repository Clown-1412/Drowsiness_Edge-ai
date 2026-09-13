"""
Thread 3: AI Inference — Quản lý ONNX sessions + Feature Buffer + TCN.
Chạy ở tần suất TARGET_FPS (5 Hz = mỗi 200ms).
Đọc detection results từ SharedState, chạy ONNX, cập nhật kết quả.

Logic inference clone 100% từ run_webcam_onnx.py L225-250.
"""
import os
import threading
import time
from collections import deque

import numpy as np

from src.config import (
    EXTRACTOR_ONNX, CLASSIFIER_ONNX,
    SEQ_LEN, SAMPLE_INTERVAL,
    DROWSY_THRESHOLD, MAR_THRESHOLD, ALPHA, YAWN_COOLDOWN_SEC,
    setup_cuda_dlls,
)
from src.inference.preprocessor import preprocess
from src.classifier.state_machine import DriverStateMachine
from src.shared_state import SharedState

# ── Nạp DLL CUDA TRƯỚC khi import onnxruntime ──
setup_cuda_dlls()
import onnxruntime as ort


class InferenceEngine(threading.Thread):
    """
    Thread chạy inference ONNX ở tần suất 5 FPS.

    Pipeline (clone 100% từ demo gốc):
      1. Đọc face_crop, eye_crop, mouth_crop từ SharedState
      2. Preprocess (numpy) → (1, 3, 224, 224)
      3. Feature Extractor ONNX → feature vector (256,)
      4. Thêm vào feature_buffer (deque maxlen=16)
      5. Khi đủ 16 frames → TCN Classifier → probs (2,)
      6. EMA smoothing + 3-Level classification
      7. Cập nhật kết quả vào SharedState
    """

    def __init__(self, shared_state: SharedState):
        super().__init__(daemon=True, name="InferenceThread")
        self.shared_state = shared_state
        self._stopped = False

        # Feature buffer — giống hệt demo gốc (deque maxlen=16)
        self.feature_buffer = deque(maxlen=SEQ_LEN)

        # State machine — dùng cùng ngưỡng với demo gốc
        self.state_machine = DriverStateMachine(
            drowsy_threshold=DROWSY_THRESHOLD,
            mar_threshold=MAR_THRESHOLD,
            alpha=ALPHA,
            yawn_cooldown_sec=YAWN_COOLDOWN_SEC,
        )

        # Khởi tạo ONNX sessions
        self._init_sessions()

    def _init_sessions(self):
        """
        Auto-chọn provider: CUDA → DML → CPU.
        Clone logic demo gốc L70-82.
        """
        for path in [EXTRACTOR_ONNX, CLASSIFIER_ONNX]:
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"❌ Không tìm thấy file ONNX: {path}\n"
                    "Vui lòng chạy export_to_onnx.py trước để xuất mô hình."
                )

        available = ort.get_available_providers()
        if 'CUDAExecutionProvider' in available:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            print("🚀 ONNX Runtime Provider: CUDAExecutionProvider (GPU NVIDIA)")
        elif 'DmlExecutionProvider' in available:
            providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
            print("🚀 ONNX Runtime Provider: DmlExecutionProvider (DirectML)")
        else:
            providers = ['CPUExecutionProvider']
            print("⚠️  ONNX Runtime Provider: CPUExecutionProvider")

        self.session_extractor = ort.InferenceSession(
            EXTRACTOR_ONNX, providers=providers
        )
        self.session_classifier = ort.InferenceSession(
            CLASSIFIER_ONNX, providers=providers
        )

    def run(self):
        """Vòng lặp inference — chạy trên thread riêng."""
        print("🧠 Inference thread started")
        last_inference_time = time.time()

        while not self._stopped:
            current_time = time.time()
            elapsed = current_time - last_inference_time

            # ── Đợi đủ SAMPLE_INTERVAL (200ms) ──
            if elapsed < SAMPLE_INTERVAL:
                time.sleep(0.01)  # Sleep ngắn tránh busy-wait
                continue

            last_inference_time = current_time

            # ── Đọc crops từ SharedState ──
            with self.shared_state.lock:
                face     = self.shared_state.face_crop
                eye      = self.shared_state.eye_crop
                mouth    = self.shared_state.mouth_crop
                mar      = self.shared_state.current_mar
                detected = self.shared_state.face_detected
                last_yawn = self.shared_state.last_yawn_time

            # Bỏ qua nếu không có mặt
            if not detected or face is None or eye is None or mouth is None:
                continue

            # ── Tiền xử lý: pure numpy — clone 100% demo gốc L229-231 ──
            f_np = preprocess(face)
            e_np = preprocess(eye)
            m_np = preprocess(mouth)

            # ── Feature Extraction — clone 100% demo gốc L234-239 ──
            frame_feature, attn_w = self.session_extractor.run(
                None,
                {'face_input': f_np, 'eye_input': e_np, 'mouth_input': m_np}
            )
            region_weights = attn_w[0].tolist()
            self.feature_buffer.append(frame_feature[0])   # shape: (256,)

            # ── TCN Classification khi đủ 16 frames — clone 100% L242-250 ──
            if len(self.feature_buffer) == SEQ_LEN:
                # (1, 16, 256) — giống hệt feat_seq trong demo gốc
                feat_seq = np.stack(
                    list(self.feature_buffer), axis=0
                )[np.newaxis].astype(np.float32)

                probs = self.session_classifier.run(
                    None, {'sequence_features': feat_seq}
                )[0][0]   # shape: (2,) — [prob_drowsy, prob_alert]

                raw_drowsy_prob = float(probs[0])

                # Tính is_recently_yawning — sử dụng last_yawn_time
                # từ SharedState (được main thread cập nhật mỗi frame)
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
                # Chưa đủ buffer — chỉ cập nhật region weights
                with self.shared_state.lock:
                    self.shared_state.region_weights = region_weights

    def stop(self):
        """Dừng thread."""
        self._stopped = True
