"""
Thread 3: AI Inference — Feature Extractor (RKNN NPU / ONNX fallback) + TCN.
Chạy ở tần suất TARGET_FPS (5 Hz = mỗi 200ms).
Đọc detection results từ SharedState, chạy inference, cập nhật kết quả.

Feature Extractor: ưu tiên RKNN NPU (RK3588), fallback ONNX nếu không khả dụng.
TCN Classifier: luôn chạy trên ONNX CPUExecutionProvider.
Logic inference giữ nguyên 100% từ run_webcam_onnx.py L225-250.
"""
import os
import threading
import time
from collections import deque

import numpy as np

from src.config import (
    EXTRACTOR_ONNX, EXTRACTOR_RKNN, CLASSIFIER_ONNX,
    SEQ_LEN, SAMPLE_INTERVAL,
    DROWSY_THRESHOLD, MAR_THRESHOLD, ALPHA, YAWN_COOLDOWN_SEC,
    setup_cuda_dlls,
)
from src.inference.preprocessor import preprocess
from src.classifier.state_machine import DriverStateMachine
from src.shared_state import SharedState

# ── Detect RKNN availability (graceful fallback) ──
try:
    from rknnlite.api import RKNNLite
    RKNN_AVAILABLE = True
except ImportError:
    RKNN_AVAILABLE = False

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
        Khởi tạo backend suy luận:
          - Feature Extractor: RKNN NPU (ưu tiên) → ONNX fallback
          - TCN Classifier: luôn ONNX CPUExecutionProvider
        """
        self._use_rknn = False

        # ── Feature Extractor: thử RKNN NPU trước ──
        if RKNN_AVAILABLE and os.path.exists(EXTRACTOR_RKNN):
            try:
                self.rknn = RKNNLite()
                ret = self.rknn.load_rknn(EXTRACTOR_RKNN)
                if ret != 0:
                    raise RuntimeError(f"load_rknn failed (code={ret})")
                ret = self.rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
                if ret != 0:
                    raise RuntimeError(f"init_runtime failed (code={ret})")
                self._use_rknn = True
                print("🚀 Feature Extractor: RKNN NPU (NPU_CORE_0)")
            except Exception as e:
                print(f"⚠️  RKNN init thất bại: {e} — fallback về ONNX")
                self._use_rknn = False

        # ── Feature Extractor: ONNX fallback ──
        if not self._use_rknn:
            if not os.path.exists(EXTRACTOR_ONNX):
                raise FileNotFoundError(
                    f"❌ Không tìm thấy file ONNX: {EXTRACTOR_ONNX}\n"
                    "Vui lòng chạy export_to_onnx.py trước để xuất mô hình."
                )
            available = ort.get_available_providers()
            if 'CUDAExecutionProvider' in available:
                ext_providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            elif 'DmlExecutionProvider' in available:
                ext_providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
            else:
                ext_providers = ['CPUExecutionProvider']
            self.session_extractor = ort.InferenceSession(
                EXTRACTOR_ONNX, providers=ext_providers
            )
            print(f"🚀 Feature Extractor: ONNX ({ext_providers[0]})")

        # ── TCN Classifier: luôn ONNX CPU ──
        if not os.path.exists(CLASSIFIER_ONNX):
            raise FileNotFoundError(
                f"❌ Không tìm thấy file ONNX: {CLASSIFIER_ONNX}\n"
                "Vui lòng chạy export_to_onnx.py trước để xuất mô hình."
            )
        self.session_classifier = ort.InferenceSession(
            CLASSIFIER_ONNX, providers=['CPUExecutionProvider']
        )
        print("🧠 TCN Classifier: ONNX (CPUExecutionProvider)")

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

            # ── Feature Extraction — RKNN NPU hoặc ONNX fallback ──
            if self._use_rknn:
                outputs = self.rknn.inference(inputs=[f_np, e_np, m_np])
                frame_feature = outputs[0]    # [1, 256] float
                attn_w = outputs[1]           # [1, 3]  float
            else:
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
        """Dừng thread và giải phóng tài nguyên NPU (nếu có)."""
        self._stopped = True
        if self._use_rknn and hasattr(self, 'rknn'):
            try:
                self.rknn.release()
                print("🧹 RKNN runtime đã giải phóng")
            except Exception:
                pass
