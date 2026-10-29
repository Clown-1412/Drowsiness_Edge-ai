"""
SharedState — Đối tượng giao tiếp giữa các thread.
Sử dụng threading.Lock đơn giản.
"""
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class SharedState:
    """
    Thread-safe shared state cho giao tiếp giữa:
      - Camera Thread  →  Main Thread (Detection)
      - Main Thread    →  Inference Thread
      - Inference Thread → Main Thread (Display / Payload)
    """
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # ── Camera → Detection / Display ──
    latest_frame: Optional[np.ndarray] = None
    frame_timestamp: float = 0.0

    # ── Detection → Inference (cập nhật bởi main thread mỗi frame) ──
    face_crop:     Optional[np.ndarray] = None
    eye_crop:      Optional[np.ndarray] = None
    mouth_crop:    Optional[np.ndarray] = None
    current_mar:   float = 0.0
    face_detected: bool  = False
    last_yawn_time: float = 0.0     # Cập nhật mỗi frame khi MAR >= ngưỡng

    # ── Inference → Display / Payload (cập nhật bởi inference thread) ──
    level_id:             int   = 1            # 0=Ngủ gật, 1=Tỉnh, 2=Buồn ngủ
    smoothed_drowsy_prob: float = 0.0
    raw_drowsy_prob:      float = 0.0
    region_weights:       List[float] = field(default_factory=lambda: [0.33, 0.33, 0.33])
    is_recently_yawning:  bool  = False
    buffer_ready:         bool  = False        # True khi đủ 16 frames
    label_text:           str   = "Buffering (3.2s)..."
    label_color:          Tuple[int, int, int] = (255, 255, 255)

    # ── Hệ thống ──
    inference_fps: float = 0.0
    camera_fps:    float = 0.0

    # ── ECU Alert Feedback (CAN 0x201 — từ ESP32 gửi về) ──
    alert_ecu_connected:  bool  = False   # True khi nhận được gói 0x201 trong khoảng watchdog
    alert_ecu_state:      int   = 0       # Trạng thái cảnh báo đang thực thi (0=IDLE, 1=DROWSY, 2=DOZING, 3=NO_FACE)
    alert_ecu_hw_health:  int   = 0x07    # Bitfield sức khỏe phần cứng (bit0=SD, bit1=I2S, bit2=Buzzer)
    alert_ecu_error_code: int   = 0       # Mã lỗi chi tiết từ ESP32
    last_feedback_time:   float = 0.0     # time.time() lần cuối nhận gói 0x201

    def get_telemetry_snapshot(self) -> dict:
        """
        Sao chép nhanh toàn bộ telemetry + CAN + status vào dict thuần Python.
        Acquire lock cực ngắn (< 50µs) — chỉ copy primitive types, không copy numpy arrays.
        Bridge Worker gọi hàm này, KHÔNG trực tiếp đọc các field.
        """
        with self.lock:
            return {
                'timestamp': time.time(),
                'telemetry': {
                    'drowsy_prob': self.smoothed_drowsy_prob,
                    'raw_drowsy_prob': self.raw_drowsy_prob,
                    'mar': self.current_mar,
                    'region_weights': list(self.region_weights),
                    'level_id': self.level_id,
                    'level_label': self.label_text,
                    'face_detected': self.face_detected,
                    'buffer_ready': self.buffer_ready,
                    'is_yawning': self.is_recently_yawning,
                    'inference_fps': self.inference_fps,
                    'camera_fps': self.camera_fps,
                },
                'can_network': {
                    'rx_0x201': {
                        'connected': self.alert_ecu_connected,
                        'executing_state': self.alert_ecu_state,
                        'hw_health': {
                            'sd_card': bool(self.alert_ecu_hw_health & 0x01),
                            'i2s_dac': bool(self.alert_ecu_hw_health & 0x02),
                            'buzzer': bool(self.alert_ecu_hw_health & 0x04),
                        },
                        'error_code': self.alert_ecu_error_code,
                    },
                },
                'system_health': {
                    'ecu_alert_online': self.alert_ecu_connected,
                    'camera_fps': self.camera_fps,
                },
            }
