"""
SharedState — Đối tượng giao tiếp giữa các thread.
Sử dụng threading.Lock đơn giản cho prototype PC.
"""
import threading
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
