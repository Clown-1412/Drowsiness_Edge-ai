"""
Thread 1: Camera I/O — Đọc frame liên tục trên thread riêng.
Giải quyết vấn đề frame lag do buffer driver (V4L2, DirectShow).
Luôn giữ frame MỚI NHẤT — main loop lấy được frame real-time nhất.
"""
import sys
import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np


class ThreadedCapture:
    """
    Đọc frame liên tục từ VideoCapture trên thread riêng.
    Luôn giữ frame MỚI NHẤT, loại bỏ frame lag.

    Tại sao cần tách thread:
      Demo gốc gọi cap.read() trong vòng lặp chính → nếu AI inference
      mất 100ms, frame tiếp theo đã cũ 100ms (stale). Thread riêng đọc
      liên tục → main loop luôn nhận frame real-time.
    """

    def __init__(self, camera_index: int = 0,
                 width: int = 640, height: int = 480):
        # Thử CAP_DSHOW trước (Windows), fallback mặc định (Linux/Jetson)
        # Clone logic từ run_webcam_onnx.py L187-189
        if sys.platform == 'win32':
            self.cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(camera_index)
        else:
            self.cap = cv2.VideoCapture(camera_index)

        if not self.cap.isOpened():
            raise RuntimeError(
                f"❌ Không thể kết nối với Webcam (index {camera_index}). "
                "Vui lòng kiểm tra lại camera."
            )

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        self._frame: Optional[np.ndarray] = None
        self._new_frame = False
        self._lock = threading.Lock()
        self._stopped = False
        self._thread = threading.Thread(
            target=self._capture_loop, daemon=True, name="CameraThread"
        )

    def start(self):
        """Bắt đầu thread đọc camera."""
        self._thread.start()

    def _capture_loop(self):
        """Vòng lặp đọc frame liên tục — chạy trên thread riêng."""
        while not self._stopped:
            ret, frame = self.cap.read()
            if ret:
                with self._lock:
                    self._frame = frame
                    self._new_frame = True
            else:
                # Không đọc được frame — sleep ngắn tránh busy-wait
                time.sleep(0.001)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Thread-safe: trả về (success, frame_mới_nhất).
        Copy frame ra để tránh race condition.
        """
        with self._lock:
            if self._frame is None:
                return False, None
            frame = self._frame.copy()
            self._new_frame = False
            return True, frame

    def stop(self):
        """Dừng thread và giải phóng camera."""
        self._stopped = True
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
        if self.cap.isOpened():
            self.cap.release()
