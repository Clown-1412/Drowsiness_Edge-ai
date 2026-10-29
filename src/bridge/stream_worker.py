"""
Stream Worker — Async generator tạo MJPEG stream theo yêu cầu (on-demand).

Chỉ encode JPEG khi có client đang kết nối → tiết kiệm CPU khi không ai xem.
Khi client ngắt kết nối → generator thoát → CPU encoding = 0%.

Luồng hoạt động:
  1. Client GET /video/live → FastAPI gọi generate_mjpeg_stream().
  2. Generator đọc frame mới nhất từ SharedMemory.
  3. Encode BGR → JPEG bằng cv2.imencode().
  4. Yield multipart chunk → trình duyệt hiển thị video liên tục.
  5. Client đóng tab → GeneratorExit → generator dừng, print log.

Format MJPEG multipart:
  --frame\\r\\n
  Content-Type: image/jpeg\\r\\n
  \\r\\n
  <jpeg_bytes>
  \\r\\n
"""
import asyncio
import sys
import time
from typing import AsyncGenerator

if sys.platform == 'win32':
    try:
        if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
            sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import cv2

from src.bridge.ipc_manager import VideoSharedMemory


async def generate_mjpeg_stream(
    video_shm: VideoSharedMemory,
    target_fps: int = 20,
    jpeg_quality: int = 70,
) -> AsyncGenerator[bytes, None]:
    """
    Async generator tạo MJPEG multipart stream từ SharedMemory.

    On-demand encoding:
      - Chỉ gọi cv2.imencode() khi generator đang chạy (có client).
      - Khi không có client → generator không tồn tại → CPU = 0%.

    Throttling:
      - Giới hạn FPS bằng asyncio.sleep() → tránh encode quá nhanh.
      - Frame interval = 1.0 / target_fps (VD: 50ms cho 20 FPS).

    Args:
        video_shm: VideoSharedMemory instance ở reader mode.
        target_fps: FPS tối đa cho stream (mặc định 20).
        jpeg_quality: Chất lượng nén JPEG 0–100 (mặc định 70).

    Yields:
        bytes: MJPEG multipart chunk (header + jpeg data + boundary).
    """
    # Tham số JPEG cho cv2.imencode
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
    # Khoảng thời gian tối thiểu giữa 2 frame (giây)
    frame_interval = 1.0 / target_fps

    print(f"[Stream] Client connected -> start JPEG encoding "
          f"(target {target_fps} FPS, quality {jpeg_quality}%)")

    last_encode_time = 0.0

    try:
        while True:
            # ── Đọc frame mới nhất từ SharedMemory ───────────────────
            frame, timestamp = video_shm.read_latest_frame()

            if frame is None:
                # Chưa có frame nào trong SHM — chờ Core DMS ghi frame đầu tiên
                await asyncio.sleep(0.05)    # 50ms retry
                continue

            # ── Throttle: chờ đủ frame_interval ──────────────────────
            now = time.monotonic()
            elapsed = now - last_encode_time
            if elapsed < frame_interval:
                # Chưa đến lúc encode frame tiếp — nhường CPU
                await asyncio.sleep(frame_interval - elapsed)

            # ── Encode BGR → JPEG ────────────────────────────────────
            success, jpeg_buffer = cv2.imencode('.jpg', frame, encode_params)
            if not success:
                # Encode thất bại (rất hiếm) — bỏ qua frame này
                await asyncio.sleep(0.01)
                continue

            last_encode_time = time.monotonic()

            # ── Yield MJPEG multipart chunk ──────────────────────────
            #   Format chuẩn multipart/x-mixed-replace:
            #   --frame\r\nContent-Type: image/jpeg\r\n\r\n<data>\r\n
            jpeg_bytes = jpeg_buffer.tobytes()
            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n'
                b'\r\n' + jpeg_bytes + b'\r\n'
            )

    except (asyncio.CancelledError, GeneratorExit):
        # Client ngắt kết nối hoặc server shutdown → dừng encoding
        pass
    finally:
        print("[Stream] Client disconnected -> stop JPEG encoding, CPU = 0%")
