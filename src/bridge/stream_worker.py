"""
Stream Worker — Async generator tạo MJPEG stream theo yêu cầu (on-demand).

Chỉ encode JPEG khi có client đang kết nối → tiết kiệm CPU khi không ai xem.
Khi client ngắt kết nối → generator thoát → CPU encoding = 0%.

Luồng hoạt động:
  1. Client GET /video/live → FastAPI gọi generate_mjpeg_stream().
  2. Generator đọc frame mới nhất từ SharedMemory.
  3. Encode BGR → JPEG bất đồng bộ trên thread riêng bằng cv2.imencode().
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
import time
from typing import AsyncGenerator

import cv2

from src.bridge.video_shm import VideoSharedMemory
from src.config import STREAM_TARGET_FPS, STREAM_JPEG_QUALITY


async def generate_mjpeg_stream(
    video_shm: VideoSharedMemory,
    target_fps: int = STREAM_TARGET_FPS,
    jpeg_quality: int = STREAM_JPEG_QUALITY,
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
        target_fps: FPS tối đa cho stream (mặc định STREAM_TARGET_FPS).
        jpeg_quality: Chất lượng nén JPEG 0–100 (mặc định STREAM_JPEG_QUALITY).

    Yields:
        bytes: MJPEG multipart chunk (header + jpeg data + boundary).
    """
    # Tham số JPEG cho cv2.imencode
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
    # Khoảng thời gian tối thiểu giữa 2 frame (giây)
    frame_interval = 1.0 / max(target_fps, 1)

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

            # ── Encode BGR → JPEG (offload sang threadpool để không block event loop) ──
            success, jpeg_buffer = await asyncio.to_thread(
                cv2.imencode, '.jpg', frame, encode_params
            )
            if not success:
                # Encode thất bại (rất hiếm) — bỏ qua frame này
                await asyncio.sleep(0.01)
                continue

            last_encode_time = time.monotonic()

            # ── Yield MJPEG multipart chunk ──────────────────────────
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
