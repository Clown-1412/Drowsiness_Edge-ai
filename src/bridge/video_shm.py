"""
Video Shared Memory — Quản lý bộ nhớ chia sẻ Double-Buffered cho Video Frames.

Kiến trúc double-buffered SharedMemory:
  ┌──────────────────────────────────────────────────────────────────┐
  │  Header (64 bytes)  │  Slot 0 (921,600 B)  │  Slot 1 (921,600 B) │
  │  [write_idx][tstamp] │  BGR frame 480×640×3  │  BGR frame 480×640×3 │
  └──────────────────────────────────────────────────────────────────┘

  - Process 1 (writer): ghi frame vào slot KHÔNG ĐANG ĐỌC, rồi flip write_index.
  - Process 2 (reader): đọc từ slot write_index hiện tại — zero-contention.
  - Tổng dung lượng SHM: 64 + 2×(480×640×3) = ~1.84 MB.
"""
from multiprocessing import shared_memory
import struct
import time
from typing import Optional, Tuple

import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Hằng số layout SharedMemory
# ──────────────────────────────────────────────────────────────────────────────
_HEADER_SIZE = 64                          # 64 bytes header cố định
_WRITE_IDX_OFFSET = 0                      # Byte 0: write_index (0 hoặc 1)
_TIMESTAMP_OFFSET = 8                      # Bytes 8–15: timestamp float64 LE


class VideoSharedMemory:
    """
    Double-buffered SharedMemory cho video frame.

    Writer (Core DMS — Process 1):
      - Ghi frame vào slot KHÔNG active (inactive = 1 - write_index).
      - Sau khi copy xong, flip write_index (atomic byte write).
      - Thời gian write_frame() < 0.3ms (memcpy 921KB).

    Reader (Dashboard Server — Process 2):
      - Đọc slot write_index hiện tại (slot chứa frame mới nhất).
      - Copy ra numpy array riêng để tránh data race.
      - Không cần lock — writer ghi slot khác trước khi flip.
    """

    def __init__(self, name: str, frame_shape: Tuple[int, int, int],
                 create: bool = False):
        """
        Khởi tạo hoặc attach vào SharedMemory.

        Args:
            name: Tên POSIX shared memory (VD: 'dms_video_shm').
            frame_shape: Kích thước frame (height, width, channels),
                         VD: (480, 640, 3).
            create: True = tạo mới (writer), False = attach (reader).
        """
        self._name = name
        self._frame_shape = frame_shape
        # Kích thước 1 frame BGR (bytes) — 480 × 640 × 3 = 921,600
        self._frame_bytes = int(np.prod(frame_shape))
        # Tổng dung lượng SHM: header + 2 slot
        self._total_size = _HEADER_SIZE + 2 * self._frame_bytes
        self._is_creator = create

        if create:
            # Dọn SHM cũ nếu còn tồn tại (crash trước đó)
            try:
                old = shared_memory.SharedMemory(name=name, create=False)
                old.close()
                old.unlink()
            except FileNotFoundError:
                pass

            self._shm = shared_memory.SharedMemory(
                name=name, create=True, size=self._total_size
            )
            # Khởi tạo header: write_index = 0, timestamp = 0.0
            self._shm.buf[_WRITE_IDX_OFFSET] = 0
            struct.pack_into('<d', self._shm.buf, _TIMESTAMP_OFFSET, 0.0)
        else:
            # Reader: attach vào SHM đã tạo bởi writer
            self._shm = shared_memory.SharedMemory(name=name, create=False)

    def _slot_offset(self, slot_index: int) -> int:
        """Trả về byte offset của slot (0 hoặc 1) trong SHM buffer."""
        return _HEADER_SIZE + slot_index * self._frame_bytes

    def write_frame(self, frame: np.ndarray) -> None:
        """
        Ghi frame BGR vào slot KHÔNG active, rồi flip write_index.

        Quy trình zero-contention:
          1. Đọc write_index hiện tại (slot đang được reader đọc).
          2. Ghi frame vào slot KHÁC (inactive slot).
          3. Ghi timestamp.
          4. Flip write_index sang slot vừa ghi (atomic byte write).
        """
        buf = self._shm.buf
        current_write_idx = buf[_WRITE_IDX_OFFSET]
        next_idx = 1 - current_write_idx          # Flip: 0→1, 1→0

        offset = self._slot_offset(next_idx)
        buf[offset:offset + self._frame_bytes] = memoryview(frame).tobytes()

        struct.pack_into('<d', buf, _TIMESTAMP_OFFSET, time.time())
        buf[_WRITE_IDX_OFFSET] = next_idx

    def read_latest_frame(self) -> Tuple[Optional[np.ndarray], float]:
        """
        Đọc frame mới nhất từ SharedMemory.

        Returns:
            (frame, timestamp):
              - frame: numpy array BGR (480,640,3) — bản copy an toàn.
                       None nếu chưa có frame nào (timestamp == 0.0).
              - timestamp: thời điểm writer ghi frame (time.time()).
        """
        buf = self._shm.buf

        timestamp = struct.unpack_from('<d', buf, _TIMESTAMP_OFFSET)[0]
        if timestamp == 0.0:
            return None, 0.0

        read_idx = buf[_WRITE_IDX_OFFSET]
        offset = self._slot_offset(read_idx)

        frame_bytes = bytes(buf[offset:offset + self._frame_bytes])
        frame = np.frombuffer(frame_bytes, dtype=np.uint8).reshape(
            self._frame_shape
        )
        return frame, timestamp

    def cleanup(self) -> None:
        """
        Đóng và giải phóng SharedMemory.
        - Reader: chỉ close() (detach khỏi SHM).
        - Writer (creator): close() + unlink() (xóa SHM khỏi OS).
        """
        try:
            self._shm.close()
            if self._is_creator:
                self._shm.unlink()
        except Exception:
            pass
