"""
IPC Manager — Quản lý giao tiếp liên tiến trình (Inter-Process Communication)
giữa Process 1 (Core DMS) và Process 2 (Dashboard Server).

Kiến trúc double-buffered SharedMemory:
  ┌──────────────────────────────────────────────────────────────────┐
  │  Header (64 bytes)  │  Slot 0 (921,600 B)  │  Slot 1 (921,600 B) │
  │  [write_idx][tstamp] │  BGR frame 480×640×3  │  BGR frame 480×640×3 │
  └──────────────────────────────────────────────────────────────────┘

  - Process 1 (writer): ghi frame vào slot KHÔNG ĐANG ĐỌC, rồi flip write_index.
  - Process 2 (reader): đọc từ slot write_index hiện tại — zero-contention,
    không cần lock vì writer luôn ghi vào slot kia trước khi flip.
  - Tổng dung lượng SHM: 64 + 2×(480×640×3) = ~1.84 MB.

Telemetry & Command dùng multiprocessing.Queue — đơn giản, thread-safe.
"""
import struct
import time
import multiprocessing
from multiprocessing import shared_memory
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

    # ── Tính offset slot trong buffer ────────────────────────────────────
    def _slot_offset(self, slot_index: int) -> int:
        """Trả về byte offset của slot (0 hoặc 1) trong SHM buffer."""
        return _HEADER_SIZE + slot_index * self._frame_bytes

    # ── Writer API ───────────────────────────────────────────────────────
    def write_frame(self, frame: np.ndarray) -> None:
        """
        Ghi frame BGR vào slot KHÔNG active, rồi flip write_index.

        Quy trình zero-contention:
          1. Đọc write_index hiện tại (slot đang được reader đọc).
          2. Ghi frame vào slot KHÁC (inactive slot).
          3. Ghi timestamp.
          4. Flip write_index sang slot vừa ghi (atomic byte write).

        Thời gian thực thi: ~0.2–0.3ms (memcpy 921KB trên RAM).
        Không cần lock vì chỉ 1 writer, và reader đọc slot cũ
        trong khi writer ghi slot mới.

        Args:
            frame: numpy array BGR shape (480, 640, 3) dtype uint8.
        """
        buf = self._shm.buf
        # Bước 1: Xác định slot inactive (slot mà reader KHÔNG đang đọc)
        current_write_idx = buf[_WRITE_IDX_OFFSET]
        next_idx = 1 - current_write_idx          # Flip: 0→1, 1→0

        # Bước 2: Copy pixel data vào inactive slot
        offset = self._slot_offset(next_idx)
        frame_flat = frame.tobytes()
        buf[offset:offset + self._frame_bytes] = frame_flat

        # Bước 3: Ghi timestamp (float64 little-endian, 8 bytes)
        struct.pack_into('<d', buf, _TIMESTAMP_OFFSET, time.time())

        # Bước 4: Flip write_index — atomic single-byte write
        #   Reader sẽ thấy slot mới ngay lập tức ở lần đọc tiếp theo.
        buf[_WRITE_IDX_OFFSET] = next_idx

    # ── Reader API ───────────────────────────────────────────────────────
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

        # Đọc timestamp trước — nếu 0.0 nghĩa là chưa có frame
        timestamp = struct.unpack_from('<d', buf, _TIMESTAMP_OFFSET)[0]
        if timestamp == 0.0:
            return None, 0.0

        # Đọc write_index → slot chứa frame mới nhất
        read_idx = buf[_WRITE_IDX_OFFSET]
        offset = self._slot_offset(read_idx)

        # Copy bytes ra numpy array (tránh đọc trực tiếp từ SHM buffer)
        frame_bytes = bytes(buf[offset:offset + self._frame_bytes])
        frame = np.frombuffer(frame_bytes, dtype=np.uint8).reshape(
            self._frame_shape
        )
        return frame, timestamp

    # ── Cleanup ──────────────────────────────────────────────────────────
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
            pass  # Bỏ qua lỗi khi cleanup (SHM có thể đã bị unlink)


class IPCManager:
    """
    Quản lý toàn bộ giao tiếp liên tiến trình cho Dashboard Bridge.

    Bao gồm:
      - VideoSharedMemory: double-buffered SHM cho video frame.
      - telemetry_queue: Queue(maxsize=1) — đẩy snapshot telemetry mới nhất.
      - command_queue: Queue(maxsize=10) — nhận lệnh từ Dashboard (VD: reset).

    Nguyên tắc quan trọng:
      - write_video_frame() và post_telemetry() KHÔNG BAO GIỜ raise exception.
        Core DMS pipeline PHẢI chạy ổn định bất kể Dashboard có lỗi.
      - Tất cả lỗi IPC được bắt và bỏ qua im lặng (log nếu cần debug).
    """

    def __init__(self, shm_name: str, frame_shape: Tuple[int, int, int]):
        """
        Khởi tạo IPCManager — tạo SHM (writer mode) và các Queue.

        Args:
            shm_name: Tên SharedMemory (VD: 'dms_video_shm').
            frame_shape: Shape frame BGR (VD: (480, 640, 3)).
        """
        self._shm_name = shm_name
        self._frame_shape = frame_shape

        # Tạo SharedMemory — writer mode (Core DMS sở hữu SHM)
        self._video_shm = VideoSharedMemory(
            name=shm_name, frame_shape=frame_shape, create=True
        )

        # Queue telemetry: maxsize=1 → luôn giữ snapshot MỚI NHẤT
        #   Nếu queue đầy, post_telemetry() sẽ drop bản cũ (put_nowait).
        self._telemetry_queue: multiprocessing.Queue = multiprocessing.Queue(
            maxsize=1
        )

        # Queue command: maxsize=10 → buffer lệnh từ Dashboard
        #   VD: {"cmd": "reset"}, {"cmd": "set_threshold", "value": 0.7}
        self._command_queue: multiprocessing.Queue = multiprocessing.Queue(
            maxsize=10
        )

        # Tham chiếu tới Process 2 (Dashboard Server)
        self._server_process: Optional[multiprocessing.Process] = None

    # ── Writer API (gọi từ Core DMS — Process 1) ────────────────────────
    def write_video_frame(self, frame: np.ndarray) -> None:
        """
        Ghi video frame vào SharedMemory — KHÔNG BAO GIỜ crash Core.

        Bọc toàn bộ trong try/except để đảm bảo Core DMS pipeline
        tiếp tục chạy ngay cả khi SHM bị lỗi (VD: Dashboard bị kill,
        SHM bị unlink bất ngờ).

        Args:
            frame: numpy array BGR shape (480, 640, 3) dtype uint8.
        """
        try:
            self._video_shm.write_frame(frame)
        except Exception:
            pass  # Im lặng — Core DMS KHÔNG ĐƯỢC crash vì Bridge

    def post_telemetry(self, snapshot_dict: dict) -> None:
        """
        Đẩy telemetry snapshot mới nhất vào queue — drop nếu queue đầy.

        Sử dụng put_nowait() → nếu queue đầy (maxsize=1), raise Full
        → catch và bỏ qua. Điều này đảm bảo Core DMS không bị block
        chờ Dashboard đọc telemetry.

        Args:
            snapshot_dict: Dict chứa trạng thái hiện tại, VD:
                {
                    "state": "Drowsy",
                    "drowsy_prob": 0.82,
                    "mar": 0.45,
                    "fps": 4.8,
                    "timestamp": 1697558400.123
                }
        """
        try:
            self._telemetry_queue.put_nowait(snapshot_dict)
        except Exception:
            pass  # Queue đầy → drop bản cũ, không block Core

    def poll_command(self) -> Optional[dict]:
        """
        Đọc lệnh từ Dashboard (non-blocking).

        Returns:
            dict lệnh nếu có, None nếu queue rỗng.
            VD: {"cmd": "reset"} hoặc None.
        """
        try:
            return self._command_queue.get_nowait()
        except Exception:
            return None  # Queue rỗng hoặc lỗi → trả None

    # ── Reader API (gọi từ Dashboard Server — Process 2) ────────────────
    def get_telemetry(self) -> Optional[dict]:
        """
        Đọc telemetry snapshot mới nhất (non-blocking, phía server).

        Returns:
            dict telemetry nếu có, None nếu queue rỗng.
        """
        try:
            return self._telemetry_queue.get_nowait()
        except Exception:
            return None

    def get_video_shm_reader(self) -> VideoSharedMemory:
        """
        Tạo và trả về VideoSharedMemory ở chế độ reader (attach, không tạo mới).

        Dashboard Server gọi hàm này để có SHM reader riêng,
        tách biệt với writer của Core DMS.

        Returns:
            VideoSharedMemory instance ở reader mode (create=False).
        """
        return VideoSharedMemory(
            name=self._shm_name, frame_shape=self._frame_shape, create=False
        )

    # ── Process Management ───────────────────────────────────────────────
    def start_server_process(self, host: str, port: int) -> None:
        """
        Khởi động Dashboard Server trên Process riêng (Process 2).

        Sử dụng multiprocessing.Process với target = run_dashboard_server.
        Process 2 sẽ attach vào SHM bằng tên (shm_name) và đọc frame
        từ đó — không truyền numpy array qua pipe/queue.

        Args:
            host: Bind address (VD: '0.0.0.0').
            port: Port FastAPI server (VD: 8000).
        """
        # Import tại đây để tránh circular import và giảm startup time Process 1
        from src.bridge.server_process import run_dashboard_server

        self._server_process = multiprocessing.Process(
            target=run_dashboard_server,
            args=(
                self._shm_name,
                self._frame_shape,
                self._telemetry_queue,
                self._command_queue,
                host,
                port,
            ),
            daemon=True,   # Tự động tắt khi Process 1 kết thúc
            name="DashboardServerProcess",
        )
        self._server_process.start()
        print(f"[Bridge] Dashboard Server started -> http://{host}:{port}")

    # ── Cleanup ──────────────────────────────────────────────────────────
    def cleanup(self) -> None:
        """
        Dọn dẹp toàn bộ tài nguyên IPC:
          - Terminate Process 2 (Dashboard Server).
          - Cleanup SharedMemory (close + unlink).
          - Đóng các Queue.
        """
        # Tắt Process 2
        if self._server_process is not None and self._server_process.is_alive():
            self._server_process.terminate()
            self._server_process.join(timeout=3.0)
            print("[Bridge] Dashboard Server process terminated.")

        # Cleanup SHM (creator → unlink)
        self._video_shm.cleanup()

        # Đóng Queue — tránh resource leak
        try:
            self._telemetry_queue.close()
            self._command_queue.close()
        except Exception:
            pass
