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
import multiprocessing
import queue
from typing import Optional, Tuple

import numpy as np

from src.bridge.video_shm import VideoSharedMemory


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
        Đẩy telemetry snapshot mới nhất vào queue (Drop-Oldest semantics).

        Nếu queue đầy (maxsize=1), xóa bỏ snapshot cũ và đưa snapshot mới vào.
        Không bao giờ raise exception để Core DMS không bị block.
        """
        try:
            self._telemetry_queue.put_nowait(snapshot_dict)
        except queue.Full:
            try:
                self._telemetry_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._telemetry_queue.put_nowait(snapshot_dict)
            except Exception:
                pass
        except Exception:
            pass

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
