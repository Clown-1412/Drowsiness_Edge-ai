"""
Dashboard Server Process — FastAPI server chạy trên Process 2 riêng biệt.

Mục đích:
  - Phục vụ video stream MJPEG qua HTTP cho Web Dashboard.
  - Cung cấp REST API cho telemetry và health check.
  - Chạy hoàn toàn độc lập với Core DMS (Process 1) — nếu server crash,
    Core DMS vẫn tiếp tục inference bình thường.

Endpoints:
  GET /video/live      → MJPEG stream (multipart/x-mixed-replace)
  GET /health          → JSON {"status": "ok", "stream_fps": 20, "port": 8000}
  GET /api/telemetry   → JSON telemetry snapshot mới nhất

Giao tiếp với Core DMS:
  - Video frame: đọc từ SharedMemory (zero-copy, attach by name).
  - Telemetry: nhận qua multiprocessing.Queue (non-blocking get).
  - Command: gửi qua multiprocessing.Queue (VD: reset, set_threshold).
"""
import sys

# ── Fix stdout UTF-8 trên Windows ────────────────────────────────────────
# Windows mặc định dùng cp1252 → print tiếng Việt / emoji sẽ crash.
# Reconfigure stdout sang UTF-8 ngay đầu file, TRƯỚC mọi print().
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

from typing import Tuple

import asyncio
import multiprocessing
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse

from src.bridge.ipc_manager import VideoSharedMemory
from src.bridge.stream_worker import generate_mjpeg_stream


def run_dashboard_server(
    shm_name: str,
    frame_shape: Tuple[int, int, int],
    telemetry_queue: multiprocessing.Queue,
    command_queue: multiprocessing.Queue,
    host: str,
    port: int,
) -> None:
    """
    Entry-point cho Dashboard Server — chạy trên Process 2.

    Hàm này là target của multiprocessing.Process, được gọi bởi
    IPCManager.start_server_process(). Nó tạo FastAPI app, attach
    vào SharedMemory đã tạo bởi Process 1, và phục vụ HTTP.

    Args:
        shm_name: Tên SharedMemory để attach (VD: 'dms_video_shm').
        frame_shape: Shape frame BGR (VD: (480, 640, 3)).
        telemetry_queue: Queue nhận telemetry snapshot từ Core DMS.
        command_queue: Queue gửi lệnh ngược về Core DMS.
        host: Bind address (VD: '0.0.0.0').
        port: Port HTTP server (VD: 8000).
    """

    # ── Attach vào SharedMemory (reader mode — KHÔNG tạo mới) ────────
    video_shm = VideoSharedMemory(
        name=shm_name, frame_shape=frame_shape, create=False
    )

    # ── Cache telemetry mới nhất ─────────────────────────────────────
    # Giữ bản cache để trả về khi client poll mà queue rỗng.
    # Dict mutable → cập nhật tại chỗ, không cần biến global.
    _telemetry_cache = {"state": "Initializing", "timestamp": 0.0}

    # ══════════════════════════════════════════════════════════════════
    # Tạo FastAPI App
    # ══════════════════════════════════════════════════════════════════
    app = FastAPI(
        title="DMS Dashboard Server",
        description="Video streaming & telemetry API cho Driver Monitoring System",
        version="1.0.0",
    )

    # ── CORS Middleware — cho phép mọi origin (truy cập LAN) ─────────
    # Dashboard chạy trên trình duyệt máy khác trong mạng LAN
    # → cần allow all origins để không bị CORS block.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],         # Cho phép mọi origin (LAN access)
        allow_credentials=True,
        allow_methods=["*"],         # GET, POST, OPTIONS, ...
        allow_headers=["*"],         # Mọi header
    )

    # ── Shutdown Event — dọn dẹp SHM reader khi server tắt ──────────
    @app.on_event("shutdown")
    async def _on_shutdown():
        """Cleanup SharedMemory reader khi server shutdown."""
        video_shm.cleanup()
        print("[Server] SharedMemory reader cleaned up.")

    # ══════════════════════════════════════════════════════════════════
    # Routes
    # ══════════════════════════════════════════════════════════════════

    @app.get("/video/live")
    async def video_live():
        """
        MJPEG video stream — multipart/x-mixed-replace.

        Trình duyệt hiển thị video liên tục khi truy cập URL này.
        JPEG chỉ được encode khi có client kết nối (on-demand).
        Khi client đóng tab → generator thoát → CPU encoding = 0%.
        """
        return StreamingResponse(
            generate_mjpeg_stream(
                video_shm=video_shm,
                target_fps=20,
                jpeg_quality=70,
            ),
            media_type='multipart/x-mixed-replace; boundary=frame',
        )

    @app.get("/health")
    async def health_check():
        """
        Health check endpoint — kiểm tra server hoạt động.

        Returns:
            JSON với status, stream FPS cấu hình, và port hiện tại.
        """
        return JSONResponse(content={
            "status": "ok",
            "stream_fps": 20,
            "port": port,
        })

    @app.get("/api/telemetry")
    async def get_telemetry():
        """
        Trả về telemetry snapshot mới nhất từ Core DMS.

        Cơ chế:
          1. Thử get_nowait() từ telemetry_queue.
          2. Nếu có data mới → cập nhật cache → trả về.
          3. Nếu queue rỗng → trả về bản cache cũ.

        Dashboard poll endpoint này theo chu kỳ (VD: mỗi 100ms)
        để hiển thị trạng thái real-time.
        """
        try:
            # Non-blocking get — lấy snapshot mới nhất nếu có
            snapshot = telemetry_queue.get_nowait()
            _telemetry_cache.update(snapshot)
        except Exception:
            pass  # Queue rỗng → giữ nguyên cache

        return JSONResponse(content=_telemetry_cache)

    @app.websocket("/ws/telemetry")
    async def websocket_telemetry(websocket: WebSocket):
        """
        WebSocket endpoint — stream JSON telemetry snapshot ở tần số 10Hz (mỗi 100ms)
        trực tiếp tới Web Dashboard.
        """
        await websocket.accept()
        try:
            while True:
                try:
                    snapshot = telemetry_queue.get_nowait()
                    _telemetry_cache.update(snapshot)
                except Exception:
                    pass

                await websocket.send_json(_telemetry_cache)
                await asyncio.sleep(0.1)
        except (WebSocketDisconnect, Exception):
            pass

    # ══════════════════════════════════════════════════════════════════
    # Khởi động Uvicorn (blocking — chạy cho đến khi process bị kill)
    # ══════════════════════════════════════════════════════════════════
    print(f"[Server] Starting Dashboard Server on http://{host}:{port}")
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level='warning',     # Chỉ log warning/error, giảm noise
    )
