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
  WS  /ws/telemetry    → WebSocket stream JSON telemetry @ 10Hz
"""
import asyncio
from contextlib import asynccontextmanager
import multiprocessing
from typing import Tuple

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn

from src.bridge.video_shm import VideoSharedMemory
from src.bridge.stream_worker import generate_mjpeg_stream
from src.config import STREAM_TARGET_FPS, STREAM_JPEG_QUALITY, TELEMETRY_PUSH_HZ
from src.utils import configure_win_console, restore_standard_logging


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
    """
    # Khởi tạo môi trường console và logging
    configure_win_console()
    restore_standard_logging()

    # ── Attach vào SharedMemory (reader mode — KHÔNG tạo mới) ────────
    video_shm = VideoSharedMemory(
        name=shm_name, frame_shape=frame_shape, create=False
    )

    # ── Cache telemetry mới nhất ─────────────────────────────────────
    _telemetry_cache = {"state": "Initializing", "timestamp": 0.0}

    # ── Lifespan Context Manager thay cho on_event deprecated ─────────
    @asynccontextmanager
    async def lifespan(app_instance: FastAPI):
        yield
        video_shm.cleanup()
        print("[Server] SharedMemory reader cleaned up.")

    # ══════════════════════════════════════════════════════════════════
    # Tạo FastAPI App
    # ══════════════════════════════════════════════════════════════════
    app = FastAPI(
        title="DMS Dashboard Server",
        description="Video streaming & telemetry API cho Driver Monitoring System",
        version="1.0.0",
        lifespan=lifespan,
    )

    # ── CORS Middleware — cho phép mọi origin (truy cập LAN) ─────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],         # Cho phép mọi origin (LAN access)
        allow_credentials=False,
        allow_methods=["*"],         # GET, POST, OPTIONS, ...
        allow_headers=["*"],         # Mọi header
    )

    # ══════════════════════════════════════════════════════════════════
    # Routes
    # ══════════════════════════════════════════════════════════════════

    @app.get("/video/live")
    async def video_live():
        """MJPEG video stream — multipart/x-mixed-replace."""
        return StreamingResponse(
            generate_mjpeg_stream(
                video_shm=video_shm,
                target_fps=STREAM_TARGET_FPS,
                jpeg_quality=STREAM_JPEG_QUALITY,
            ),
            media_type='multipart/x-mixed-replace; boundary=frame',
        )

    @app.get("/health")
    async def health_check():
        """Health check endpoint — kiểm tra server hoạt động."""
        return JSONResponse(content={
            "status": "ok",
            "stream_fps": STREAM_TARGET_FPS,
            "port": port,
        })

    @app.get("/api/telemetry")
    async def get_telemetry():
        """Trả về telemetry snapshot mới nhất từ Core DMS."""
        try:
            snapshot = telemetry_queue.get_nowait()
            _telemetry_cache.update(snapshot)
        except Exception:
            pass

        return JSONResponse(content=_telemetry_cache)

    @app.websocket("/ws/telemetry")
    async def websocket_telemetry(websocket: WebSocket):
        """WebSocket endpoint — stream JSON telemetry snapshot."""
        await websocket.accept()
        interval = 1.0 / max(TELEMETRY_PUSH_HZ, 1)
        try:
            while True:
                try:
                    snapshot = telemetry_queue.get_nowait()
                    _telemetry_cache.update(snapshot)
                except Exception:
                    pass

                await websocket.send_json(_telemetry_cache)
                await asyncio.sleep(interval)
        except (WebSocketDisconnect, Exception):
            pass

    # ══════════════════════════════════════════════════════════════════
    # Khởi động Uvicorn
    # ══════════════════════════════════════════════════════════════════
    print(f"[Server] Starting Dashboard Server on http://{host}:{port}")
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_config=None,  # Bỏ qua dictConfig để tránh lỗi xung đột logging của rknnlite
    )
