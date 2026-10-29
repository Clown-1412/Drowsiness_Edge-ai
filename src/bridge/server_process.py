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
import sys
import logging

# ── Fix stdout UTF-8 trên Windows ────────────────────────────────────────
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass


def _restore_standard_logging():
    """
    Khắc phục bug của rknnlite làm biến dạng logging._nameToLevel trong Python.
    Đảm bảo uvicorn và standard logging không bị lỗi 'Unknown level: INFO'.
    """
    standard_levels = {
        'CRITICAL': logging.CRITICAL,
        'ERROR': logging.ERROR,
        'WARNING': logging.WARNING,
        'INFO': logging.INFO,
        'DEBUG': logging.DEBUG,
        'NOTSET': logging.NOTSET,
    }
    for name, level in standard_levels.items():
        logging._nameToLevel[name] = level
        logging._levelToName[level] = name


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
    """
    # Khôi phục logging chuẩn ngay khi Process 2 vừa khởi động
    _restore_standard_logging()

    # ── Attach vào SharedMemory (reader mode — KHÔNG tạo mới) ────────
    video_shm = VideoSharedMemory(
        name=shm_name, frame_shape=frame_shape, create=False
    )

    # ── Cache telemetry mới nhất ─────────────────────────────────────
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
        """MJPEG video stream — multipart/x-mixed-replace."""
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
        """Health check endpoint — kiểm tra server hoạt động."""
        return JSONResponse(content={
            "status": "ok",
            "stream_fps": 20,
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
        """WebSocket endpoint — stream JSON telemetry snapshot ở tần số 10Hz."""
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
    # Khởi động Uvicorn
    # ══════════════════════════════════════════════════════════════════
    print(f"[Server] Starting Dashboard Server on http://{host}:{port}")
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_config=None,  # Bỏ qua dictConfig để tránh lỗi xung đột logging của rknnlite
    )
