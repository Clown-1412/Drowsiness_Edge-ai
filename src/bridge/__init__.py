# Bridge sub-package — Dashboard Bridge: streaming video & telemetry qua SharedMemory + FastAPI
from src.bridge.ipc_manager import IPCManager, VideoSharedMemory
from src.bridge.stream_worker import generate_mjpeg_stream

__all__ = ['IPCManager', 'VideoSharedMemory', 'generate_mjpeg_stream']
