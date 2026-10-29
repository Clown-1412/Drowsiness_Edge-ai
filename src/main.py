"""
DMS Pipeline — Main Entry Point (Headless Architecture).
Kiến trúc đa luồng:
  - Camera Thread: Đọc camera liên tục, loại bỏ buffer lag
  - Inference Thread: 100% RKNN NPU (Feature Extractor + TCN Classifier)
  - Main Thread: Detection (MediaPipe) + Frame Decorator + CAN Bus + Dashboard Bridge
  - Process 2: Dashboard Server (FastAPI + WebSocket + MJPEG Streaming)

Hoạt động hoàn toàn ở chế độ Headless (không mở cửa sổ OpenCV GUI trên desktop).
Mọi hình ảnh (kèm Bounding Box / HUD Overlay) và thông số telemetry được đẩy qua
SharedMemory và IPC Queue lên Web Dashboard.
"""
import signal
import sys
import time
from pathlib import Path

# Đảm bảo console Windows không bị lỗi UnicodeEncodeError khi print tiếng Việt / emoji
if sys.platform == 'win32':
    try:
        if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
            sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# Đảm bảo project root luôn có trong sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2

from src.config import (
    CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT,
    LANDMARK_TASK, LANDMARK_URL,
    MIN_FACE_DETECTION_CONFIDENCE, MIN_TRACKING_CONFIDENCE,
    MAR_THRESHOLD,
    CAN_ENABLED, CAN_INTERFACE, CAN_CHANNEL,
    CAN_BAUDRATE, CAN_ARBITRATION_ID, CAN_SEND_INTERVAL,
    CAN_FEEDBACK_ID, CAN_WATCHDOG_TIMEOUT,
    HUD_OVERLAY_ENABLED, FACE_BBOX_ENABLED,
    BRIDGE_ENABLED, BRIDGE_HOST, BRIDGE_PORT, SHM_VIDEO_NAME,
)
from src.shared_state import SharedState
from src.camera.capture import ThreadedCapture
from src.detection.face_extractor import FaceRegionExtractor
from src.inference.engine import InferenceEngine
from src.payload.can_payload import build_can_payload
from src.can_bus.can_sender import CANSenderThread
from src.display.renderer import Renderer
from src.bridge.ipc_manager import IPCManager


def main():
    """Điều phối toàn bộ pipeline DMS ở chế độ Headless."""

    print("=" * 60)
    print("  DMS Pipeline — Headless Edge-AI (RK3588 NPU + CAN Bus)")
    print("=" * 60)

    # ── 1. Đăng ký bộ xử lý tín hiệu thoát (Graceful Shutdown) ──
    stop_requested = False

    def handle_signal(sig, frame):
        nonlocal stop_requested
        print(f"\n[DMS] Nhận tín hiệu dừng ({sig}), đang dọn dẹp hệ thống...")
        stop_requested = True

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    # ── 2. Khởi tạo SharedState ──
    state = SharedState()

    # ── 3. Khởi tạo Camera Thread ──
    print(f"Đang kết nối camera (index {CAMERA_INDEX})...")
    camera = ThreadedCapture(CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT)

    # ── 4. Khởi tạo Face Extractor (MediaPipe — chạy trên main thread) ──
    print("Khởi tạo MediaPipe FaceLandmarker...")
    face_extractor = FaceRegionExtractor(
        landmark_path=LANDMARK_TASK,
        landmark_url=LANDMARK_URL,
        min_detection_confidence=MIN_FACE_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    )

    # ── 5. Khởi tạo RKNN Inference Thread (100% NPU) ──
    print("🧠 Khởi tạo RKNN Inference Engine (NPU)...")
    inference = InferenceEngine(state)

    # ── 6. Khởi tạo Renderer (Vẽ BBox/HUD lên frame cho Dashboard Video Stream) ──
    renderer = Renderer()

    # ── 7. Khởi tạo CAN Bus Sender (Giao tiếp Ngoại vi Ô tô — Hai chiều) ──
    can_sender = CANSenderThread(
        interface=CAN_INTERFACE,
        channel=CAN_CHANNEL,
        can_id=CAN_ARBITRATION_ID,
        feedback_id=CAN_FEEDBACK_ID,
        baudrate=CAN_BAUDRATE,
        send_interval=CAN_SEND_INTERVAL,
        watchdog_timeout=CAN_WATCHDOG_TIMEOUT,
        enabled=CAN_ENABLED,
        shared_state=state,
    )

    # ── 8. Khởi tạo Dashboard Bridge (Process 2: Web Server) ──
    ipc_manager = None
    if BRIDGE_ENABLED:
        print(f"Khởi tạo Bridge (SharedMemory + Server Process trên cổng {BRIDGE_PORT})...")
        ipc_manager = IPCManager(
            shm_name=SHM_VIDEO_NAME,
            frame_shape=(CAMERA_HEIGHT, CAMERA_WIDTH, 3),
        )
        ipc_manager.start_server_process(host=BRIDGE_HOST, port=BRIDGE_PORT)

    # ── 9. Khởi chạy các Threads nền ──
    camera.start()
    inference.start()
    can_sender.start()

    print("\n✅ Tất cả threads đã khởi động thành công.")
    print("   Chế độ: HEADLESS (Không mở cửa sổ OpenCV GUI).")
    print(f"   Dashboard Stream: http://{BRIDGE_HOST}:{BRIDGE_PORT}/video/live")
    print("   Nhấn Ctrl+C để dừng an toàn.\n")

    prev_time = time.time()

    try:
        while not stop_requested:
            # ── Đọc frame từ Camera Thread ──
            ret, frame = camera.read()
            if not ret or frame is None:
                time.sleep(0.001)
                continue

            current_time = time.time()

            # ══════════════════════════════════════════════════════════
            # Detection: MediaPipe Face Landmark
            # ══════════════════════════════════════════════════════════
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_crop, eye_crop, mouth_crop, mar, face_bbox = face_extractor.extract(
                frame_rgb
            )

            face_detected = face_crop is not None

            # ══════════════════════════════════════════════════════════
            # Cập nhật SharedState cho RKNN Inference Thread
            # ══════════════════════════════════════════════════════════
            with state.lock:
                state.face_crop     = face_crop
                state.eye_crop      = eye_crop
                state.mouth_crop    = mouth_crop
                state.face_detected = face_detected

                if face_detected:
                    state.current_mar = mar
                    if mar >= MAR_THRESHOLD:
                        state.last_yawn_time = current_time

            # ══════════════════════════════════════════════════════════
            # Đọc state để chuẩn bị Payload và Decorate Frame
            # ══════════════════════════════════════════════════════════
            with state.lock:
                label_text          = state.label_text
                label_color         = state.label_color
                smoothed_prob       = state.smoothed_drowsy_prob
                current_mar         = state.current_mar
                region_weights      = list(state.region_weights)
                level_id            = state.level_id
                buffer_ready        = state.buffer_ready
                alert_ecu_connected = state.alert_ecu_connected
                alert_ecu_state     = state.alert_ecu_state
                alert_ecu_hw_health = state.alert_ecu_hw_health

            # ── Build Payload CAN 8-byte và đẩy sang CAN Bus Thread ──
            payload = build_can_payload(level_id, face_detected, buffer_ready)
            can_sender.post(payload)

            # ── Tính FPS Camera và cập nhật telemetry ──
            camera_fps = 1.0 / max(current_time - prev_time, 0.001)
            prev_time = current_time
            with state.lock:
                state.camera_fps = camera_fps

            # ── Decorate Bounding Box / HUD Overlay lên frame cho Dashboard Video ──
            renderer.draw(
                frame=frame,
                label_text=label_text,
                label_color=label_color,
                smoothed_drowsy_prob=smoothed_prob,
                current_mar=current_mar,
                region_weights=region_weights,
                camera_fps=camera_fps,
                payload=payload,
                face_detected=face_detected,
                alert_ecu_connected=alert_ecu_connected,
                alert_ecu_state=alert_ecu_state,
                alert_ecu_hw_health=alert_ecu_hw_health,
                hud_overlay_enabled=HUD_OVERLAY_ENABLED,
                face_bbox_enabled=FACE_BBOX_ENABLED,
                face_bbox=face_bbox,
            )

            # ── Đẩy video frame & telemetry sang Dashboard Bridge (non-blocking) ──
            if ipc_manager is not None:
                ipc_manager.write_video_frame(frame)
                ipc_manager.post_telemetry(state.get_telemetry_snapshot())

            # Nhường CPU nhẹ nhàng cho các thread khác
            time.sleep(0.001)

    except KeyboardInterrupt:
        print("\n[DMS] Dừng bởi người dùng (KeyboardInterrupt)")
    finally:
        print("Đang giải phóng tài nguyên...")
        if ipc_manager is not None:
            ipc_manager.cleanup()
        camera.stop()
        can_sender.stop()
        inference.stop()
        print("Đã giải phóng sạch toàn bộ hệ thống.")


if __name__ == '__main__':
    main()
