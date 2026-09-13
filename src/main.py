"""
DMS Pipeline v2 — Entry Point.
Kiến trúc đa luồng: Camera Thread + Inference Thread + Main Thread.

Main Thread chạy: Detection (MediaPipe) + Display + Payload.
MediaPipe FaceLandmarker không thread-safe → bắt buộc chạy trên main thread.

Logic pipeline giữ nguyên 100% từ run_webcam_onnx.py.
"""
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

# Đảm bảo project root luôn có trong sys.path để hỗ trợ cả 2 cách chạy:
# 1. python src/main.py
# 2. python -m src.main
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2

from src.config import (
    CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT,
    LANDMARK_TASK, LANDMARK_URL,
    MIN_FACE_DETECTION_CONFIDENCE, MIN_TRACKING_CONFIDENCE,
    MAR_THRESHOLD,
)
from src.shared_state import SharedState
from src.camera.capture import ThreadedCapture
from src.detection.face_extractor import FaceRegionExtractor
from src.inference.engine import InferenceEngine
from src.payload.builder import build_payload
from src.display.renderer import Renderer


def main():
    """Điều phối toàn bộ pipeline DMS."""

    print("=" * 60)
    print("  DMS Pipeline v2 — Multi-Thread Architecture")
    print("  Giữ nguyên 100% logic dự đoán từ demo gốc")
    print("=" * 60)

    # ── 1. Khởi tạo SharedState ──
    state = SharedState()

    # ── 2. Khởi tạo Camera Thread ──
    print(f"📷 Đang kết nối camera (index {CAMERA_INDEX})...")
    camera = ThreadedCapture(CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT)

    # ── 3. Khởi tạo Face Extractor (MediaPipe — chạy trên main thread) ──
    print("🔍 Khởi tạo MediaPipe FaceLandmarker...")
    face_extractor = FaceRegionExtractor(
        landmark_path=LANDMARK_TASK,
        landmark_url=LANDMARK_URL,
        min_detection_confidence=MIN_FACE_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    )

    # ── 4. Khởi tạo Inference Thread (ONNX sessions sẽ load ở đây) ──
    print("🧠 Khởi tạo ONNX Inference Engine...")
    inference = InferenceEngine(state)

    # ── 5. Khởi tạo Renderer ──
    renderer = Renderer()

    # ── 6. Start threads ──
    camera.start()
    inference.start()

    print("\n✅ Tất cả threads đã khởi động")
    print("   Nhấn 'q' để thoát\n")

    prev_time = time.time()

    try:
        while True:
            # ── Đọc frame từ Camera Thread ──
            ret, frame = camera.read()
            if not ret or frame is None:
                time.sleep(0.001)
                continue

            current_time = time.time()

            # ══════════════════════════════════════════════════════════
            # Detection: MediaPipe (chạy trên main thread vì không
            # thread-safe). Logic giống demo gốc L215-216.
            # ══════════════════════════════════════════════════════════
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_crop, eye_crop, mouth_crop, mar = face_extractor.extract(
                frame_rgb
            )

            face_detected = face_crop is not None

            # ══════════════════════════════════════════════════════════
            # Cập nhật SharedState cho Inference Thread.
            # Yawn timer cập nhật MỌI frame — giống demo gốc L221-223.
            # ══════════════════════════════════════════════════════════
            with state.lock:
                state.face_crop     = face_crop
                state.eye_crop      = eye_crop
                state.mouth_crop    = mouth_crop
                state.face_detected = face_detected

                if face_detected:
                    state.current_mar = mar
                    # Cập nhật yawn timer ở MỌI frame khi miệng mở
                    # Clone 100% từ demo gốc L222-223
                    if mar >= MAR_THRESHOLD:
                        state.last_yawn_time = current_time

            # ══════════════════════════════════════════════════════════
            # Đọc state hiện tại để hiển thị
            # ══════════════════════════════════════════════════════════
            with state.lock:
                label_text     = state.label_text
                label_color    = state.label_color
                smoothed_prob  = state.smoothed_drowsy_prob
                current_mar    = state.current_mar
                region_weights = list(state.region_weights)
                level_id       = state.level_id

            # ── Build Payload ──
            payload = build_payload(level_id, face_detected)

            # ── Tính FPS camera (display) ──
            camera_fps = 1.0 / max(current_time - prev_time, 0.001)
            prev_time = current_time

            # ── Render overlay + payload lên frame ──
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
            )

            cv2.imshow('DMS Pipeline v2 (Multi-Thread)', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\n⏹️  Dừng bởi người dùng")
    finally:
        print("🧹 Đang dọn dẹp...")
        camera.stop()
        inference.stop()
        cv2.destroyAllWindows()
        print("✅ Đã thoát sạch")


if __name__ == '__main__':
    main()
