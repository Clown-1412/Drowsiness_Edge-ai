"""
Display Renderer — Vẽ overlay trạng thái + Payload lên frame OpenCV.
Layout dòng 1-3 giữ nguyên từ demo gốc (L279-293) + dòng 4-5 mới cho Payload.
"""
import cv2
import numpy as np

from src.payload.builder import DMSPayload


class Renderer:
    """
    Vẽ thông tin trạng thái DMS lên frame OpenCV.

    Layout:
    ┌─────────────────────────────────────────────┐
    │              Camera Frame                   │
    ├─────────────────────────────────────────────┤
    │ LEVEL X: ... (%)                            │  ← Giống demo gốc
    │ FPS: x | Drowsy: x% | MAR: x.xx             │  ← Giống demo gốc
    │ Attn → Face: x | Eye: x | Mouth: x          │  ← Giống demo gốc
    │ PAYLOAD: 0x02 [00000010]                    │  ← MỚI
    │ State: MICROSLEEP | Tracking: FACE_OK       │  ← MỚI
    └─────────────────────────────────────────────┘
    """

    def draw(self, frame: np.ndarray,
             label_text: str, label_color: tuple,
             smoothed_drowsy_prob: float, current_mar: float,
             region_weights: list, camera_fps: float,
             payload: DMSPayload, face_detected: bool) -> np.ndarray:
        """
        Vẽ overlay lên frame (in-place). Trả về frame đã vẽ.

        Args:
            frame:                BGR frame từ camera
            label_text:           Text trạng thái (từ StateMachine)
            label_color:          BGR color cho label
            smoothed_drowsy_prob: Xác suất buồn ngủ đã smoothing
            current_mar:          Mouth Aspect Ratio hiện tại
            region_weights:       [face_w, eye_w, mouth_w] từ attention
            camera_fps:           FPS hiển thị camera
            payload:              DMSPayload object
            face_detected:        True nếu có mặt
        """
        h, w = frame.shape[:2]

        # ── Nền mờ cho text — mở rộng chứa thêm Payload (5 dòng) ──
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, h - 170), (520, h - 10), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        # ── Dòng 1: Level label — giống hệt demo gốc L285-286 ──
        display_label = label_text if face_detected else "No Face Detected"
        display_color = label_color if face_detected else (0, 255, 255)
        cv2.putText(
            frame, display_label,
            (20, h - 140), cv2.FONT_HERSHEY_SIMPLEX, 0.75, display_color, 2
        )

        # ── Dòng 2: FPS + Drowsy + MAR — giống hệt demo gốc L287-288 ──
        cv2.putText(
            frame,
            f"FPS: {camera_fps:.1f} | Model Drowsy: "
            f"{smoothed_drowsy_prob * 100:.1f}% | MAR: {current_mar:.2f}",
            (20, h - 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
        )

        # ── Dòng 3: Attention weights — giống hệt demo gốc L289-290 ──
        cv2.putText(
            frame,
            f"Attn -> Face: {region_weights[0]:.2f} | "
            f"Eye: {region_weights[1]:.2f} | Mouth: {region_weights[2]:.2f}",
            (20, h - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1
        )

        # ── Dòng 4: Payload hex + binary — MỚI ──
        payload_info = payload.to_display_dict()
        cv2.putText(
            frame,
            f"PAYLOAD: {payload_info['hex']} [{payload_info['binary']}]",
            (20, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2
        )

        # ── Dòng 5: Payload decoded — MỚI ──
        cv2.putText(
            frame,
            f"State: {payload_info['state']} | "
            f"Tracking: {payload_info['tracking']}",
            (20, h - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1
        )

        return frame
