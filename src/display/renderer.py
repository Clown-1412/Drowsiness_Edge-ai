"""
Display Renderer — Vẽ overlay trạng thái + Payload lên frame OpenCV.
Layout dòng 1-3 giữ nguyên từ demo gốc (L279-293) + dòng 4-5 mới cho Payload.
Dòng 6 mới: Trạng thái ECU Alert (có cờ enable riêng).
"""
import cv2
import numpy as np

from src.payload.can_payload import CANPayload


# ── Bảng tra tên trạng thái ECU Alert ──
_ECU_STATE_NAMES = {0: "IDLE", 1: "DROWSY", 2: "DOZING", 3: "NO_FACE"}


class Renderer:
    """
    Vẽ thông tin trạng thái DMS lên frame OpenCV.

    Layout:
    ┌─────────────────────────────────────────────────────────┐
    │                     Camera Frame                        │
    ├─────────────────────────────────────────────────────────┤
    │ LEVEL X: ... (%)                                        │  ← Giống demo gốc
    │ FPS: x | Drowsy: x% | MAR: x.xx                         │  ← Giống demo gốc
    │ Attn → Face: x | Eye: x | Mouth: x                      │  ← Giống demo gốc
    │ CAN 8B: 00 01 02 03 01 00 00 00                         │  ← CAN Payload Hex
    │ State: MICROSLEEP | Face: OK | Alert: CRITICAL | AC: 3  │  ← Chi tiết giải mã
    │ [ECU ALERT: CONNECTED | HW: OK | State: DOZING]         │  ← ECU Alert HUD (tuỳ chọn)
    └─────────────────────────────────────────────────────────┘
    """

    def draw(self, frame: np.ndarray,
             label_text: str, label_color: tuple,
             smoothed_drowsy_prob: float, current_mar: float,
             region_weights: list, camera_fps: float,
             payload: CANPayload, face_detected: bool,
             alert_ecu_connected: bool = False,
             alert_ecu_state: int = 0,
             alert_ecu_hw_health: int = 0x07,
             hud_alert_ecu_enabled: bool = False) -> np.ndarray:
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
            payload:              CANPayload object
            face_detected:        True nếu có mặt
            alert_ecu_connected:  True nếu ECU Alert ESP32 đang phản hồi
            alert_ecu_state:      Trạng thái cảnh báo đang thực thi trên ESP32
            alert_ecu_hw_health:  Bitfield sức khỏe phần cứng ESP32
            hud_alert_ecu_enabled: Bật/tắt hiển thị dòng ECU Alert HUD
        """
        h, w = frame.shape[:2]

        # ── Tính chiều cao nền mờ dựa trên có bật HUD hay không ──
        overlay_height = 200 if hud_alert_ecu_enabled else 175

        # ── Nền mờ cho text ──
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, h - overlay_height), (580, h - 10), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        # ── Dòng 1: Level label — giống hệt demo gốc L285-286 ──
        display_label = label_text if face_detected else "No Face Detected"
        display_color = label_color if face_detected else (0, 255, 255)
        cv2.putText(
            frame, display_label,
            (20, h - overlay_height + 35), cv2.FONT_HERSHEY_SIMPLEX,
            0.75, display_color, 2
        )

        # ── Dòng 2: FPS + Drowsy + MAR — giống hệt demo gốc L287-288 ──
        cv2.putText(
            frame,
            f"FPS: {camera_fps:.1f} | Model Drowsy: "
            f"{smoothed_drowsy_prob * 100:.1f}% | MAR: {current_mar:.2f}",
            (20, h - overlay_height + 65), cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (255, 255, 255), 1
        )

        # ── Dòng 3: Attention weights — giống hệt demo gốc L289-290 ──
        cv2.putText(
            frame,
            f"Attn -> Face: {region_weights[0]:.2f} | "
            f"Eye: {region_weights[1]:.2f} | Mouth: {region_weights[2]:.2f}",
            (20, h - overlay_height + 95), cv2.FONT_HERSHEY_SIMPLEX,
            0.48, (0, 255, 255), 1
        )

        # ── Dòng 4: Payload CAN 8-byte hex ──
        payload_info = payload.to_display_dict()
        cv2.putText(
            frame,
            f"CAN 8B: {payload_info['hex']}",
            (20, h - overlay_height + 125), cv2.FONT_HERSHEY_SIMPLEX,
            0.55, (0, 200, 255), 2
        )

        # ── Dòng 5: Payload decoded ──
        cv2.putText(
            frame,
            f"State: {payload_info['state']} | Face: {payload_info['tracking']} | "
            f"Alert: {payload_info['alert']} | AC: {payload_info['alive']}",
            (20, h - overlay_height + 153), cv2.FONT_HERSHEY_SIMPLEX,
            0.44, (200, 200, 200), 1
        )

        # ── Dòng 6: ECU Alert HUD (chỉ hiển thị khi hud_alert_ecu_enabled=True) ──
        if hud_alert_ecu_enabled:
            if alert_ecu_connected:
                # Giải mã sức khỏe phần cứng
                sd_ok    = "OK" if (alert_ecu_hw_health & 0x01) else "ERR"
                i2s_ok   = "OK" if (alert_ecu_hw_health & 0x02) else "ERR"
                buz_ok   = "OK" if (alert_ecu_hw_health & 0x04) else "ERR"
                hw_all_ok = (alert_ecu_hw_health & 0x07) == 0x07

                ecu_state_name = _ECU_STATE_NAMES.get(alert_ecu_state, "???")
                hw_label = "OK" if hw_all_ok else f"SD:{sd_ok} I2S:{i2s_ok} BUZ:{buz_ok}"

                hud_text = f"[ECU ALERT: CONNECTED | HW: {hw_label} | Act: {ecu_state_name}]"
                hud_color = (0, 255, 0)  # Xanh lá khi kết nối bình thường
            else:
                hud_text = "[ECU ALERT: OFFLINE]"
                # Nhấp nháy đỏ / vàng mỗi 500ms khi mất kết nối
                import time
                blink = int(time.time() * 2) % 2
                hud_color = (0, 0, 255) if blink else (0, 165, 255)

            cv2.putText(
                frame, hud_text,
                (20, h - overlay_height + 180), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, hud_color, 1
            )

        return frame
