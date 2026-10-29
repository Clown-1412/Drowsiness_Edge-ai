"""
Display Renderer — Vẽ Bounding Box tracking khuôn mặt và HUD Overlay lên frame OpenCV.
Hỗ trợ bật/tắt toàn bộ bảng HUD Overlay và Bounding Box độc lập qua config.
"""
from typing import Optional, Tuple
import cv2
import numpy as np

from src.payload.can_payload import CANPayload


# ── Bảng tra tên trạng thái ECU Alert ──
_ECU_STATE_NAMES = {0: "IDLE", 1: "DROWSY", 2: "DOZING", 3: "NO_FACE"}


class Renderer:
    """
    Vẽ thông tin trạng thái DMS lên frame OpenCV.

    Các thành phần hiển thị:
      1. Bounding Box tracking khuôn mặt kèm nhãn trạng thái tài xế (tùy chọn)
      2. Bảng HUD Overlay thông số ở đáy màn hình (tùy chọn)
    """

    def draw(self, frame: np.ndarray,
             label_text: str, label_color: tuple,
             smoothed_drowsy_prob: float, current_mar: float,
             region_weights: list, camera_fps: float,
             payload: CANPayload, face_detected: bool,
             alert_ecu_connected: bool = False,
             alert_ecu_state: int = 0,
             alert_ecu_hw_health: int = 0x07,
             hud_overlay_enabled: bool = True,
             face_bbox_enabled: bool = True,
             face_bbox: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
        """
        Vẽ Bounding Box và HUD Overlay lên frame (in-place). Trả về frame đã vẽ.

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
            hud_overlay_enabled:  Bật/tắt toàn bộ bảng HUD Overlay dưới đáy màn hình
            face_bbox_enabled:    Bật/tắt Bounding Box tracking khuôn mặt
            face_bbox:            Tọa độ (x1, y1, x2, y2) của khuôn mặt
        """
        h, w = frame.shape[:2]
        display_color = label_color if face_detected else (0, 255, 255)

        # ══════════════════════════════════════════════════════════════════════
        # 1. Bounding Box Tracking Khuôn Mặt + Nhãn Trạng Thái
        # ══════════════════════════════════════════════════════════════════════
        if face_bbox_enabled and face_detected and face_bbox is not None:
            x1, y1, x2, y2 = face_bbox

            # Khung chữ nhật bao quanh khuôn mặt
            cv2.rectangle(frame, (x1, y1), (x2, y2), display_color, 2)

            # 4 góc nhấn mạnh (Corner brackets phong cách hiện đại)
            line_len = min(20, max(8, (x2 - x1) // 5), max(8, (y2 - y1) // 5))
            thick = 3
            # Top-left
            cv2.line(frame, (x1, y1), (x1 + line_len, y1), display_color, thick)
            cv2.line(frame, (x1, y1), (x1, y1 + line_len), display_color, thick)
            # Top-right
            cv2.line(frame, (x2, y1), (x2 - line_len, y1), display_color, thick)
            cv2.line(frame, (x2, y1), (x2, y1 + line_len), display_color, thick)
            # Bottom-left
            cv2.line(frame, (x1, y2), (x1 + line_len, y2), display_color, thick)
            cv2.line(frame, (x1, y2), (x1, y2 - line_len), display_color, thick)
            # Bottom-right
            cv2.line(frame, (x2, y2), (x2 - line_len, y2), display_color, thick)
            cv2.line(frame, (x2, y2), (x2, y2 - line_len), display_color, thick)

            # Nhãn trạng thái tài xế gắn trên đỉnh bounding box:
            # Xóa tiền tố "LEVEL X: " nếu có, chỉ giữ lại phần nội dung trạng thái phía sau
            if ":" in label_text:
                tag_text = label_text.split(":", 1)[1].strip()
            else:
                tag_text = label_text

            # Loại bỏ "| MAR: ..." nếu có để nhãn luôn đồng bộ, gọn gàng (VD: "BUON NGU (78.5%)")
            if " | MAR" in tag_text:
                tag_text = tag_text.split(" | MAR")[0] + ")"

            font_scale = 0.52
            txt_thick = 1
            (tw, th), _ = cv2.getTextSize(tag_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, txt_thick)

            # Vị trí tag: trên bounding box (hoặc lật xuống trong box nếu sát mép trên ảnh)
            tag_y2 = y1 if (y1 - th - 8) >= 0 else min(h, y1 + th + 10)
            tag_y1 = max(0, tag_y2 - th - 8)
            tag_x2 = min(w, x1 + tw + 14)

            # Nền nhãn đặc theo màu trạng thái
            cv2.rectangle(frame, (x1, tag_y1), (tag_x2, tag_y2), display_color, -1)

            # Tính màu chữ tương phản cao: Đen cho nền sáng (Xanh, Cam), Trắng cho nền tối (Đỏ)
            lum = display_color[0] * 0.114 + display_color[1] * 0.587 + display_color[2] * 0.299
            txt_color = (0, 0, 0) if lum > 130 else (255, 255, 255)

            cv2.putText(
                frame, tag_text,
                (x1 + 6, tag_y2 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, txt_color, txt_thick, cv2.LINE_AA
            )

        # ══════════════════════════════════════════════════════════════════════
        # 2. Bảng HUD Overlay Dưới Đáy Màn Hình
        # ══════════════════════════════════════════════════════════════════════
        if hud_overlay_enabled:
            overlay_height = 200

            # Nền mờ cho text
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, h - overlay_height), (580, h - 10), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

            # ── Dòng 1: Level label ──
            display_label = label_text if face_detected else "No Face Detected"
            cv2.putText(
                frame, display_label,
                (20, h - overlay_height + 35), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, display_color, 2
            )

            # ── Dòng 2: FPS + Drowsy + MAR ──
            cv2.putText(
                frame,
                f"FPS: {camera_fps:.1f} | Model Drowsy: "
                f"{smoothed_drowsy_prob * 100:.1f}% | MAR: {current_mar:.2f}",
                (20, h - overlay_height + 65), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1
            )

            # ── Dòng 3: Attention weights ──
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

            # ── Dòng 6: ECU Alert HUD ──
            if alert_ecu_connected:
                # Giải mã sức khỏe phần cứng
                sd_ok     = "OK" if (alert_ecu_hw_health & 0x01) else "ERR"
                i2s_ok    = "OK" if (alert_ecu_hw_health & 0x02) else "ERR"
                buz_ok    = "OK" if (alert_ecu_hw_health & 0x04) else "ERR"
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
