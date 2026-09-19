"""
Payload Builder — Đóng gói trạng thái DMS thành bytes đơn giản.
Prototype: chỉ State (2 bit) + TrackingStatus (1 bit) = 1 byte.
Hiển thị hex/binary trên màn hình, chưa gửi CAN Bus.
"""
from dataclasses import dataclass
from typing import Dict


# ── Mapping tên trạng thái ──
_STATE_NAMES    = {0: "ALERT", 1: "DROWSY/YAWN", 2: "MICROSLEEP", 3: "INIT"}
_TRACKING_NAMES = {0: "FACE_OK", 1: "NO_FACE"}


@dataclass
class DMSPayload:
    """
    Payload đơn giản cho prototype.

    Byte 0 layout (LSB first):
      Bit [1:0] = DriverState    (0=Alert, 1=Drowsy/Yawn, 2=Microsleep, 3=Init)
      Bit [2]   = TrackingStatus (0=Face OK, 1=No Face)
      Bit [7:3] = Reserved (0)
    """
    driver_state:    int    # 0..3
    tracking_status: int    # 0..1

    def to_bytes(self) -> bytes:
        """Đóng gói thành 1 byte."""
        byte_0 = (self.driver_state & 0x03) | ((self.tracking_status & 0x01) << 2)
        return bytes([byte_0])

    def to_hex_string(self) -> str:
        """Biểu diễn hex: '0x02'"""
        return f"0x{self.to_bytes()[0]:02X}"

    def to_binary_string(self) -> str:
        """Biểu diễn binary: '00000010'"""
        return f"{self.to_bytes()[0]:08b}"

    def to_display_dict(self) -> Dict[str, str]:
        """Dict cho renderer hiển thị chi tiết trên OpenCV window."""
        return {
            'state':    _STATE_NAMES.get(self.driver_state, "???"),
            'tracking': _TRACKING_NAMES.get(self.tracking_status, "???"),
            'hex':      self.to_hex_string(),
            'binary':   self.to_binary_string(),
        }


def build_payload(level_id: int, face_detected: bool,
                   buffer_ready: bool = True) -> DMSPayload:
    """
    Chuyển đổi output của StateMachine thành Payload.

    Mapping level_id (pipeline) → driver_state (payload):
        Level 1 (Alert)     → 0 (ALERT)
        Level 2 (Buồn ngủ)  → 1 (DROWSY/YAWN)
        Level 0 (Ngủ gật)   → 2 (MICROSLEEP)
        Khác                → 3 (INIT)

    Khi ``buffer_ready is False`` (chưa đủ 16 frames cho TCN), luôn trả về
    trạng thái INIT (3) để ESP32 biết hệ thống đang khởi động.

    Args:
        level_id:      Từ StateMachine (0, 1, hoặc 2)
        face_detected: True nếu MediaPipe phát hiện khuôn mặt
        buffer_ready:  True khi TCN đã tích đủ 16 frames đặc trưng
    """
    if not buffer_ready:
        driver_state = 3  # INIT — hệ thống đang buffer
    else:
        state_map = {1: 0, 2: 1, 0: 2}
        driver_state = state_map.get(level_id, 3)
    tracking_status = 0 if face_detected else 1
    return DMSPayload(driver_state=driver_state, tracking_status=tracking_status)
