"""
CAN Payload — Đóng gói trạng thái DMS thành chuẩn CAN Bus 8-byte.
Thay thế DMSPayload 1-byte cũ, phục vụ thống nhất cho cả Serial UART và CAN Bus.

Cấu trúc frame CAN tiêu chuẩn 8-byte:
  - Byte 0: State ID [1:0] (0=ALERT, 1=DROWSY/YAWN, 2=MICROSLEEP, 3=INIT), [7:2]=0
  - Byte 1: Tracking Face [0] (0=No Face, 1=Face OK), [7:1]=0
  - Byte 2: Alert Level [1:0] (0=Normal, 1=Warning, 2=Critical), [7:2]=0
  - Byte 3: Alive Counter [3:0] (rolling counter 0–15), [7:4]=0
  - Byte 4: System Status: [0]=buffer_ready, [1]=can_bus_ok, [7:2]=0
  - Byte 5–7: Reserved (0x00) cho mở rộng tương lai (E2E CRC-8, MAR, HUD)
"""
from dataclasses import dataclass
from typing import Dict, Any


# ── Mapping tên hiển thị ──
_STATE_NAMES = {0: "ALERT", 1: "DROWSY/YAWN", 2: "MICROSLEEP", 3: "INIT"}
_TRACKING_NAMES = {False: "NO_FACE", True: "FACE_OK"}
_ALERT_NAMES = {0: "NORMAL", 1: "WARNING", 2: "CRITICAL"}


@dataclass
class CANPayload:
    """
    CAN Payload 8-byte thống nhất cho DMS.
    """
    driver_state:   int     # 0=ALERT, 1=DROWSY/YAWN, 2=MICROSLEEP, 3=INIT
    face_detected:  bool    # True = Face OK, False = No Face
    alert_level:    int     # 0=Normal, 1=Warning, 2=Critical
    alive_counter:  int = 0 # 0..15 (do CANSenderThread cập nhật trước khi phát)
    buffer_ready:   bool = True
    can_bus_ok:     bool = True

    def to_bytes(self) -> bytes:
        """Đóng gói payload thành frame tiêu chuẩn 8 bytes."""
        byte_0 = self.driver_state & 0x03
        byte_1 = 0x01 if self.face_detected else 0x00
        byte_2 = self.alert_level & 0x03
        byte_3 = self.alive_counter & 0x0F
        byte_4 = (0x01 if self.buffer_ready else 0x00) | (0x02 if self.can_bus_ok else 0x00)
        return bytes([byte_0, byte_1, byte_2, byte_3, byte_4, 0x00, 0x00, 0x00])

    def to_hex_string(self) -> str:
        """Biểu diễn chuỗi hex cách nhau bởi dấu cách: '00 01 02 03 01 00 00 00'"""
        return " ".join(f"{b:02X}" for b in self.to_bytes())

    def to_binary_string(self) -> str:
        """Biểu diễn binary byte 0 để tương thích hiển thị."""
        return f"{self.to_bytes()[0]:08b}"

    def to_display_dict(self) -> Dict[str, Any]:
        """Dict cho renderer hiển thị chi tiết trên OpenCV window."""
        return {
            'state':        _STATE_NAMES.get(self.driver_state, "???"),
            'tracking':     _TRACKING_NAMES.get(bool(self.face_detected), "???"),
            'alert':        _ALERT_NAMES.get(self.alert_level, "???"),
            'alive':        self.alive_counter,
            'hex':          self.to_hex_string(),
            'binary':       self.to_binary_string(),
            'buffer_ready': self.buffer_ready,
            'can_bus_ok':   self.can_bus_ok,
        }


def build_can_payload(level_id: int, face_detected: bool,
                      buffer_ready: bool = True) -> CANPayload:
    """
    Chuyển đổi output của StateMachine thành CANPayload 8-byte.

    Mapping level_id (pipeline) → driver_state (payload):
        Level 1 (Alert)     → 0 (ALERT)
        Level 2 (Buồn ngủ)  → 1 (DROWSY/YAWN)
        Level 0 (Ngủ gật)   → 2 (MICROSLEEP)
        Khác                → 3 (INIT)

    Mapping driver_state → alert_level:
        0 (ALERT)           → 0 (Normal)
        1 (DROWSY/YAWN)     → 1 (Warning)
        2 (MICROSLEEP)      → 2 (Critical)
        3 (INIT)            → 0 (Normal)

    Khi buffer_ready is False (chưa đủ 16 frames cho TCN):
        Luôn trả về driver_state = 3 (INIT) và alert_level = 0 (Normal).
    """
    if not buffer_ready:
        driver_state = 3   # INIT
        alert_level = 0    # Normal
    else:
        state_map = {1: 0, 2: 1, 0: 2}
        driver_state = state_map.get(level_id, 3)
        alert_map = {0: 0, 1: 1, 2: 2, 3: 0}
        alert_level = alert_map.get(driver_state, 0)

    return CANPayload(
        driver_state=driver_state,
        face_detected=bool(face_detected),
        alert_level=alert_level,
        alive_counter=0,
        buffer_ready=buffer_ready,
        can_bus_ok=True,
    )
