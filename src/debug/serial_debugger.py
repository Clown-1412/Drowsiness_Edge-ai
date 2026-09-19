"""
Serial Debugger — Điều phối gửi trạng thái DMS tới ESP32.
Kèm bộ lọc On-Change + Heartbeat để chống spam còi Buzzer.

Tách biệt logic lọc trạng thái (tầng ứng dụng) khỏi tầng vật lý
(SerialSender), giúp main.py chỉ cần gọi duy nhất ``debugger.update(...)``.
"""
from src.debug.serial_sender import SerialSender


class SerialDebugger:
    """
    Bộ điều phối truyền Serial cho hệ thống DMS.

    Tính năng:
      - Chỉ gửi dữ liệu khi có **thay đổi trạng thái** (On-Change):
        ``level_id`` hoặc ``face_detected`` thay đổi so với lần gửi trước.
      - Tự động gửi lại theo chu kỳ **Heartbeat** để ESP32 xác nhận đường
        truyền còn sống (mặc định mỗi 3 giây).
      - Có thể tắt hoàn toàn bằng cờ ``enabled=False`` (chế độ test không
        phần cứng).

    Args:
        port: Cổng COM (Windows) hoặc device path (Linux).
        baudrate: Tốc độ baud.
        heartbeat_interval: Chu kỳ heartbeat (giây).
        enabled: ``False`` để bỏ qua toàn bộ việc mở cổng và gửi dữ liệu.
    """

    def __init__(self, port: str = "COM1", baudrate: int = 115200,
                 heartbeat_interval: float = 3.0, enabled: bool = True):
        self.enabled = enabled
        self.heartbeat_interval = heartbeat_interval

        # Trạng thái gửi gần nhất — dùng cho bộ lọc On-Change
        self._last_sent_level = None
        self._last_sent_face_status = None
        self._last_sent_time = 0.0

        # Khởi tạo tầng vật lý
        self._sender = None
        if self.enabled:
            try:
                print("🔌 Đang kết nối ESP32 qua cổng Serial COM...")
            except UnicodeEncodeError:
                print("[SERIAL] Dang ket noi ESP32 qua cong Serial COM...")
            self._sender = SerialSender(port=port, baudrate=baudrate)

    # ── API chính ─────────────────────────────────────────────────────────────

    def update(self, level_id: int, face_detected: bool,
               payload_bytes: bytes, current_time: float) -> bool:
        """
        Quyết định có gửi payload hay không dựa trên On-Change + Heartbeat.

        Args:
            level_id: Mã trạng thái tài xế (0–3) từ StateMachine / Payload.
            face_detected: Khuôn mặt có đang được theo dõi hay không.
            payload_bytes: Gói tin đã đóng gói (``DMSPayload.to_bytes()``).
            current_time: ``time.time()`` hiện tại (tránh gọi lại bên trong).

        Returns:
            ``True`` nếu gói tin vừa được gửi ra cổng Serial.
        """
        if not self.enabled or self._sender is None:
            return False

        state_changed = (
            level_id != self._last_sent_level
            or face_detected != self._last_sent_face_status
        )
        heartbeat_due = (
            current_time - self._last_sent_time >= self.heartbeat_interval
        )

        if state_changed or heartbeat_due:
            self._sender.send(payload_bytes)
            self._last_sent_level = level_id
            self._last_sent_face_status = face_detected
            self._last_sent_time = current_time
            return True

        return False

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def close(self):
        """Đóng kết nối Serial nếu đang mở."""
        if self._sender is not None:
            self._sender.close()
