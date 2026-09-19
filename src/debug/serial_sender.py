"""
Serial Sender — Driver tầng vật lý điều khiển cổng Serial UART.
Quản lý kết nối, phát hiện lỗi đường truyền và tự phục hồi (reconnect)
khi cáp USB bị rút/cắm lại.
"""
import time

import serial


def _safe_print(msg: str):
    """Print với fallback khi terminal không hỗ trợ Unicode emoji."""
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode('ascii', errors='replace').decode('ascii'))


class SerialSender:
    """
    Lớp giao tiếp UART tầng thấp tới ESP32.

    Tính năng:
      - Mở cổng COM an toàn, không crash ứng dụng chính nếu cổng không tồn tại.
      - Thuộc tính ``is_connected`` kiểm tra trạng thái cổng.
      - Tự phục hồi khi đường truyền đứt (rút cáp USB) bằng cách thử
        kết nối lại trước khi gửi.
      - Dùng ``print()`` đồng bộ với phong cách logging của project.

    Args:
        port: Tên cổng COM (Windows: 'COM1', 'COM2' / Linux: '/dev/ttyUSB0').
        baudrate: Tốc độ baud (phải khớp với DEBUG_SERIAL_BAUD trên ESP32).
    """

    # Thời gian tối thiểu giữa hai lần thử reconnect để tránh spam mở cổng.
    _RECONNECT_COOLDOWN = 5.0  # giây

    def __init__(self, port: str = "COM1", baudrate: int = 115200):
        self.port = port
        self.baudrate = baudrate
        self.ser = None
        self._last_reconnect_attempt = 0.0
        self._connect()

    # ── Trạng thái kết nối ────────────────────────────────────────────────────

    @property
    def is_connected(self) -> bool:
        """True nếu cổng Serial đang mở và sẵn sàng ghi."""
        return self.ser is not None and self.ser.is_open

    # ── Kết nối / Kết nối lại ─────────────────────────────────────────────────

    def _connect(self):
        """Mở cổng Serial. Ghi log kết quả, không ném ngoại lệ."""
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=0.1)
            # ESP32 thường tự reset khi DTR/RTS thay đổi lúc mở cổng COM.
            # Cần chờ khoảng 1.5 – 2 s để firmware ESP32 khởi động xong.
            time.sleep(2.0)
            _safe_print(f"✅ [SERIAL] Kết nối thành công tới {self.port}@{self.baudrate}")
        except Exception as e:
            _safe_print(f"⚠️  [SERIAL] Không thể mở cổng {self.port}: {e}")
            self.ser = None

    def _try_reconnect(self) -> bool:
        """
        Thử mở lại cổng Serial nếu đã qua thời gian cooldown.
        Trả về True nếu kết nối lại thành công.
        """
        now = time.time()
        if now - self._last_reconnect_attempt < self._RECONNECT_COOLDOWN:
            return False
        self._last_reconnect_attempt = now
        _safe_print(f"🔄 [SERIAL] Thử kết nối lại tới {self.port}...")
        self._connect()
        return self.is_connected

    # ── Gửi dữ liệu ─────────────────────────────────────────────────────────

    def send(self, data: bytes):
        """
        Gửi byte payload sang ESP32.

        Nếu cổng đang đóng, tự động thử reconnect một lần trước khi bỏ qua.
        Nếu ghi thất bại (rút dây USB giữa chừng), đóng handle hỏng và
        đánh dấu để lần gửi tiếp theo sẽ thử reconnect.
        """
        if not self.is_connected:
            if not self._try_reconnect():
                return

        try:
            self.ser.write(data)
            self.ser.flush()
        except Exception as e:
            _safe_print(f"❌ [SERIAL] Lỗi gửi dữ liệu: {e}")
            self._close_handle()

    # ── Đóng cổng ─────────────────────────────────────────────────────────────

    def _close_handle(self):
        """Đóng handle cổng COM hiện tại (nếu còn mở)."""
        if self.ser is not None:
            try:
                self.ser.close()
            except Exception:
                pass
            self.ser = None

    def close(self):
        """Đóng cổng COM an toàn và ghi log."""
        if self.is_connected:
            self._close_handle()
            _safe_print("🔌 [SERIAL] Đã đóng cổng COM.")
