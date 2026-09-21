"""
CAN Sender — Driver truyền phát CAN Bus chạy nền đa luồng.
Quản lý kết nối SocketCAN (RK3588) hoặc Virtual CAN (PC dev test).

Đặc điểm kiến trúc:
  - Tách biệt hoàn toàn khỏi Main Thread và Inference Thread (daemon thread).
  - Sử dụng collections.deque(maxlen=1) làm buffer truyền (non-blocking, latest-value-wins).
  - Định kỳ phát tín hiệu (cyclic transmission) theo chu kỳ send_interval.
  - Tự động quản lý và tăng Alive Counter (0–15 rolling counter).
  - Tự phục hồi (reconnect cooldown) khi đứt kết nối hoặc bus-off mà không làm crash app.
"""
import sys
import time
import threading
from collections import deque
from typing import Optional

from src.payload.can_payload import CANPayload

# ── Import python-can an toàn ──
try:
    import can
    CAN_LIB_AVAILABLE = True
except ImportError:
    can = None
    CAN_LIB_AVAILABLE = False


def _safe_print(msg: str):
    """Print với fallback khi terminal không hỗ trợ Unicode emoji."""
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode('ascii', errors='replace').decode('ascii'))


class CANSenderThread(threading.Thread):
    """
    Thread truyền phát CAN Bus nền, tối ưu hiệu năng song song và chống tranh bus.

    Args:
        interface: Loại driver python-can ('socketcan' trên RK3588, 'virtual' trên PC dev).
        channel: Kênh CAN ('can0' trên RK3588, 'vcan0' cho virtual).
        can_id: Arbitration ID (mặc định: 0x200 standard 11-bit).
        baudrate: Tốc độ bit (mặc định: 500000 bps).
        send_interval: Chu kỳ phát định kỳ (mặc định: 0.2s = 5Hz).
        enabled: False để bỏ qua kết nối phần cứng khi chạy test.
    """

    _RECONNECT_COOLDOWN = 5.0  # giây giữa các lần thử mở lại bus khi có lỗi

    def __init__(
        self,
        interface: str = "socketcan",
        channel: str = "can0",
        can_id: int = 0x200,
        baudrate: int = 500000,
        send_interval: float = 0.2,
        enabled: bool = True,
    ):
        super().__init__(daemon=True, name="CANSenderThread")
        self.interface = interface
        self.channel = channel
        self.can_id = can_id
        self.baudrate = baudrate
        self.send_interval = send_interval
        self.enabled = enabled

        # Hàng đợi 1 phần tử non-blocking — luôn ghi đè trạng thái mới nhất
        self._payload_slot: deque = deque(maxlen=1)
        self._last_payload: Optional[CANPayload] = None

        # Alive Counter quản lý nội bộ bởi sender (0..15 rolling counter)
        self._alive_counter: int = 0

        self._stopped: bool = False
        self._bus = None
        self._last_reconnect_attempt: float = 0.0

        if self.enabled:
            self._connect()

    # ── Trạng thái kết nối ────────────────────────────────────────────────────

    @property
    def is_connected(self) -> bool:
        """True nếu CAN Bus handle đang mở hợp lệ."""
        return self._bus is not None

    def _connect(self):
        """Mở kết nối tới CAN Bus. Không ném ngoại lệ làm crash app."""
        if not CAN_LIB_AVAILABLE:
            _safe_print("⚠️  [CAN] Thư viện python-can chưa được cài đặt. CAN Bus bị vô hiệu hóa.")
            self._bus = None
            return

        try:
            kwargs = {}
            if self.baudrate > 0 and self.interface != "virtual":
                kwargs["bitrate"] = self.baudrate

            self._bus = can.Bus(interface=self.interface, channel=self.channel, **kwargs)
            _safe_print(f"✅ [CAN] Kết nối thành công tới {self.interface}:{self.channel} (ID=0x{self.can_id:X})")
        except Exception as e:
            _safe_print(f"⚠️  [CAN] Không thể mở kết nối {self.interface}:{self.channel}: {e}")
            self._bus = None

    def _try_reconnect(self) -> bool:
        """Thử kết nối lại nếu đã vượt qua khoảng cooldown."""
        now = time.time()
        if now - self._last_reconnect_attempt < self._RECONNECT_COOLDOWN:
            return False
        self._last_reconnect_attempt = now
        _safe_print(f"🔄 [CAN] Thử kết nối lại CAN {self.interface}:{self.channel}...")
        self._connect()
        return self.is_connected

    # ── Giao diện nhận dữ liệu (Non-blocking Producer) ─────────────────────────

    def post(self, payload: CANPayload):
        """
        Main Thread hoặc Inference Thread gọi hàm này để cập nhật payload mới nhất.
        Hàm hoàn toàn non-blocking: chỉ ghi đè vào deque(maxlen=1).
        """
        self._payload_slot.append(payload)

    # ── Vòng lặp truyền phát (Consumer) ───────────────────────────────────────

    def run(self):
        """Vòng lặp gửi dữ liệu định kỳ trên thread riêng."""
        _safe_print("📡 CAN Sender thread started")

        while not self._stopped:
            start_time = time.time()

            # Lấy payload mới nhất nếu có trong slot
            if len(self._payload_slot) > 0:
                self._last_payload = self._payload_slot.pop()

            if self.enabled and self._last_payload is not None:
                self._send_frame(self._last_payload)

            # Giữ nhịp gửi định kỳ theo send_interval (5Hz)
            elapsed = time.time() - start_time
            sleep_time = self.send_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _send_frame(self, payload: CANPayload):
        """Đóng gói và gửi một CAN frame."""
        if not self.is_connected:
            if not self._try_reconnect():
                payload.can_bus_ok = False
                return

        # Gán Alive Counter vào payload và tăng rolling counter
        payload.alive_counter = self._alive_counter
        payload.can_bus_ok = True
        self._alive_counter = (self._alive_counter + 1) & 0x0F

        data = payload.to_bytes()

        try:
            msg = can.Message(
                arbitration_id=self.can_id,
                data=data,
                is_extended_id=False,
            )
            self._bus.send(msg)
        except Exception as e:
            _safe_print(f"❌ [CAN] Lỗi truyền dữ liệu: {e}")
            payload.can_bus_ok = False
            self._close_handle()

    # ── Dọn dẹp & Giải phóng ──────────────────────────────────────────────────

    def _close_handle(self):
        """Đóng an toàn bus handle hiện tại."""
        if self._bus is not None:
            try:
                self._bus.shutdown()
            except Exception:
                pass
            self._bus = None

    def stop(self):
        """Dừng thread và đóng bus kết nối."""
        self._stopped = True
        if self.enabled:
            self._close_handle()
            _safe_print("🔌 [CAN] Đã đóng kết nối CAN Bus.")
