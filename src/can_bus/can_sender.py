"""
CAN Sender — Driver truyền phát CAN Bus chạy nền đa luồng.
Quản lý kết nối SocketCAN (RK3588) hoặc Virtual CAN (PC dev test).

Phiên bản 2.0: Hỗ trợ giao tiếp 2 chiều (Bidirectional CAN)
  - Phát frame 0x200 (DMS Command/Status) theo chu kỳ 5Hz.
  - Nhận frame 0x201 (Alert Feedback/Telemetry) từ ECU Alert ESP32.
  - Cập nhật trạng thái kết nối và sức khỏe phần cứng của ECU Alert vào SharedState.

Đặc điểm kiến trúc:
  - Tách biệt hoàn toàn khỏi Main Thread và Inference Thread (daemon thread).
  - Sử dụng collections.deque(maxlen=1) làm buffer truyền (non-blocking, latest-value-wins).
  - Định kỳ phát tín hiệu (cyclic transmission) theo chu kỳ send_interval.
  - Tự động quản lý và tăng Alive Counter (0 to 15 rolling counter).
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


# ── Tên trạng thái ECU Alert (giải mã 0x201) ──
_ALERT_STATE_NAMES = {0: "IDLE", 1: "DROWSY", 2: "DOZING", 3: "NO_FACE"}


class CANSenderThread(threading.Thread):
    """
    Thread truyền phát CAN Bus nền, tối ưu hiệu năng song song và chống tranh bus.
    Phiên bản 2.0: Hỗ trợ nhận phản hồi từ ECU Alert (0x201).

    Args:
        interface: Loại driver python-can ('socketcan' trên RK3588, 'virtual' trên PC dev).
        channel: Kênh CAN ('can0' trên RK3588, 'vcan0' cho virtual).
        can_id: Arbitration ID phát (mặc định: 0x200 standard 11-bit).
        feedback_id: Arbitration ID nhận phản hồi (mặc định: 0x201).
        baudrate: Tốc độ bit (mặc định: 500000 bps).
        send_interval: Chu kỳ phát định kỳ (mặc định: 0.2s = 5Hz).
        watchdog_timeout: Thời gian tối đa chờ phản hồi ECU Alert (giây).
        enabled: False để bỏ qua kết nối phần cứng khi chạy test.
        shared_state: SharedState object để ghi nhận trạng thái ECU Alert.
    """

    _RECONNECT_COOLDOWN = 5.0  # giây giữa các lần thử mở lại bus khi có lỗi

    def __init__(
        self,
        interface: str = "socketcan",
        channel: str = "can0",
        can_id: int = 0x200,
        feedback_id: int = 0x201,
        baudrate: int = 500000,
        send_interval: float = 0.2,
        watchdog_timeout: float = 1.5,
        enabled: bool = True,
        shared_state=None,
    ):
        super().__init__(daemon=True, name="CANSenderThread")
        self.interface = interface
        self.channel = channel
        self.can_id = can_id
        self.feedback_id = feedback_id
        self.baudrate = baudrate
        self.send_interval = send_interval
        self.watchdog_timeout = watchdog_timeout
        self.enabled = enabled
        self._shared_state = shared_state

        # Hàng đợi 1 phần tử non-blocking — luôn ghi đè trạng thái mới nhất
        self._payload_slot: deque = deque(maxlen=1)
        self._last_payload: Optional[CANPayload] = None

        # Alive Counter quản lý nội bộ bởi sender (0..15 rolling counter)
        self._alive_counter: int = 0

        self._stopped: bool = False
        self._bus = None
        self._last_reconnect_attempt: float = 0.0

        # Trạng thái phản hồi từ ECU Alert
        self._last_feedback_time: float = 0.0
        self._alert_ecu_connected: bool = False

        if self.enabled:
            self._connect()

    # ── Trạng thái kết nối ────────────────────────────────────────────────────

    @property
    def is_connected(self) -> bool:
        """True nếu CAN Bus handle đang mở hợp lệ."""
        return self._bus is not None

    @property
    def alert_ecu_connected(self) -> bool:
        """True nếu ECU Alert ESP32 đang phản hồi trong khoảng watchdog."""
        return self._alert_ecu_connected

    def _connect(self):
        """Mở kết nối tới CAN Bus. Không ném ngoại lệ làm crash app."""
        if not CAN_LIB_AVAILABLE:
            _safe_print("[CAN] Thu vien python-can chua duoc cai dat. CAN Bus bi vo hieu hoa.")
            self._bus = None
            return

        try:
            kwargs = {}
            # Trên Linux SocketCAN, bitrate được cấu hình ở mức OS (ip link set can0).
            # Không truyền bitrate vào Bus(...) để tránh lỗi quyền root / OS permission.
            if self.baudrate > 0 and self.interface not in ("virtual", "socketcan"):
                kwargs["bitrate"] = self.baudrate

            self._bus = can.Bus(interface=self.interface, channel=self.channel, **kwargs)
            _safe_print(f"[CAN] Ket noi thanh cong toi {self.interface}:{self.channel} "
                        f"(TX=0x{self.can_id:X}, RX=0x{self.feedback_id:X})")
        except Exception as e:
            _safe_print(f"[CAN] Khong the mo ket noi {self.interface}:{self.channel}: {e}")
            self._bus = None

    def _try_reconnect(self) -> bool:
        """Thử kết nối lại nếu đã vượt qua khoảng cooldown."""
        now = time.time()
        if now - self._last_reconnect_attempt < self._RECONNECT_COOLDOWN:
            return False
        self._last_reconnect_attempt = now
        _safe_print(f"[CAN] Thu ket noi lai CAN {self.interface}:{self.channel}...")
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
        _safe_print("[CAN] CAN Sender thread started (Bidirectional mode)")

        while not self._stopped:
            start_time = time.time()

            # Lấy payload mới nhất nếu có trong slot
            if len(self._payload_slot) > 0:
                self._last_payload = self._payload_slot.pop()

            if self.enabled and self._last_payload is not None:
                self._send_frame(self._last_payload)

            # Nhận phản hồi từ ECU Alert (non-blocking)
            if self.enabled:
                self._recv_feedback()

            # Cập nhật trạng thái watchdog ECU Alert
            self._update_watchdog()

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
            _safe_print(f"[CAN] Loi truyen du lieu: {e}")
            payload.can_bus_ok = False
            self._close_handle()

    # ── Nhận phản hồi từ ECU Alert (CAN ID 0x201) ─────────────────────────────

    def _recv_feedback(self):
        """
        Đọc không chặn các frame CAN phản hồi (0x201) từ ECU Alert ESP32.
        Giải mã và cập nhật trạng thái kết nối / sức khỏe phần cứng vào SharedState.
        """
        if not self.is_connected:
            return

        try:
            # Đọc không chặn (timeout=0)
            msg = self._bus.recv(timeout=0)
            if msg is None:
                return

            # Chỉ xử lý frame phản hồi từ ECU Alert
            if msg.arbitration_id != self.feedback_id:
                return
            if msg.dlc < 5:
                return

            # Giải mã frame 0x201
            executing_state = msg.data[0] & 0x03
            hw_health       = msg.data[1]
            echo_alive      = msg.data[2] & 0x0F
            esp32_alive     = msg.data[3] & 0x0F
            error_code      = msg.data[4]

            now = time.time()
            self._last_feedback_time = now
            self._alert_ecu_connected = True

            # Cập nhật SharedState nếu có
            if self._shared_state is not None:
                with self._shared_state.lock:
                    self._shared_state.alert_ecu_connected = True
                    self._shared_state.alert_ecu_state = executing_state
                    self._shared_state.alert_ecu_hw_health = hw_health
                    self._shared_state.alert_ecu_error_code = error_code
                    self._shared_state.last_feedback_time = now

        except Exception:
            # Không để lỗi recv làm crash thread
            pass

    def _update_watchdog(self):
        """Cập nhật trạng thái watchdog: ECU Alert offline nếu quá thời gian chờ."""
        if self._last_feedback_time == 0.0:
            # Chưa bao giờ nhận được phản hồi
            return

        now = time.time()
        was_connected = self._alert_ecu_connected

        if now - self._last_feedback_time > self.watchdog_timeout:
            self._alert_ecu_connected = False
            if was_connected:
                _safe_print("[CAN] ECU Alert OFFLINE — khong nhan duoc phan hoi")
        else:
            self._alert_ecu_connected = True

        # Cập nhật SharedState
        if self._shared_state is not None:
            with self._shared_state.lock:
                self._shared_state.alert_ecu_connected = self._alert_ecu_connected

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
            _safe_print("[CAN] Da dong ket noi CAN Bus.")
