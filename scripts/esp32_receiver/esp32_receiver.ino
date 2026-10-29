/*
 * ==============================================================================
 * DMS ESP32 Receiver — Nhận và giải mã Payload 8-byte từ DMS Edge-AI
 * ==============================================================================
 * Tốc độ Baud: 115200 bps
 * Cấu trúc gói tin 8 bytes:
 *   - Byte 0: State ID        (0: ALERT, 1: DROWSY/YAWN, 2: MICROSLEEP, 3: INIT)
 *   - Byte 1: Tracking Face   (0: NO_FACE, 1: FACE_OK)
 *   - Byte 2: Alert Level     (0: NORMAL, 1: WARNING, 2: CRITICAL)
 *   - Byte 3: Alive Counter   (0..15 rolling counter)
 *   - Byte 4: System Status   (Bit 0: Buffer Ready, Bit 1: CAN Bus OK)
 *   - Byte 5–7: Reserved      (0x00)
 * ==============================================================================
 */

// Định nghĩa chân LED báo hiệu (GPIO 2 thường là LED tích hợp trên ESP32 DevKit)
#define LED_PIN 2

// Cấu hình kích thước frame
const int PAYLOAD_SIZE = 8;
uint8_t rx_buffer[PAYLOAD_SIZE];
int rx_index = 0;
unsigned long last_rx_time = 0;
const unsigned long FRAME_TIMEOUT_MS = 50; // Timeout reset buffer nếu nhận dở dang

// Bảng tra cứu tên trạng thái
const char* STATE_NAMES[]   = {"ALERT", "DROWSY/YAWN", "MICROSLEEP", "INIT"};
const char* ALERT_NAMES[]   = {"NORMAL", "WARNING", "CRITICAL"};
const char* TRACKING_NAMES[]= {"NO_FACE", "FACE_OK"};

void setup() {
  // Khởi tạo cổng Serial giao tiếp với PC và Serial Monitor
  Serial.begin(115200);
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);

  // Chờ ổn định cổng serial
  delay(1000);
  Serial.println("\n==================================================");
  Serial.println("   DMS ESP32 Receiver Sẵn Sàng (Baud: 115200)   ");
  Serial.println("==================================================");
}

void loop() {
  // Nếu nhận dở dang và bị ngắt quãng quá lâu -> Reset index để đồng bộ frame
  if (rx_index > 0 && (millis() - last_rx_time > FRAME_TIMEOUT_MS)) {
    rx_index = 0;
  }

  // Đọc dữ liệu từ Serial buffer
  while (Serial.available() > 0) {
    uint8_t b = Serial.read();
    rx_buffer[rx_index++] = b;
    last_rx_time = millis();

    // Khi đã nhận đủ 8 bytes của 1 frame
    if (rx_index == PAYLOAD_SIZE) {
      process_dms_payload(rx_buffer);
      rx_index = 0; // Sẵn sàng cho frame tiếp theo
    }
  }
}

void process_dms_payload(const uint8_t* payload) {
  // 1. Trích xuất các trường dữ liệu theo đặc tả
  uint8_t driver_state  = payload[0] & 0x03; // Bit [1:0]
  uint8_t tracking_face = payload[1] & 0x01; // Bit [0]
  uint8_t alert_level   = payload[2] & 0x03; // Bit [1:0]
  uint8_t alive_counter = payload[3] & 0x0F; // Bit [3:0]
  
  bool buffer_ready = (payload[4] & 0x01) != 0; // Bit [0]
  bool can_bus_ok   = (payload[4] & 0x02) != 0; // Bit [1]

  // 2. Điều khiển LED cảnh báo
  if (alert_level == 2) {
    // Ngủ gật (CRITICAL) -> Bật sáng LED liên tục
    digitalWrite(LED_PIN, HIGH);
  } else if (alert_level == 1) {
    // Buồn ngủ (WARNING) -> Nhấp nháy LED
    digitalWrite(LED_PIN, (millis() / 200) % 2);
  } else {
    // Bình thường (NORMAL / INIT) -> Tắt LED
    digitalWrite(LED_PIN, LOW);
  }

  // 3. In thông tin chi tiết lên Serial Monitor
  Serial.println("--------------------------------------------------");
  
  // Dòng 1: In chuỗi Hex thô của frame
  Serial.print("HEX DUMP: [ ");
  for (int i = 0; i < PAYLOAD_SIZE; i++) {
    if (payload[i] < 0x10) Serial.print("0");
    Serial.print(payload[i], HEX);
    Serial.print(" ");
  }
  Serial.println("]");

  // Dòng 2: Chi tiết giải mã
  Serial.print("State: ");
  if (driver_state < 4) Serial.print(STATE_NAMES[driver_state]);
  else Serial.print("UNKNOWN");

  Serial.print(" | Face: ");
  if (tracking_face < 2) Serial.print(TRACKING_NAMES[tracking_face]);
  else Serial.print("UNKNOWN");

  Serial.print(" | Alert: ");
  if (alert_level < 3) Serial.print(ALERT_NAMES[alert_level]);
  else Serial.print("UNKNOWN");

  Serial.print(" | Alive: ");
  Serial.println(alive_counter);

  // Dòng 3: Trạng thái hệ thống
  Serial.print("System : Buffer=");
  Serial.print(buffer_ready ? "READY" : "BUFFERING");
  Serial.print(" | CAN_Bus=");
  Serial.println(can_bus_ok ? "OK" : "ERROR");
}
