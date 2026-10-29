#include "driver/twai.h"

// Chân thực tế đang nối module TJA1050 của bạn
#define CAN_TX_PIN GPIO_NUM_26
#define CAN_RX_PIN GPIO_NUM_27

// Chân LED tích hợp (báo động khi ngủ gật)
#define LED_PIN 2

// ID tiêu chuẩn của DMS
#define DMS_CAN_ID 0x200

// Bảng tra cứu tên trạng thái
const char* STATE_NAMES[]    = {"ALERT", "DROWSY/YAWN", "MICROSLEEP", "INIT"};
const char* ALERT_NAMES[]    = {"NORMAL", "WARNING", "CRITICAL"};
const char* TRACKING_NAMES[] = {"NO_FACE", "FACE_OK"};

void setup() {
  Serial.begin(115200);
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);

  delay(1000);
  Serial.println("\n========================================================");
  Serial.println("  ESP32 CAN Receiver (TJA1050: TX=26, RX=27 | 500kbps) ");
  Serial.println("========================================================");

  twai_general_config_t g_config = TWAI_GENERAL_CONFIG_DEFAULT(CAN_TX_PIN, CAN_RX_PIN, TWAI_MODE_NORMAL);
  twai_timing_config_t t_config = TWAI_TIMING_CONFIG_500KBITS();
  twai_filter_config_t f_config = TWAI_FILTER_CONFIG_ACCEPT_ALL();

  if (twai_driver_install(&g_config, &t_config, &f_config) == ESP_OK) {
    Serial.println("✅ Driver CAN đã cài đặt thành công!");
  } else {
    Serial.println("❌ Lỗi cài đặt driver CAN! Kiểm tra lại phần cứng.");
    return;
  }

  if (twai_start() == ESP_OK) {
    Serial.println("🚀 Đã khởi động CAN Bus (500 kbps) sẵn sàng nhận dữ liệu!");
  } else {
    Serial.println("❌ Lỗi khởi động CAN Bus!");
  }
}

void loop() {
  twai_message_t rx_msg;

  // Chờ đọc dữ liệu từ CAN Bus (timeout 50ms)
  if (twai_receive(&rx_msg, pdMS_TO_TICKS(50)) == ESP_OK) {
    if (!rx_msg.rtr) {
      // TRƯỜNG HỢP 1: Nhận đúng gói DMS chuẩn (ID 0x200 và 8 bytes)
      if (rx_msg.identifier == DMS_CAN_ID && rx_msg.data_length_code == 8) {
        process_dms_payload(rx_msg.data);
      } 
      // TRƯỜNG HỢP 2: Nhận frame test bất kỳ (ví dụ lệnh cansend can0 ...)
      else {
        Serial.printf("[CAN IN] ID: 0x%03X | DLC: %d | Data: ", rx_msg.identifier, rx_msg.data_length_code);
        for (int i = 0; i < rx_msg.data_length_code; i++) {
          Serial.printf("%02X ", rx_msg.data[i]);
        }
        Serial.println();
      }
    }
  }

  // Tự động phục hồi khi Bus bị ngắt / chập đường truyền
  uint32_t alerts;
  twai_read_alerts(&alerts, pdMS_TO_TICKS(0));
  if (alerts & TWAI_ALERT_BUS_OFF) {
    Serial.println("⚠️ Phát hiện Bus-off! Đang khôi phục...");
    twai_initiate_recovery();
  }
  if (alerts & TWAI_ALERT_BUS_RECOVERED) {
    Serial.println("✅ Bus đã khôi phục. Bắt đầu lại...");
    twai_start();
  }
}

// Hàm giải mã Payload 8 bytes của DMS
void process_dms_payload(const uint8_t* payload) {
  uint8_t driver_state  = payload[0] & 0x03; // Bit [1:0]
  uint8_t tracking_face = payload[1] & 0x01; // Bit [0]
  uint8_t alert_level   = payload[2] & 0x03; // Bit [1:0]
  uint8_t alive_counter = payload[3] & 0x0F; // Bit [3:0]
  
  bool buffer_ready = (payload[4] & 0x01) != 0;
  bool can_bus_ok   = (payload[4] & 0x02) != 0;

  // Điều khiển LED cảnh báo
  if (alert_level == 2) {
    digitalWrite(LED_PIN, HIGH); // Ngủ gật -> Bật sáng
  } else if (alert_level == 1) {
    digitalWrite(LED_PIN, (millis() / 200) % 2); // Buồn ngủ -> Nháy
  } else {
    digitalWrite(LED_PIN, LOW); // Bình thường -> Tắt
  }

  // In ra Serial Monitor
  Serial.println("--------------------------------------------------------");
  Serial.printf("DMS [0x200] HEX: [ ");
  for (int i = 0; i < 8; i++) {
    Serial.printf("%02X ", payload[i]);
  }
  Serial.println("]");

  Serial.printf("Trạng thái: %-12s | Mặt: %-7s | Cảnh báo: %-8s | Alive: %d\n",
                (driver_state < 4) ? STATE_NAMES[driver_state] : "UNKNOWN",
                (tracking_face < 2) ? TRACKING_NAMES[tracking_face] : "UNKNOWN",
                (alert_level < 3) ? ALERT_NAMES[alert_level] : "UNKNOWN",
                alive_counter);

  Serial.printf("Hệ thống  : Buffer=%s | Bus=%s\n",
                buffer_ready ? "SẴN SÀNG" : "BUFFERING",
                can_bus_ok ? "OK" : "LỖI");
}