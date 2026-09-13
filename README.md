# Driver Drowsiness Monitoring System (Edge AI)

Hệ thống giám sát và phát hiện buồn ngủ của tài xế theo thời gian thực sử dụng mạng nơ-ron kết hợp **CNN-TCN** tối ưu hóa trên **ONNX Runtime** (hỗ trợ tăng tốc GPU với CUDA).

---

## 📌 Tính năng chính

- **Trích xuất khuôn mặt & Landmarks:** Tích hợp MediaPipe Face Landmarker phát hiện khuôn mặt và tính toán tỉ lệ mở miệng (MAR - Mouth Aspect Ratio) để nhận diện hành vi ngáp.
- **Mô hình học sâu kết hợp (CNN-TCN):**
  - **Feature Extractor (CNN):** Trích xuất vector đặc trưng không gian từ khuôn mặt tài xế (chuẩn hóa ImageNet 224x224).
  - **Temporal Classifier (TCN):** Phân tích chuỗi đặc trưng thời gian (Sequence Length = 16) để xác định mức độ buồn ngủ.
- **Inference Tốc độ cao trên Edge:** Đóng gói mô hình dưới dạng ONNX Runtime hỗ trợ cả CPU và GPU (CUDA Execution Provider).
- **Bộ máy trạng thái (State Machine):** Lọc nhiễu qua EMA (Exponential Moving Average) và quản lý trạng thái: Tỉnh táo (Alert), Buồn ngủ (Drowsy), Đang ngáp (Yawning), v.v.

---

## 🗂️ Cấu trúc thư mục

```text
Drowsiness_Edge-ai/
├── models/                     # Chứa các file trọng số ONNX và MediaPipe task
│   ├── dms_feature_extractor.onnx
│   ├── dms_tcn_classifier.onnx
│   └── face_landmarker.task
├── src/                        # Mã nguồn chính của ứng dụng
│   ├── camera/                 # Quản lý luồng capture webcam
│   ├── classifier/             # State machine phân loại trạng thái
│   ├── detection/              # Xử lý trích xuất khuôn mặt với MediaPipe
│   ├── display/                # Renderer giao diện và overlay trực quan
│   ├── inference/              # Tiền xử lý và ONNX Runtime engine
│   ├── payload/                # Đóng gói dữ liệu đầu ra
│   ├── config.py               # Cấu hình tập trung (ngưỡng, kích thước, fps, ...)
│   ├── main.py                 # Điểm khởi chạy ứng dụng
│   └── shared_state.py         # Quản lý trạng thái đa luồng
└── trainings/                  # Pipeline huấn luyện & thử nghiệm
    ├── pipeline/               # Jupyter Notebooks huấn luyện CNN-TCN
    └── preview_result/         # Script kiểm thử và tối ưu mô hình ONNX
```

---

## 🚀 Cài đặt & Chạy ứng dụng

### 1. Yêu cầu hệ thống
- Python 3.10+
- Webcam
- (Tùy chọn) GPU NVIDIA hỗ trợ CUDA để tăng tốc độ xử lý

### 2. Cài đặt thư viện phụ thuộc

```bash
pip install numpy opencv-python onnxruntime mediapipe
# Hoặc cài onnxruntime-gpu nếu sử dụng card đồ họa NVIDIA:
# pip install onnxruntime-gpu
```

### 3. Chạy hệ thống

Khởi chạy từ thư mục gốc của dự án:

```bash
python -m src.main
```

---

## ⚙️ Cấu hình tùy chỉnh

Các tham số có thể điều chỉnh trong file `src/config.py`:
- `DROWSY_THRESHOLD`: Ngưỡng kích hoạt cảnh báo buồn ngủ (mặc định: `0.55`).
- `MAR_THRESHOLD`: Ngưỡng nhận diện ngáp qua tỷ lệ mở miệng (mặc định: `0.50`).
- `TARGET_FPS`: Tần suất suy luận (mặc định: `5 FPS`).
- `SEQ_LEN`: Độ dài chuỗi chu kỳ TCN (mặc định: `16 frames`).

---

## 📄 Bản quyền & Tác giả

Phát triển bởi **Nguyen Pham Thai Tri** ([@Clown-1412](https://github.com/Clown-1412)).
