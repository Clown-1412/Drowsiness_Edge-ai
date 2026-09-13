import os
from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort
import timm
import torch
import torch.nn as nn

# ── Cấu hình & Thiết bị ──
SCRIPT_DIR = 'D:\\Python_pj\\4_model'
MODEL_PATH = str(Path(SCRIPT_DIR) / 'best_model.pth')
OUTPUT_DIR = Path(SCRIPT_DIR) / 'onnx_models'
OUTPUT_DIR.mkdir(exist_ok=True)

EXTRACTOR_ONNX_PATH  = str(OUTPUT_DIR / 'dms_feature_extractor.onnx')
CLASSIFIER_ONNX_PATH = str(OUTPUT_DIR / 'dms_tcn_classifier.onnx')

device = torch.device('cpu')  # Export bằng CPU để file ONNX mang tính tổng quát

# ── 1. Định nghĩa Kiến trúc Model ──
class TemporalAttention(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden, hidden // 4),
            nn.Tanh(),
            nn.Linear(hidden // 4, 1)
        )

    def forward(self, x):
        weights = torch.softmax(self.attn(x), dim=1)
        return (x * weights).sum(dim=1)


class MultiRegionCNNTCN(nn.Module):
    def __init__(self, backbone='tf_efficientnet_lite2', hidden=256, num_classes=2, seq_len=16):
        super().__init__()
        self.seq_len = seq_len
        self.cnn_face  = timm.create_model(backbone, pretrained=False, num_classes=0)
        self.cnn_eye   = timm.create_model(backbone, pretrained=False, num_classes=0)
        self.cnn_mouth = timm.create_model(backbone, pretrained=False, num_classes=0)
        feat_dim = self.cnn_face.num_features

        self.region_attn = nn.Sequential(
            nn.Linear(feat_dim * 3, 3),
            nn.Softmax(dim=1)
        )

        self.fusion = nn.Sequential(
            nn.Linear(feat_dim * 3, hidden),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.tcn = nn.Sequential(
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(0.2),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(0.2),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=4, dilation=4),
            nn.BatchNorm1d(hidden), nn.ReLU(),
        )

        self.temporal_attn = TemporalAttention(hidden)
        self.image_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.classifier = nn.Linear(hidden, num_classes)

    def extract_regions(self, face, eye, mouth):
        f_feat = self.cnn_face(face)
        e_feat = self.cnn_eye(eye)
        m_feat = self.cnn_mouth(mouth)
        combined = torch.cat([f_feat, e_feat, m_feat], dim=1)
        weights  = self.region_attn(combined)
        weighted = torch.cat([
            f_feat * weights[:, 0:1],
            e_feat * weights[:, 1:2],
            m_feat * weights[:, 2:3],
        ], dim=1)
        return self.fusion(weighted), weights

    def forward_sequence_from_features(self, feat_seq):
        feats = feat_seq.permute(0, 2, 1)
        feats = self.tcn(feats)
        feats = feats.permute(0, 2, 1)
        feats = self.temporal_attn(feats)
        return self.classifier(feats)


# ── 2. Wrapper phục vụ Export ──
class ONNXFeatureExtractor(nn.Module):
    def __init__(self, main_model):
        super().__init__()
        self.main_model = main_model

    def forward(self, face, eye, mouth):
        feat, weights = self.main_model.extract_regions(face, eye, mouth)
        return feat, weights


class ONNXTCNClassifier(nn.Module):
    def __init__(self, main_model):
        super().__init__()
        self.main_model = main_model

    def forward(self, feat_seq):
        logits = self.main_model.forward_sequence_from_features(feat_seq)
        probs  = torch.softmax(logits, dim=1)
        return probs


# ── 3. Nạp Trọng số PyTorch ──
base_model = MultiRegionCNNTCN(seq_len=16).to(device)
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Không tìm thấy checkpoint: {MODEL_PATH}")

state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
if 'model_state_dict' in state_dict:
    base_model.load_state_dict(state_dict['model_state_dict'], strict=False)
else:
    base_model.load_state_dict(state_dict, strict=False)
base_model.eval()

extractor_module  = ONNXFeatureExtractor(base_model).eval()
classifier_module = ONNXTCNClassifier(base_model).eval()


# ── 4. Thực hiện Export ONNX ──
print("🚀 Bắt đầu Export ONNX...")

# A. Export Feature Extractor
dummy_face  = torch.randn(1, 3, 224, 224, dtype=torch.float32)
dummy_eye   = torch.randn(1, 3, 224, 224, dtype=torch.float32)
dummy_mouth = torch.randn(1, 3, 224, 224, dtype=torch.float32)

torch.onnx.export(
    extractor_module,
    (dummy_face, dummy_eye, dummy_mouth),
    EXTRACTOR_ONNX_PATH,
    export_params=True,
    opset_version=17,
    do_constant_folding=True,
    input_names=['face_input', 'eye_input', 'mouth_input'],
    output_names=['feature_vector', 'attention_weights'],
    dynamic_axes={
        'face_input': {0: 'batch_size'},
        'eye_input': {0: 'batch_size'},
        'mouth_input': {0: 'batch_size'},
        'feature_vector': {0: 'batch_size'},
        'attention_weights': {0: 'batch_size'}
    }
)
print(f"✅ Đã lưu Feature Extractor: {EXTRACTOR_ONNX_PATH}")

# B. Export TCN Classifier
dummy_feat_seq = torch.randn(1, 16, 256, dtype=torch.float32)

torch.onnx.export(
    classifier_module,
    dummy_feat_seq,
    CLASSIFIER_ONNX_PATH,
    export_params=True,
    opset_version=17,
    do_constant_folding=True,
    input_names=['sequence_features'],
    output_names=['probabilities'],
    dynamic_axes={
        'sequence_features': {0: 'batch_size'},
        'probabilities': {0: 'batch_size'}
    }
)
print(f"✅ Đã lưu TCN Classifier   : {CLASSIFIER_ONNX_PATH}")


# ── 5. Kiểm tra tính toàn vẹn và đối chiếu kết quả (Validation) ──
print("\n🔍 Đang kiểm tra tính tương thích giữa PyTorch và ONNX Runtime...")

# Check cấu trúc đồ thị ONNX
onnx.checker.check_model(onnx.load(EXTRACTOR_ONNX_PATH))
onnx.checker.check_model(onnx.load(CLASSIFIER_ONNX_PATH))

# Chạy thử với ONNX Runtime
ort_extractor  = ort.InferenceSession(EXTRACTOR_ONNX_PATH, providers=['CPUExecutionProvider'])
ort_classifier = ort.InferenceSession(CLASSIFIER_ONNX_PATH, providers=['CPUExecutionProvider'])

with torch.no_grad():
    pt_feat, pt_w = extractor_module(dummy_face, dummy_eye, dummy_mouth)
    pt_probs      = classifier_module(dummy_feat_seq)

ort_feat, ort_w = ort_extractor.run(
    None, 
    {
        'face_input': dummy_face.numpy(),
        'eye_input': dummy_eye.numpy(),
        'mouth_input': dummy_mouth.numpy()
    }
)
ort_probs = ort_classifier.run(None, {'sequence_features': dummy_feat_seq.numpy()})[0]

diff_feat  = np.max(np.abs(pt_feat.numpy() - ort_feat))
diff_probs = np.max(np.abs(pt_probs.numpy() - ort_probs))

print(f"  • Độ lệch trích xuất đặc trưng (Max Diff Feature): {diff_feat:.2e}")
print(f"  • Độ lệch kết quả phân loại (Max Diff Probs)     : {diff_probs:.2e}")

if diff_probs < 1e-4:
    print("\n🎉 Export thành công hoàn toàn! Trọng số và đầu ra giữa PyTorch và ONNX đồng nhất.")