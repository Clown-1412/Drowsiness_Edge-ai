"""
Logic phân loại 3 cấp độ trạng thái tài xế + EMA smoothing.
Clone 100% từ run_webcam_onnx.py (L249-271).

Yawn timer (last_yawn_time) được quản lý bên ngoài (main thread cập nhật
mỗi frame qua SharedState), class này chỉ nhận is_recently_yawning làm input.
"""
from dataclasses import dataclass
from typing import Tuple


@dataclass
class DriverState:
    """Kết quả phân loại trạng thái tài xế."""
    level_id: int                        # 0=Ngủ gật, 1=Tỉnh táo, 2=Buồn ngủ/Ngáp
    smoothed_drowsy_prob: float
    label_text: str
    color: Tuple[int, int, int]          # BGR color cho OpenCV


class DriverStateMachine:
    """
    EMA smoothing + Phân loại 3 Level.
    CLONE 100% logic từ run_webcam_onnx.py — không thay đổi tham số.

    Tham số:
        drowsy_threshold: 0.55 (demo gốc)
        mar_threshold:    0.50 (demo gốc)
        alpha:            0.65 (demo gốc)
        yawn_cooldown_sec: 3.0 (demo gốc) — dùng để tham chiếu, nhưng
                          is_recently_yawning được tính bên ngoài.
    """

    def __init__(self, drowsy_threshold: float, mar_threshold: float,
                 alpha: float, yawn_cooldown_sec: float):
        self.drowsy_threshold  = drowsy_threshold
        self.mar_threshold     = mar_threshold
        self.alpha             = alpha
        self.yawn_cooldown_sec = yawn_cooldown_sec
        self.smoothed_drowsy_prob = 0.0

    def update(self, raw_drowsy_prob: float, current_mar: float,
               is_recently_yawning: bool) -> DriverState:
        """
        Cập nhật EMA và phân loại Level.

        Args:
            raw_drowsy_prob:     Xác suất buồn ngủ thô từ TCN (probs[0])
            current_mar:         MAR hiện tại (cập nhật mỗi frame bởi main thread)
            is_recently_yawning: True nếu vừa ngáp trong YAWN_COOLDOWN_SEC

        Returns:
            DriverState với level, probability, label và color

        Logic clone 100% từ run_webcam_onnx.py L249-271:
            smoothed = ALPHA * raw + (1 - ALPHA) * smoothed
            if smoothed >= 0.55:
                if mar >= 0.50 or is_recently_yawning:
                    → Level 2: BUỒN NGỦ (cam)
                else:
                    → Level 0: NGỦ GẬT (đỏ)
            else:
                → Level 1: TỈNH TÁO (xanh)
        """
        # ── EMA smoothing — giống hệt demo gốc L250 ──
        self.smoothed_drowsy_prob = (
            self.alpha * raw_drowsy_prob
            + (1.0 - self.alpha) * self.smoothed_drowsy_prob
        )

        # ── Phân loại 3 Levels — đồng bộ 100% với demo gốc L255-271 ──
        if self.smoothed_drowsy_prob >= self.drowsy_threshold:
            if current_mar >= self.mar_threshold or is_recently_yawning:
                # Đang ngáp HOẶC vừa ngáp xong đang chờ Model xả bộ đệm
                return DriverState(
                    level_id=2,
                    smoothed_drowsy_prob=self.smoothed_drowsy_prob,
                    label_text=(
                        f"LEVEL 2: BUON NGU "
                        f"({self.smoothed_drowsy_prob * 100:.1f}% "
                        f"| MAR: {current_mar:.2f})"
                    ),
                    color=(0, 165, 255),   # Cam
                )
            else:
                # Model báo buồn ngủ và KHÔNG có hành vi ngáp gần đây → Ngủ gật
                return DriverState(
                    level_id=0,
                    smoothed_drowsy_prob=self.smoothed_drowsy_prob,
                    label_text=(
                        f"LEVEL 0: NGU GAT "
                        f"({self.smoothed_drowsy_prob * 100:.1f}%)"
                    ),
                    color=(0, 0, 255),     # Đỏ
                )
        else:
            # Model đã hạ về dưới ngưỡng an toàn → Tỉnh táo
            smoothed_alert_prob = 1.0 - self.smoothed_drowsy_prob
            return DriverState(
                level_id=1,
                smoothed_drowsy_prob=self.smoothed_drowsy_prob,
                label_text=f"LEVEL 1: ALERT ({smoothed_alert_prob * 100:.1f}%)",
                color=(0, 255, 0),         # Xanh lá
            )
