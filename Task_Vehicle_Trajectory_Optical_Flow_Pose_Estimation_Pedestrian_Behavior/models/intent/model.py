import torch
import torch.nn as nn


class IntentionGRU(nn.Module):
    """GRU-based binary classifier cho pedestrian crossing intention.

    INPUT FEATURES (per timestep, input_dim=6):
        [cx_norm, cy_norm, speed, heading, looking_flag, crossing_flag]
        - cx_norm, cy_norm : vị trí center bbox chuẩn hóa (0~1)
        - speed            : ||dx,dy|| / frame_width (normalized displacement)
        - heading          : arctan2(dy, dx) in radians
        - looking_flag     : 1.0 nếu pedestrian đang nhìn sang đường, 0.0 nếu không
        - crossing_flag    : 1.0 nếu crossing_action đang xảy ra, 0.0 nếu không

    LABEL (binary):
        0 = will NOT cross (intention_prob < threshold, hoặc crossing=-1)
        1 = WILL cross     (intention_prob >= threshold, hoặc crossing=1)

    CRITICAL — Phân biệt rõ:
        crossing_action   = nhãn hành vi HIỆN TẠI (đang diễn ra tại frame này).
        crossing_intention = DỰ ĐOÁN về tương lai (model này dự đoán cái này).
        Không được gộp hai nhãn này.

    Observation window: obs kết thúc TRƯỚC critical_point, không được
    include frame sau critical_point vào input (data leakage).

    Config mặc định:
        obs_len   = 16
        input_dim = 6
        hidden_dim = 128
        num_layers = 2
        num_classes = 2
    """

    def __init__(
        self,
        input_dim: int = 6,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, obs_len, input_dim)
        Returns:
            logits: (batch, num_classes)  — raw, chưa softmax
        """
        _, h = self.gru(x)
        return self.classifier(h[-1])
