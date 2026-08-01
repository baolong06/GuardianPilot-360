import torch
import torch.nn as nn


class BehaviorGRU(nn.Module):
    """GRU classifier cho vehicle/agent behavior.

    Defaults khớp với training/behavior_clf/train.py:
        input_dim=4    : [cx_norm, cy_norm, speed, heading]
        hidden_dim=128
        num_layers=2   : behavior_clf training config
        num_classes=4  : stop | straight | turn_left | turn_right
        dropout=0.2    : chỉ active khi num_layers > 1

    BehaviorStage có thể override num_layers=3 nếu checkpoint được
    train với config đó (vd: training/behavior_clf/config đặt num_layers=3).
    """

    def __init__(
        self,
        input_dim: int = 4,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            logits: (batch, num_classes)  — raw, chưa softmax
        """
        _, h = self.gru(x)
        return self.fc(h[-1])