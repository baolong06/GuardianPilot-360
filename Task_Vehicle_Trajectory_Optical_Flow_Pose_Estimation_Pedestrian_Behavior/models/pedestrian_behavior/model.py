import torch
import torch.nn as nn


class PedestrianBehaviorGRU(nn.Module):
    """GRU classifier cho pedestrian behavior (walking / standing).

    Default params khớp với training/pedestrian_behavior/train.py và
    pipeline/stages/pedestrian_behavior_stage.py:
        input_dim=4 : [cx_norm, cy_norm, speed, heading]
        hidden_dim=128
        num_layers=3
        num_classes=2  : 0=standing, 1=walking
        dropout=0.2    : chỉ active khi num_layers > 1
    """

    def __init__(
        self,
        input_dim: int = 4,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_classes: int = 2,
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