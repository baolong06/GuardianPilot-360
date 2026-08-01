"""
Unit tests cho các model class.
Chạy: pytest tests/unit/test_models.py -v
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import pytest


# ─────────────────────────────────────────────────
# BehaviorGRU
# ─────────────────────────────────────────────────
class TestBehaviorGRU:
    def test_output_shape(self):
        from models.behavior_clf.model import BehaviorGRU
        m = BehaviorGRU(input_dim=4, hidden_dim=128, num_layers=2, num_classes=4)
        x = torch.randn(2, 10, 4)
        out = m(x)
        assert out.shape == (2, 4), f"Expected (2,4), got {out.shape}"

    def test_no_nan(self):
        from models.behavior_clf.model import BehaviorGRU
        m = BehaviorGRU()
        x = torch.randn(4, 10, 4)
        out = m(x)
        assert not torch.isnan(out).any(), "Output contains NaN"

    def test_num_layers_3(self):
        """Stage BehaviorStage dùng num_layers=3 — phải hoạt động."""
        from models.behavior_clf.model import BehaviorGRU
        m = BehaviorGRU(input_dim=4, hidden_dim=128, num_layers=3, num_classes=4)
        x = torch.randn(1, 10, 4)
        out = m(x)
        assert out.shape == (1, 4)

    def test_dropout_disabled_single_layer(self):
        """GRU dropout phải disabled khi num_layers=1 để tránh warning."""
        from models.behavior_clf.model import BehaviorGRU
        m = BehaviorGRU(input_dim=4, hidden_dim=64, num_layers=1, dropout=0.3)
        x = torch.randn(2, 5, 4)
        out = m(x)
        assert out.shape == (2, 4)


# ─────────────────────────────────────────────────
# PedestrianBehaviorGRU
# ─────────────────────────────────────────────────
class TestPedestrianBehaviorGRU:
    def test_output_shape(self):
        from models.pedestrian_behavior.model import PedestrianBehaviorGRU
        m = PedestrianBehaviorGRU(input_dim=4, hidden_dim=128, num_layers=3, num_classes=2)
        x = torch.randn(3, 10, 4)
        out = m(x)
        assert out.shape == (3, 2)

    def test_no_nan(self):
        from models.pedestrian_behavior.model import PedestrianBehaviorGRU
        m = PedestrianBehaviorGRU()
        x = torch.randn(2, 10, 4)
        out = m(x)
        assert not torch.isnan(out).any()

    def test_default_params_match_stage(self):
        """Default params phải khớp với PedestrianBehaviorStage defaults."""
        from models.pedestrian_behavior.model import PedestrianBehaviorGRU
        m = PedestrianBehaviorGRU()
        # Stage dùng: hidden_dim=128, num_layers=3, num_classes=2
        x = torch.randn(1, 10, 4)
        out = m(x)
        assert out.shape == (1, 2)


# ─────────────────────────────────────────────────
# TrajectoryLSTM
# ─────────────────────────────────────────────────
class TestTrajectoryLSTM:
    def test_output_shape(self):
        from models.trajectory.model import TrajectoryLSTM
        m = TrajectoryLSTM(input_dim=4, hidden_dim=128, num_layers=3, pred_len=12, obs_len=10)
        obs = torch.randn(2, 10, 4)
        pred = m(obs)
        assert pred.shape == (2, 12, 2), f"Expected (2,12,2), got {pred.shape}"

    def test_no_nan(self):
        from models.trajectory.model import TrajectoryLSTM
        m = TrajectoryLSTM()
        obs = torch.randn(4, 3, 4)
        pred = m(obs)
        assert not torch.isnan(pred).any()

    def test_pred_len_1(self):
        from models.trajectory.model import TrajectoryLSTM
        m = TrajectoryLSTM(pred_len=1, obs_len=5)
        obs = torch.randn(2, 5, 4)
        pred = m(obs)
        assert pred.shape == (2, 1, 2)


# ─────────────────────────────────────────────────
# IntentionGRU
# ─────────────────────────────────────────────────
class TestIntentionGRU:
    def test_output_shape(self):
        from models.intent.model import IntentionGRU
        m = IntentionGRU(input_dim=6, hidden_dim=128, num_layers=2, num_classes=2)
        x = torch.randn(4, 16, 6)
        out = m(x)
        assert out.shape == (4, 2)

    def test_no_nan(self):
        from models.intent.model import IntentionGRU
        m = IntentionGRU()
        x = torch.randn(2, 16, 6)
        out = m(x)
        assert not torch.isnan(out).any()

    def test_backward(self):
        from models.intent.model import IntentionGRU
        m = IntentionGRU()
        x = torch.randn(2, 16, 6, requires_grad=False)
        out = m(x)
        loss = out.mean()
        loss.backward()


# ─────────────────────────────────────────────────
# TrafficLightClassifier
# ─────────────────────────────────────────────────
class TestTrafficLightClassifier:
    def test_output_shape(self):
        from models.traffic_light.model import TrafficLightClassifier
        m = TrafficLightClassifier(num_classes=3)
        img = torch.randn(2, 3, 128, 128)
        out = m(img)
        assert out.shape == (2, 3)

    def test_no_nan(self):
        from models.traffic_light.model import TrafficLightClassifier
        m = TrafficLightClassifier(num_classes=3)
        img = torch.randn(1, 3, 128, 128)
        out = m(img)
        assert not torch.isnan(out).any()


# ─────────────────────────────────────────────────
# ADE/FDE metrics
# ─────────────────────────────────────────────────
class TestTrajectoryMetrics:
    def test_ade_fde_zero_error(self):
        from evaluation.metrics.trajectory import compute_ade_fde
        pred = torch.zeros(4, 12, 2)
        gt = torch.zeros(4, 12, 2)
        ade, fde = compute_ade_fde(pred, gt)
        assert abs(ade) < 1e-6
        assert abs(fde) < 1e-6

    def test_ade_fde_known_error(self):
        from evaluation.metrics.trajectory import compute_ade_fde
        pred = torch.ones(2, 5, 2)
        gt = torch.zeros(2, 5, 2)
        ade, fde = compute_ade_fde(pred, gt)
        # Khoảng cách L2 từ (1,1) đến (0,0) = sqrt(2)
        expected = (2 ** 0.5)
        assert abs(ade - expected) < 1e-4, f"ADE={ade}, expected ~{expected}"

    def test_miss_rate(self):
        from evaluation.metrics.trajectory import compute_miss_rate
        pred = torch.ones(4, 5, 2) * 10  # tất cả đều sai xa
        gt = torch.zeros(4, 5, 2)
        mr = compute_miss_rate(pred, gt, threshold=1.0)
        assert mr == 1.0
