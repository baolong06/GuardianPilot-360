"""
Test PERCLOS integration với FusionState.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.fusion import FusionState


def make_mock_mlp(prob_non_drowsy=0.7):
    m = MagicMock()
    m.predict = MagicMock(return_value=np.array([[prob_non_drowsy]], dtype=np.float32))
    return m


def make_mock_lstm(prob_non_drowsy=0.7):
    m = MagicMock()
    m.predict = MagicMock(return_value=np.array([[prob_non_drowsy]], dtype=np.float32))
    return m


def make_mock_scaler():
    s = MagicMock()
    s.transform = MagicMock(side_effect=lambda x: np.zeros_like(x, dtype=float))
    return s


def feat_with(ear=0.3, pitch=0.0, neck_tilt=0.0):
    return {
        "ear_left": ear,
        "ear_right": ear,
        "ear_avg": ear,
        "mar": 0.05,
        "mouth_aspect": 0.3,
        "pitch": pitch,
        "yaw": 0.0,
        "roll": 0.0,
        "neck_tilt": neck_tilt,
        "has_pose": True,
    }


def test_perclos_in_fusion_output():
    """FusionState.update() phải trả về field 'perclos'"""
    fs = FusionState()
    mlp = make_mock_mlp(0.7)
    lstm = make_mock_lstm(0.7)
    scaler = make_mock_scaler()
    
    t = 0.0
    # Mắt nhắm 50% thời gian
    for i in range(60):
        ear = 0.30 if i % 2 == 0 else 0.10
        result = fs.update(feat_with(ear=ear), mlp, lstm, scaler, scaler, timestamp_ms=t)
        t += 100.0
    
    assert "perclos" in result, "Missing 'perclos' field in result"
    perclos = result["perclos"]
    
    # PERCLOS nên ~50%
    assert 0.45 < perclos < 0.55, f"Expected PERCLOS ~50%, got {perclos:.3f}"
    print(f"PASS test_perclos_in_fusion_output: perclos={perclos:.3f}")


def test_perclos_eyes_always_open():
    """Mắt luôn mở → PERCLOS = 0%"""
    fs = FusionState()
    mlp = make_mock_mlp(0.7)
    lstm = make_mock_lstm(0.7)
    scaler = make_mock_scaler()
    
    t = 0.0
    result = None
    for _ in range(50):
        result = fs.update(feat_with(ear=0.30), mlp, lstm, scaler, scaler, timestamp_ms=t)
        t += 100.0
    
    perclos = result["perclos"]
    assert perclos < 0.01, f"Expected PERCLOS ~0%, got {perclos:.3f}"
    print(f"PASS test_perclos_eyes_always_open: perclos={perclos:.3f}")


def test_perclos_eyes_always_closed():
    """Mắt luôn nhắm → PERCLOS = 100%"""
    fs = FusionState()
    mlp = make_mock_mlp(0.7)
    lstm = make_mock_lstm(0.7)
    scaler = make_mock_scaler()
    
    t = 0.0
    result = None
    for _ in range(50):
        result = fs.update(feat_with(ear=0.10), mlp, lstm, scaler, scaler, timestamp_ms=t)
        t += 100.0
    
    perclos = result["perclos"]
    assert perclos > 0.99, f"Expected PERCLOS ~100%, got {perclos:.3f}"
    print(f"PASS test_perclos_eyes_always_closed: perclos={perclos:.3f}")


def test_perclos_reset():
    """Reset phải xóa PERCLOS tracker state"""
    fs = FusionState()
    mlp = make_mock_mlp(0.7)
    lstm = make_mock_lstm(0.7)
    scaler = make_mock_scaler()
    
    t = 0.0
    # Mắt nhắm liên tục
    for _ in range(30):
        fs.update(feat_with(ear=0.10), mlp, lstm, scaler, scaler, timestamp_ms=t)
        t += 100.0
    
    assert fs.perclos_tracker.get_perclos() > 0.99
    
    # Reset
    fs.reset()
    
    assert fs.perclos_tracker.get_perclos() == 0.0
    print(f"PASS test_perclos_reset: perclos after reset={fs.perclos_tracker.get_perclos():.3f}")


if __name__ == "__main__":
    test_perclos_in_fusion_output()
    test_perclos_eyes_always_open()
    test_perclos_eyes_always_closed()
    test_perclos_reset()
    
    print("\n=== All PERCLOS integration tests passed ===")
