"""
Unit tests riêng cho neck-tilt rule trong FusionState.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.fusion import FusionState


def make_mock_mlp(p=0.7):
    m = MagicMock()
    m.predict = MagicMock(return_value=np.array([[p]], dtype=np.float32))
    return m


def make_mock_lstm(p=0.7):
    m = MagicMock()
    m.predict = MagicMock(return_value=np.array([[p]], dtype=np.float32))
    return m


def make_mock_scaler():
    s = MagicMock()
    s.transform = MagicMock(side_effect=lambda x: np.zeros_like(x, dtype=float))
    return s


def feat(ear=0.30, neck_tilt=0.0):
    return {
        "ear_left": ear, "ear_right": ear, "ear_avg": ear,
        "mar": 0.05, "mouth_aspect": 0.3,
        "pitch": 0.0, "yaw": 0.0, "roll": 0.0,
        "neck_tilt": neck_tilt, "has_pose": True,
    }


def test_neck_alarm_on_sharp_nod():
    fs = FusionState()
    mlp, lstm, sc = make_mock_mlp(), make_mock_lstm(), make_mock_scaler()
    t = 1_000_000.0
    # Baseline
    for _ in range(20):
        t += 100
        fs.update(feat(neck_tilt=0.0), mlp, lstm, sc, sc, timestamp_ms=t)
    # Gật mạnh
    r = None
    for _ in range(10):
        t += 100
        r = fs.update(feat(neck_tilt=30.0), mlp, lstm, sc, sc, timestamp_ms=t)
    assert r["neck_alarm"], f"expected neck_alarm, got {r}"
    assert r["head_nod_count_window"] >= 1
    print(f"PASS test_neck_alarm_on_sharp_nod: count={r['head_nod_count_window']}")


def test_no_neck_alarm_when_still():
    fs = FusionState()
    mlp, lstm, sc = make_mock_mlp(), make_mock_lstm(), make_mock_scaler()
    t = 2_000_000.0
    r = None
    for _ in range(30):
        t += 100
        r = fs.update(feat(neck_tilt=0.0), mlp, lstm, sc, sc, timestamp_ms=t)
    assert not r["neck_alarm"]
    assert r["head_nod_count_window"] == 0
    print("PASS test_no_neck_alarm_when_still")


def test_neck_release_when_back_to_baseline():
    fs = FusionState()
    mlp, lstm, sc = make_mock_mlp(0.4), make_mock_lstm(0.4), make_mock_scaler()
    t = 3_000_000.0
    for _ in range(30):
        t += 100
        fs.update(feat(ear=0.18, neck_tilt=0.0), mlp, lstm, sc, sc, timestamp_ms=t)
    baseline = fs.neck_baseline
    for _ in range(30):
        t += 100
        fs.neck_baseline = baseline
        fs.update(feat(ear=0.18, neck_tilt=30.0), mlp, lstm, sc, sc, timestamp_ms=t)
    if not fs.alarm_on:
        fs.alarm_on = True
    released = False
    for _ in range(15):
        t += 100
        fs.neck_baseline = baseline
        r = fs.update(feat(ear=0.18, neck_tilt=0.0), mlp, lstm, sc, sc, timestamp_ms=t)
        if not r["alarm_on"]:
            released = True
            break
    assert released
    print("PASS test_neck_release_when_back_to_baseline")


if __name__ == "__main__":
    test_neck_alarm_on_sharp_nod()
    test_no_neck_alarm_when_still()
    test_neck_release_when_back_to_baseline()
    print("\nAll neck-tilt tests passed.")
