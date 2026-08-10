"""
Unit tests riêng cho yawn detector trong FusionState.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.fusion import (
    FusionState,
    YAWN_MAR_THRESH,
    YAWN_ASPECT_MIN,
    YAWN_DURATION_MIN_SEC,
)


def make_mock_mlp(p=0.8):
    m = MagicMock()
    m.predict = MagicMock(return_value=np.array([[p]], dtype=np.float32))
    return m


def make_mock_lstm(p=0.8):
    m = MagicMock()
    m.predict = MagicMock(return_value=np.array([[p]], dtype=np.float32))
    return m


def make_mock_scaler():
    s = MagicMock()
    s.transform = MagicMock(side_effect=lambda x: np.zeros_like(x, dtype=float))
    return s


def feat(ear=0.30, mar=0.05, mouth_aspect=0.3, neck_tilt=0.0):
    return {
        "ear_left": ear, "ear_right": ear, "ear_avg": ear,
        "mar": mar, "mouth_aspect": mouth_aspect,
        "pitch": 0.0, "yaw": 0.0, "roll": 0.0,
        "neck_tilt": neck_tilt, "has_pose": True,
    }


def test_short_speech_not_yawn():
    """MAR cao nhưng ngắn (< YAWN_DURATION) → không phải ngáp."""
    fs = FusionState()
    mlp, lstm, sc = make_mock_mlp(), make_mock_lstm(), make_mock_scaler()
    t = 1_000_000.0
    fs.update(feat(), mlp, lstm, sc, sc, timestamp_ms=t)
    # Mở miệng 0.5s (< 1.2s)
    r = None
    for _ in range(5):
        t += 100
        r = fs.update(
            feat(mar=YAWN_MAR_THRESH + 0.2, mouth_aspect=YAWN_ASPECT_MIN + 0.2),
            mlp, lstm, sc, sc, timestamp_ms=t,
        )
    assert not r["yawn_alarm"]
    # Đóng miệng
    r = fs.update(feat(mar=0.1), mlp, lstm, sc, sc, timestamp_ms=t + 100)
    assert not r["yawn_alarm"]
    assert r["yawn_count_window"] == 0
    print("PASS test_short_speech_not_yawn")


def test_sustained_yawn_triggers():
    """Giữ MAR+aspect đủ lâu → yawn_alarm + count."""
    fs = FusionState()
    mlp, lstm, sc = make_mock_mlp(), make_mock_lstm(), make_mock_scaler()
    t = 2_000_000.0
    fs.update(feat(), mlp, lstm, sc, sc, timestamp_ms=t)

    frames = int(YAWN_DURATION_MIN_SEC * 10) + 3  # 100ms steps
    r = None
    saw_alarm = False
    for _ in range(frames):
        t += 100
        r = fs.update(
            feat(mar=0.7, mouth_aspect=0.7),
            mlp, lstm, sc, sc, timestamp_ms=t,
        )
        if r["yawn_alarm"]:
            saw_alarm = True
    assert saw_alarm, f"expected yawn_alarm after {YAWN_DURATION_MIN_SEC}s"
    assert r["yawn_count_window"] >= 1
    print(f"PASS test_sustained_yawn_triggers: count={r['yawn_count_window']} state={fs.yawn_state}")


def test_flat_mouth_not_yawn():
    """MAR cao nhưng mouth_aspect thấp (miệng ngang / nói) → không ngáp."""
    fs = FusionState()
    mlp, lstm, sc = make_mock_mlp(), make_mock_lstm(), make_mock_scaler()
    t = 3_000_000.0
    fs.update(feat(), mlp, lstm, sc, sc, timestamp_ms=t)
    r = None
    for _ in range(20):
        t += 100
        r = fs.update(
            feat(mar=0.7, mouth_aspect=0.20),  # dưới YAWN_ASPECT_MIN
            mlp, lstm, sc, sc, timestamp_ms=t,
        )
    assert not r["yawn_alarm"]
    assert fs.yawn_state == "IDLE"
    print("PASS test_flat_mouth_not_yawn")


def test_yawn_cooldown_blocks_double_trigger():
    fs = FusionState()
    mlp, lstm, sc = make_mock_mlp(), make_mock_lstm(), make_mock_scaler()
    t = 4_000_000.0
    fs.update(feat(), mlp, lstm, sc, sc, timestamp_ms=t)

    def hold_yawn(n):
        nonlocal t
        last = None
        for _ in range(n):
            t += 100
            last = fs.update(feat(mar=0.7, mouth_aspect=0.7), mlp, lstm, sc, sc, timestamp_ms=t)
        return last

    hold_yawn(20)
    assert fs.yawn_state in ("CONFIRMED", "COOLDOWN")
    # Đóng miệng → COOLDOWN
    for _ in range(5):
        t += 100
        fs.update(feat(mar=0.1), mlp, lstm, sc, sc, timestamp_ms=t)
    assert fs.yawn_state == "COOLDOWN"
    count_after_first = fs.yawn_counter.count()

    # Cố trigger lại ngay — phải bị block bởi cooldown
    hold_yawn(20)
    assert fs.yawn_counter.count() == count_after_first
    print(f"PASS test_yawn_cooldown_blocks_double_trigger: count={count_after_first}")


if __name__ == "__main__":
    test_short_speech_not_yawn()
    test_sustained_yawn_triggers()
    test_flat_mouth_not_yawn()
    test_yawn_cooldown_blocks_double_trigger()
    print("\nAll yawn tests passed.")
