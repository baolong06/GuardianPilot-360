"""Tests cho CameraObstructionDetector + EventFrequencyCounter."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.camera_obstruction import CameraObstructionDetector
from src.frequency import EventFrequencyCounter


def test_obstruction_requires_prior_face():
    det = CameraObstructionDetector(threshold_sec=1.0)
    # Chưa từng thấy face → không obstructed dù mất face lâu
    t = 0.0
    for _ in range(20):
        t += 100
        assert det.update(False, t) is False
    print("PASS test_obstruction_requires_prior_face")


def test_obstruction_after_10s_no_face():
    det = CameraObstructionDetector(threshold_sec=1.0)
    det.update(True, 0.0)
    obstructed = False
    t = 0.0
    for i in range(15):
        t += 100
        obstructed = det.update(False, t)
        if i < 9:
            assert obstructed is False
    assert obstructed is True
    # Thấy face lại → clear
    assert det.update(True, t + 100) is False
    print("PASS test_obstruction_after_10s_no_face")


def test_frequency_rising_edge_only():
    c = EventFrequencyCounter(window_sec=1.0)
    # Hold True nhiều frame → chỉ 1 event
    assert c.update(0, True) == 1
    assert c.update(100, True) == 1
    assert c.update(200, True) == 1
    # Release rồi trigger lại
    assert c.update(300, False) == 1
    assert c.update(400, True) == 2
    print("PASS test_frequency_rising_edge_only")


def test_frequency_window_expires():
    c = EventFrequencyCounter(window_sec=1.0)
    c.update(0, True)
    c.update(100, False)
    c.update(200, True)  # 2 events
    assert c.count() == 2
    # Sau 1.1s từ event đầu → event cũ bị drop
    n = c.update(1100, False)
    assert n == 1  # chỉ còn event ở t=200
    print(f"PASS test_frequency_window_expires: n={n}")


def test_frequency_reset():
    c = EventFrequencyCounter()
    c.update(0, True)
    c.reset()
    assert c.count() == 0
    print("PASS test_frequency_reset")


if __name__ == "__main__":
    test_obstruction_requires_prior_face()
    test_obstruction_after_10s_no_face()
    test_frequency_rising_edge_only()
    test_frequency_window_expires()
    test_frequency_reset()
    print("\nAll camera/frequency tests passed.")
