"""
Unit tests cho PERCLOS Tracker.

Test scenarios:
- Mắt luôn mở → PERCLOS = 0%
- Mắt luôn nhắm → PERCLOS = 100%
- Mắt nhắm 50% thời gian → PERCLOS = 50%
- Rolling window: samples cũ bị loại bỏ đúng cách
- Edge case: ít samples, window chưa đầy
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.perclos import PERCLOSTracker


def test_eyes_always_open():
    """Mắt luôn mở → PERCLOS = 0%"""
    tracker = PERCLOSTracker(window_sec=30.0, eye_closed_threshold=0.18)
    
    t = 0.0
    for _ in range(50):
        perclos = tracker.update(t, ear_smooth=0.30)  # EAR cao = mắt mở
        t += 100.0  # mỗi 100ms
    
    assert perclos < 0.01, f"Expected PERCLOS ~0%, got {perclos:.3f}"
    print(f"PASS test_eyes_always_open: PERCLOS={perclos:.3f}")


def test_eyes_always_closed():
    """Mắt luôn nhắm → PERCLOS = 100%"""
    tracker = PERCLOSTracker(window_sec=30.0, eye_closed_threshold=0.18)
    
    t = 0.0
    for _ in range(50):
        perclos = tracker.update(t, ear_smooth=0.10)  # EAR thấp = mắt nhắm
        t += 100.0
    
    assert perclos > 0.99, f"Expected PERCLOS ~100%, got {perclos:.3f}"
    print(f"PASS test_eyes_always_closed: PERCLOS={perclos:.3f}")


def test_eyes_half_closed():
    """Mắt nhắm 50% thời gian → PERCLOS ~50%"""
    tracker = PERCLOSTracker(window_sec=30.0, eye_closed_threshold=0.18)
    
    t = 0.0
    for i in range(60):
        # Xen kẽ: mở → nhắm → mở → nhắm
        if i % 2 == 0:
            ear = 0.30  # mở
        else:
            ear = 0.10  # nhắm
        
        perclos = tracker.update(t, ear_smooth=ear)
        t += 100.0
    
    # PERCLOS nên ~50% (có thể lệch chút do discrete sampling)
    assert 0.45 < perclos < 0.55, f"Expected PERCLOS ~50%, got {perclos:.3f}"
    print(f"PASS test_eyes_half_closed: PERCLOS={perclos:.3f}")


def test_rolling_window_drops_old_samples():
    """Samples cũ ngoài 30s phải bị loại bỏ"""
    tracker = PERCLOSTracker(window_sec=5.0, eye_closed_threshold=0.18)  # window ngắn 5s
    
    t = 0.0
    # Phase 1: mắt nhắm liên tục 3s
    for _ in range(30):
        tracker.update(t, ear_smooth=0.10)
        t += 100.0
    
    perclos_closed = tracker.get_perclos()
    assert perclos_closed > 0.99, f"Phase 1: expected PERCLOS ~100%, got {perclos_closed:.3f}"
    
    # Phase 2: mắt mở liên tục 6s (vượt window 5s)
    for _ in range(60):
        tracker.update(t, ear_smooth=0.30)
        t += 100.0
    
    perclos_open = tracker.get_perclos()
    # Sau 6s mở mắt, window 5s gần nhất chỉ chứa samples mắt mở → PERCLOS ~0%
    assert perclos_open < 0.01, f"Phase 2: expected PERCLOS ~0%, got {perclos_open:.3f}"
    
    print(f"PASS test_rolling_window_drops_old_samples: Phase1={perclos_closed:.3f}, Phase2={perclos_open:.3f}")


def test_insufficient_data():
    """Edge case: ít hơn 2 samples → PERCLOS = 0"""
    tracker = PERCLOSTracker(window_sec=30.0, eye_closed_threshold=0.18)
    
    # Chỉ có 1 sample
    perclos = tracker.update(0.0, ear_smooth=0.10)
    assert perclos == 0.0, f"Expected 0.0 with 1 sample, got {perclos:.3f}"
    
    print(f"PASS test_insufficient_data: PERCLOS={perclos:.3f}")


def test_gradual_buildup():
    """PERCLOS tăng dần khi mắt bắt đầu nhắm nhiều hơn"""
    tracker = PERCLOSTracker(window_sec=10.0, eye_closed_threshold=0.18)
    
    t = 0.0
    perclos_values = []
    
    # Phase 1: mắt mở 5s
    for _ in range(50):
        perclos = tracker.update(t, ear_smooth=0.30)
        t += 100.0
    perclos_values.append(perclos)
    
    # Phase 2: mắt nhắm 5s
    for _ in range(50):
        perclos = tracker.update(t, ear_smooth=0.10)
        t += 100.0
    perclos_values.append(perclos)
    
    # PERCLOS phải tăng từ ~0% lên ~50% (vì window 10s chứa 5s mở + 5s nhắm)
    assert perclos_values[0] < 0.05, f"Phase 1: expected low PERCLOS, got {perclos_values[0]:.3f}"
    assert 0.45 < perclos_values[1] < 0.55, f"Phase 2: expected ~50% PERCLOS, got {perclos_values[1]:.3f}"
    
    print(f"PASS test_gradual_buildup: Phase1={perclos_values[0]:.3f}, Phase2={perclos_values[1]:.3f}")


def test_reset():
    """Reset phải xóa toàn bộ state"""
    tracker = PERCLOSTracker(window_sec=30.0, eye_closed_threshold=0.18)
    
    # Thêm data
    t = 0.0
    for _ in range(30):
        tracker.update(t, ear_smooth=0.10)
        t += 100.0
    
    assert tracker.get_perclos() > 0.99
    
    # Reset
    tracker.reset()
    
    assert tracker.get_perclos() == 0.0
    assert len(tracker.samples) == 0
    
    print(f"PASS test_reset: PERCLOS after reset={tracker.get_perclos():.3f}")


def test_threshold_boundary():
    """Test ngưỡng EAR đúng boundary (0.18)"""
    tracker = PERCLOSTracker(window_sec=30.0, eye_closed_threshold=0.18)
    
    t = 0.0
    # EAR = 0.18 (đúng bằng threshold) → nên coi là MỞ (not closed)
    for _ in range(30):
        tracker.update(t, ear_smooth=0.18)
        t += 100.0
    
    perclos_at_threshold = tracker.get_perclos()
    
    # EAR = 0.17 (dưới threshold) → nhắm
    tracker.reset()
    t = 0.0
    for _ in range(30):
        tracker.update(t, ear_smooth=0.17)
        t += 100.0
    
    perclos_below_threshold = tracker.get_perclos()
    
    assert perclos_at_threshold < 0.01, f"At threshold: expected open, got PERCLOS={perclos_at_threshold:.3f}"
    assert perclos_below_threshold > 0.99, f"Below threshold: expected closed, got PERCLOS={perclos_below_threshold:.3f}"
    
    print(f"PASS test_threshold_boundary: at_threshold={perclos_at_threshold:.3f}, below={perclos_below_threshold:.3f}")


if __name__ == "__main__":
    test_eyes_always_open()
    test_eyes_always_closed()
    test_eyes_half_closed()
    test_rolling_window_drops_old_samples()
    test_insufficient_data()
    test_gradual_buildup()
    test_reset()
    test_threshold_boundary()
    
    print("\n=== All PERCLOS tests passed ===")
