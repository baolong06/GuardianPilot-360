"""Unit tests cho Alert Manager đa cấp."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.alert_manager import AlertManager, ALERT_MESSAGES, AlertStatus
from src.scoring import DriverState


def test_level_maps_to_state():
    mgr = AlertManager()
    for state in DriverState:
        status = mgr.update(state)
        assert status.alert_level == int(state)
        assert status.drowsiness_state == state.name
        assert status.alert_message == ALERT_MESSAGES[int(state)]
    print("PASS test_level_maps_to_state")


def test_changed_flag_and_callback():
    changes = []
    mgr = AlertManager(on_level_change=lambda s: changes.append(s.alert_level))

    s1 = mgr.update(DriverState.NORMAL)
    assert s1.changed is False  # khởi tạo level=0, vẫn 0
    assert changes == []

    s2 = mgr.update(DriverState.FATIGUE)
    assert s2.changed is True
    assert changes == [1]

    s3 = mgr.update(DriverState.FATIGUE)
    assert s3.changed is False
    assert changes == [1]

    s4 = mgr.update(DriverState.DROWSY)
    assert s4.changed is True
    assert changes == [1, 2]
    print(f"PASS test_changed_flag_and_callback: {changes}")


def test_auto_downgrade():
    mgr = AlertManager()
    mgr.update(DriverState.CRITICAL)
    status = mgr.update(DriverState.NORMAL)
    assert status.alert_level == 0
    assert status.changed is True
    print("PASS test_auto_downgrade")


def test_accepts_string_and_int():
    mgr = AlertManager()
    assert mgr.update("DROWSY").alert_level == 2
    assert mgr.update(3).alert_level == 3
    print("PASS test_accepts_string_and_int")


def test_reset_fires_callback_if_needed():
    changes = []
    mgr = AlertManager(on_level_change=lambda s: changes.append(s))
    mgr.update(DriverState.DROWSY)
    mgr.reset()
    assert mgr.alert_level == 0
    assert changes[-1].alert_level == 0
    print("PASS test_reset_fires_callback_if_needed")


if __name__ == "__main__":
    test_level_maps_to_state()
    test_changed_flag_and_callback()
    test_auto_downgrade()
    test_accepts_string_and_int()
    test_reset_fires_callback_if_needed()
    print("\nAll alert_manager tests passed.")
