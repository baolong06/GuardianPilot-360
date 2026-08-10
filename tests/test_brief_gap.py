"""Tests for looking-away, phone heuristic, context, trip, privacy, channels."""
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.looking_away import LookingAwayDetector
from src.phone_distraction import HandNearFacePhoneHeuristic
from src.context import DrivingContext
from src.trip_memory import TripMemory
from src.event_logger import EventLogger
from src.alert_manager import AlertManager, channels_for_level
from src.scoring import DriverState
from src import thresholds as threshold_store


def test_looking_away_requires_duration():
    d = LookingAwayDetector(yaw_thresh_deg=25, min_duration_sec=1.0)
    r = d.update(0, 40.0)
    assert r["looking_away"] is False
    r = d.update(500, 40.0)
    assert r["looking_away"] is False
    r = d.update(1100, 40.0)
    assert r["looking_away"] is True
    r = d.update(1200, 0.0)
    assert r["looking_away"] is False
    print("PASS test_looking_away_requires_duration")


def test_phone_near_face():
    d = HandNearFacePhoneHeuristic(near_frac=0.5, min_duration_sec=1.0)
    # wrist far
    r = d.update(0, face_center=(100, 100), face_size=80, wrists=[(400, 400)])
    assert r["phone_suspected"] is False
    # wrist near for 1.1s
    for t in range(0, 1200, 100):
        r = d.update(t, face_center=(100, 100), face_size=80, wrists=[(110, 110)])
    assert r["phone_suspected"] is True
    print("PASS test_phone_near_face")


def test_driving_context_risk():
    ctx = DrivingContext(high_speed_kmh=80, long_drive_sec=100)
    ctx.set_speed(90)
    ctx._last_tick = 0
    ctx.update(now=150)  # +150s driving
    assert ctx.risk_multiplier() > 1.0
    boosted = ctx.apply_to_score(0.5)
    assert boosted > 0.5
    print(f"PASS test_driving_context_risk: m={ctx.risk_multiplier()}")


def test_trip_memory_summary():
    m = TripMemory()
    m.update(perclos=0.2, drowsiness_state="NORMAL", alert_level=0)
    m.update(perclos=0.8, drowsiness_state="DROWSY", alert_level=2, looking_away=True)
    s = m.summary(driving_time_sec=120)
    assert s["samples"] == 2
    assert s["perclos_peak"] == 0.8
    assert s["alert_peak"] == 2
    assert s["looking_away_frames"] == 1
    print("PASS test_trip_memory_summary")


def test_privacy_default_no_snapshot():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        logger = EventLogger(
            db_path=tmp / "e.db",
            snapshot_dir=tmp / "s",
            save_face_snapshots=False,
        )
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        eid = logger.log_event(3, frame=frame, driver_id="d1")
        ev = logger.get_event(eid)
        assert ev["snapshot_path"] is None
        pending = logger.get_pending_upload()
        payload = logger.to_sync_payload(pending)
        assert "snapshot_path" not in payload[0]
        print("PASS test_privacy_default_no_snapshot")


def test_privacy_debug_snapshot_opt_in():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        logger = EventLogger(
            db_path=tmp / "e.db",
            snapshot_dir=tmp / "s",
            save_face_snapshots=True,
        )
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        eid = logger.log_event(2, frame=frame)
        ev = logger.get_event(eid)
        assert ev["snapshot_path"] is not None
        assert Path(ev["snapshot_path"]).is_file()
        print("PASS test_privacy_debug_snapshot_opt_in")


def test_alert_channels():
    assert channels_for_level(0) == {
        "sound": False, "vibration": False, "break_suggested": False
    }
    assert channels_for_level(2)["sound"] is True
    assert channels_for_level(3)["vibration"] is True
    mgr = AlertManager()
    st = mgr.update(DriverState.CRITICAL)
    assert st.channels["sound"] and st.channels["vibration"] and st.channels["break_suggested"]
    print("PASS test_alert_channels")


def test_thresholds_hitl():
    threshold_store.reset_thresholds()
    th = threshold_store.update_thresholds({"yaw_thresh_deg": 30}, actor="test")
    assert th["yaw_thresh_deg"] == 30.0
    assert threshold_store.audit_log()[-1]["actor"] == "test"
    threshold_store.reset_thresholds()
    print("PASS test_thresholds_hitl")


if __name__ == "__main__":
    test_looking_away_requires_duration()
    test_phone_near_face()
    test_driving_context_risk()
    test_trip_memory_summary()
    test_privacy_default_no_snapshot()
    test_privacy_debug_snapshot_opt_in()
    test_alert_channels()
    test_thresholds_hitl()
    print("\nAll brief-gap tests passed.")
