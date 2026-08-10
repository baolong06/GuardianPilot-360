"""Unit tests cho Event Logger (SQLite)."""
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.event_logger import EventLogger


def _make_logger(tmp: Path, save_face_snapshots: bool = False) -> EventLogger:
    return EventLogger(
        db_path=tmp / "events.db",
        snapshot_dir=tmp / "snaps",
        save_face_snapshots=save_face_snapshots,
    )


def test_log_and_query():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        logger = _make_logger(tmp)
        eid = logger.log_event(
            2,
            driver_id="d1",
            vehicle_id="v1",
            ear_avg=0.15,
            perclos=0.6,
            neck_tilt=12.0,
            gps_lat=10.7,
            gps_lng=106.6,
        )
        assert eid >= 1
        events = logger.get_events(driver_id="d1")
        assert len(events) == 1
        e = events[0]
        assert e["alert_level"] == 2
        assert e["uploaded"] is False
        assert e["gps_lat"] == 10.7
        print(f"PASS test_log_and_query: id={eid}")


def test_snapshot_saved_when_level_ge_2():
    """DEBUG opt-in: save_face_snapshots=True mới ghi JPEG."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        logger = _make_logger(tmp, save_face_snapshots=True)
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        eid = logger.log_event(2, driver_id="d1", frame=frame)
        event = logger.get_event(eid)
        assert event["snapshot_path"] is not None
        assert Path(event["snapshot_path"]).is_file()
        print(f"PASS test_snapshot_saved_when_level_ge_2: {event['snapshot_path']}")


def test_default_privacy_no_snapshot():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        logger = _make_logger(tmp, save_face_snapshots=False)
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        eid = logger.log_event(3, frame=frame)
        event = logger.get_event(eid)
        assert event["snapshot_path"] is None
        print("PASS test_default_privacy_no_snapshot")


def test_no_snapshot_for_fatigue():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        logger = _make_logger(tmp)
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        eid = logger.log_event(1, frame=frame)
        event = logger.get_event(eid)
        assert event["snapshot_path"] is None
        print("PASS test_no_snapshot_for_fatigue")


def test_mark_uploaded_sync():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        logger = _make_logger(tmp)
        ids = [logger.log_event(2, driver_id="d1") for _ in range(3)]
        pending = logger.get_pending_upload()
        assert len(pending) == 3
        n = logger.mark_uploaded(ids[:2])
        assert n == 2
        pending2 = logger.get_pending_upload()
        assert len(pending2) == 1
        print(f"PASS test_mark_uploaded_sync: pending={len(pending2)}")


def test_filter_by_date():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        logger = _make_logger(tmp)
        logger.log_event(1, driver_id="d1", timestamp="2026-08-10T10:00:00+00:00")
        logger.log_event(2, driver_id="d1", timestamp="2026-08-11T10:00:00+00:00")
        day = logger.get_events(date="2026-08-10")
        assert len(day) == 1
        assert day[0]["alert_level"] == 1
        print("PASS test_filter_by_date")


if __name__ == "__main__":
    test_log_and_query()
    test_snapshot_saved_when_level_ge_2()
    test_no_snapshot_for_fatigue()
    test_mark_uploaded_sync()
    test_filter_by_date()
    print("\nAll event_logger tests passed.")
