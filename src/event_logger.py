"""
Event Logger — SQLite local, privacy-first (metadata-only by default).

Schema:
  id, timestamp, driver_id, vehicle_id, alert_level,
  ear_avg, perclos, neck_tilt, snapshot_path,
  gps_lat, gps_lng, uploaded

Privacy (project outline):
  - SAVE_FACE_SNAPSHOTS=false by default → never write JPEG
  - sync payloads must strip snapshot_path / never include face bytes
"""
from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np


DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "events.db"
DEFAULT_SNAPSHOT_DIR = Path(__file__).resolve().parent.parent / "data" / "snapshots"
SNAPSHOT_MIN_LEVEL = 2

# Privacy: default OFF. Set env SAVE_FACE_SNAPSHOTS=1|true|yes for DEBUG local only.
def _env_flag(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


class EventLogger:
    """Ghi / đọc event cảnh báo từ SQLite."""

    METADATA_SYNC_FIELDS = (
        "id", "timestamp", "driver_id", "vehicle_id", "alert_level",
        "ear_avg", "perclos", "neck_tilt", "gps_lat", "gps_lng", "uploaded",
    )

    def __init__(
        self,
        db_path: Path | str | None = None,
        snapshot_dir: Path | str | None = None,
        save_face_snapshots: bool | None = None,
    ):
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self.snapshot_dir = Path(snapshot_dir) if snapshot_dir else DEFAULT_SNAPSHOT_DIR
        self.save_face_snapshots = (
            _env_flag("SAVE_FACE_SNAPSHOTS", False)
            if save_face_snapshots is None
            else bool(save_face_snapshots)
        )
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def close(self):
        pass

    def _init_db(self):
        conn = self._connect()
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp     TEXT    NOT NULL,
                    driver_id     TEXT,
                    vehicle_id    TEXT,
                    alert_level   INTEGER NOT NULL,
                    ear_avg       REAL,
                    perclos       REAL,
                    neck_tilt     REAL,
                    snapshot_path TEXT,
                    gps_lat       REAL,
                    gps_lng       REAL,
                    uploaded      INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_events_driver_ts "
                "ON events(driver_id, timestamp)"
            )
            conn.commit()
        finally:
            conn.close()

    def save_snapshot(
        self,
        frame: np.ndarray | None,
        alert_level: int,
        prefix: str = "alert",
    ) -> Optional[str]:
        """Chỉ lưu JPEG khi DEBUG flag bật và level >= 2."""
        if not self.save_face_snapshots:
            return None
        if frame is None or alert_level < SNAPSHOT_MIN_LEVEL:
            return None
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{prefix}_L{alert_level}_{ts}.jpg"
        path = self.snapshot_dir / filename
        ok = cv2.imwrite(str(path), frame)
        return str(path) if ok else None

    def log_event(
        self,
        alert_level: int,
        *,
        driver_id: str | None = None,
        vehicle_id: str | None = None,
        ear_avg: float | None = None,
        perclos: float | None = None,
        neck_tilt: float | None = None,
        snapshot_path: str | None = None,
        frame: np.ndarray | None = None,
        gps_lat: float | None = None,
        gps_lng: float | None = None,
        timestamp: str | None = None,
    ) -> int:
        if snapshot_path is None and frame is not None:
            snapshot_path = self.save_snapshot(frame, alert_level)

        ts = timestamp or datetime.now(timezone.utc).isoformat()

        conn = self._connect()
        try:
            cur = conn.execute(
                """
                INSERT INTO events (
                    timestamp, driver_id, vehicle_id, alert_level,
                    ear_avg, perclos, neck_tilt, snapshot_path,
                    gps_lat, gps_lng, uploaded
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
                """,
                (
                    ts, driver_id, vehicle_id, int(alert_level),
                    ear_avg, perclos, neck_tilt, snapshot_path,
                    gps_lat, gps_lng,
                ),
            )
            conn.commit()
            return int(cur.lastrowid)
        finally:
            conn.close()

    def get_events(
        self,
        *,
        driver_id: str | None = None,
        date: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit), 500))
        offset = max(0, int(offset))
        clauses: list[str] = []
        params: list[Any] = []
        if driver_id:
            clauses.append("driver_id = ?")
            params.append(driver_id)
        if date:
            clauses.append("timestamp LIKE ?")
            params.append(f"{date}%")
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"SELECT * FROM events {where} ORDER BY id DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        conn = self._connect()
        try:
            rows = conn.execute(sql, params).fetchall()
            return [self._row_to_dict(r) for r in rows]
        finally:
            conn.close()

    def get_event(self, event_id: int) -> dict[str, Any] | None:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM events WHERE id = ?", (event_id,)
            ).fetchone()
            return self._row_to_dict(row) if row else None
        finally:
            conn.close()

    def mark_uploaded(self, event_ids: list[int]) -> int:
        if not event_ids:
            return 0
        placeholders = ",".join("?" * len(event_ids))
        conn = self._connect()
        try:
            cur = conn.execute(
                f"UPDATE events SET uploaded = 1 WHERE id IN ({placeholders})",
                [int(i) for i in event_ids],
            )
            conn.commit()
            return cur.rowcount
        finally:
            conn.close()

    def get_pending_upload(self, limit: int = 100) -> list[dict[str, Any]]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM events WHERE uploaded = 0 "
                "ORDER BY id ASC LIMIT ?",
                (max(1, min(int(limit), 500)),),
            ).fetchall()
            return [self._row_to_dict(r) for r in rows]
        finally:
            conn.close()

    def to_sync_payload(self, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Strip snapshot_path — never upload face references/bytes."""
        out = []
        for e in events:
            out.append({k: e.get(k) for k in self.METADATA_SYNC_FIELDS})
        return out

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        d = dict(row)
        d["uploaded"] = bool(d.get("uploaded", 0))
        return d
