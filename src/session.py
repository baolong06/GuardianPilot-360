"""
Per-driver session state (H2).

Trước đây toàn bộ state (FusionState, AlertManager, TripMemory…) là biến global
duy nhất trong app.py — hai tab / hai xe gọi API cùng lúc sẽ trộn state của nhau
và `/api/reset` của người này xoá state của người kia.

Module này gói toàn bộ state đó vào `DriverSession` và cấp phát theo `session_id`.
Request không gửi session_id → dùng session "default" ⇒ tương thích ngược 100%
với frontend cũ, test cũ và các script trong tools/.

Lưu ý về khoá: SessionStore chỉ bảo vệ *bảng session*. Việc serialize inference
(MediaPipe landmarker là singleton, không thread-safe) vẫn do app.py giữ bằng một
khoá toàn cục riêng.
"""
from __future__ import annotations

import threading
import time
from typing import Any, Iterable

from .alert_manager import AlertManager
from .camera_obstruction import CameraObstructionDetector
from .context import DrivingContext
from .fusion import FusionState
from .phone_distraction import PhoneDistractionDetector
from .trip_memory import TripMemory

DEFAULT_SESSION_ID = "default"
MAX_SESSION_ID_LEN = 64


def normalize_session_id(session_id: Any) -> str:
    """Chuẩn hoá id do client gửi lên (tránh key rác / quá dài)."""
    if session_id is None:
        return DEFAULT_SESSION_ID
    sid = str(session_id).strip()
    if not sid:
        return DEFAULT_SESSION_ID
    # Chỉ giữ ký tự an toàn — id đến từ client, dùng làm key dict + log
    sid = "".join(ch for ch in sid if ch.isalnum() or ch in "-_.")
    return (sid or DEFAULT_SESSION_ID)[:MAX_SESSION_ID_LEN]


class DriverSession:
    """Toàn bộ state runtime của MỘT tài xế / MỘT tab trình duyệt."""

    def __init__(self, session_id: str = DEFAULT_SESSION_ID):
        self.session_id = session_id
        self.fusion = FusionState()
        self.alert_manager = AlertManager()
        self.camera_obstruction = CameraObstructionDetector(threshold_sec=10.0)
        self.driving_context = DrivingContext()
        self.trip_memory = TripMemory()
        self.phone_detector = PhoneDistractionDetector()

        # Metadata phiên (có thể set qua /api/init)
        self.driver_id: str | None = "driver_demo"
        self.vehicle_id: str | None = "vehicle_demo"
        self.gps_lat: float | None = None
        self.gps_lng: float | None = None

        # M2: FPS đo bằng đồng hồ thực, KHÔNG dùng timestamp của media
        self.inference_fps: float = 0.0
        self._last_infer_monotonic: float | None = None

        self.created_at = time.monotonic()
        self.last_seen = time.monotonic()

    # ── Lifecycle ────────────────────────────────────────────────────────
    def touch(self) -> None:
        self.last_seen = time.monotonic()

    def reset(self) -> None:
        """Xoá state nhận diện, GIỮ metadata phiên (driver_id/vehicle_id/GPS)."""
        self.fusion.reset()
        self.alert_manager.reset()
        self.camera_obstruction.reset()
        self.driving_context.reset()
        self.trip_memory.reset()
        self.phone_detector.reset()
        self.inference_fps = 0.0
        self._last_infer_monotonic = None
        self.touch()

    # ── HITL thresholds (H4) ─────────────────────────────────────────────
    def apply_thresholds(self, thresholds: dict) -> None:
        """Đẩy knob HITL xuống mọi detector của session này."""
        self.fusion.apply_thresholds(thresholds)
        self.phone_detector.near_frac = float(thresholds["phone_near_frac"])
        self.phone_detector.min_duration_ms = float(thresholds["phone_min_sec"]) * 1000.0
        self.driving_context.high_speed_kmh = float(thresholds["high_speed_kmh"])
        self.driving_context.long_drive_sec = float(thresholds["long_drive_sec"])

    # ── Metrics ──────────────────────────────────────────────────────────
    def note_inference(self) -> float:
        """
        M2: cập nhật inference FPS bằng `time.monotonic()`.

        Trước đây FPS tính từ `ts_ms` của frame — với video upload thì đó là
        media timeline, nên số liệu báo ra là FPS của video chứ không phải
        tốc độ xử lý thật của server.
        """
        now = time.monotonic()
        prev = self._last_infer_monotonic
        self._last_infer_monotonic = now
        if prev is None:
            return self.inference_fps
        dt = max(1e-3, now - prev)
        instant = 1.0 / dt
        self.inference_fps = (
            0.8 * self.inference_fps + 0.2 * instant if self.inference_fps else instant
        )
        return self.inference_fps

    def __repr__(self) -> str:
        return f"DriverSession(id={self.session_id!r}, driver={self.driver_id!r})"


class SessionStore:
    """Bảng session có TTL + giới hạn số lượng."""

    def __init__(self, ttl_sec: float = 1800.0, max_sessions: int = 32):
        self.ttl_sec = ttl_sec
        self.max_sessions = max_sessions
        self._sessions: dict[str, DriverSession] = {}
        self._lock = threading.RLock()

    # ── Truy cập ─────────────────────────────────────────────────────────
    def get(self, session_id: Any = None) -> DriverSession:
        """Lấy (hoặc tạo) session. Không bao giờ trả None."""
        sid = normalize_session_id(session_id)
        with self._lock:
            self._evict_locked()
            session = self._sessions.get(sid)
            if session is None:
                self._make_room_locked()
                session = DriverSession(sid)
                self._sessions[sid] = session
            session.touch()
            return session

    def reset(self, session_id: Any = None) -> DriverSession:
        session = self.get(session_id)
        session.reset()
        return session

    def apply_thresholds_all(self, thresholds: dict) -> None:
        """PUT /api/thresholds áp cho mọi session đang sống."""
        with self._lock:
            sessions = list(self._sessions.values())
        for session in sessions:
            session.apply_thresholds(thresholds)

    def all_sessions(self) -> Iterable[DriverSession]:
        with self._lock:
            return list(self._sessions.values())

    def stats(self) -> dict:
        with self._lock:
            now = time.monotonic()
            return {
                "active_sessions": len(self._sessions),
                "session_ids": sorted(self._sessions.keys()),
                "ttl_sec": self.ttl_sec,
                "max_sessions": self.max_sessions,
                "idle_sec": {
                    sid: round(now - s.last_seen, 1)
                    for sid, s in self._sessions.items()
                },
            }

    def clear(self) -> None:
        with self._lock:
            self._sessions.clear()

    # ── Nội bộ ───────────────────────────────────────────────────────────
    def _evict_locked(self) -> None:
        now = time.monotonic()
        stale = [
            sid for sid, s in self._sessions.items()
            if now - s.last_seen > self.ttl_sec
        ]
        for sid in stale:
            self._sessions.pop(sid, None)

    def _make_room_locked(self) -> None:
        """Đầy bảng → bỏ session ít hoạt động nhất (ưu tiên giữ 'default')."""
        if len(self._sessions) < self.max_sessions:
            return
        candidates = [
            (s.last_seen, sid) for sid, s in self._sessions.items()
            if sid != DEFAULT_SESSION_ID
        ] or [(s.last_seen, sid) for sid, s in self._sessions.items()]
        if candidates:
            candidates.sort()
            self._sessions.pop(candidates[0][1], None)
