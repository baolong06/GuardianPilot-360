"""
Phone distraction heuristic — wrist (pose) near face center for sustained time.

P0: Holistic pose wrists (idx 15/16) vs face bbox center.
P1: swap implementation via PhoneDetector protocol / YOLO.

Interface:
  update(timestamp_ms, face_center_xy, wrist_points) -> dict
"""
from __future__ import annotations

import math
from typing import Iterable, Optional, Protocol, Sequence

from .frequency import EventFrequencyCounter


class PhoneDetector(Protocol):
    def update(self, timestamp_ms: float, **kwargs) -> dict: ...
    def reset(self) -> None: ...


class HandNearFacePhoneHeuristic:
    """
    phone_suspected when any wrist is within near_frac of face diagonal
    for >= min_duration_sec.
    """

    def __init__(
        self,
        near_frac: float = 0.45,
        min_duration_sec: float = 1.2,
        window_sec: float = 60.0,
    ):
        self.near_frac = near_frac
        self.min_duration_ms = min_duration_sec * 1000.0
        self.streak_ms = 0.0
        self.phone_suspected = False
        self._counter = EventFrequencyCounter(window_sec=window_sec)
        self._last_ts: float | None = None

    def update(
        self,
        timestamp_ms: float,
        *,
        face_center: Optional[tuple[float, float]] = None,
        face_size: Optional[float] = None,
        wrists: Optional[Sequence[tuple[float, float]]] = None,
        **_kwargs,
    ) -> dict:
        if self._last_ts is None:
            dt = 0.0
        else:
            dt = max(0.0, timestamp_ms - self._last_ts)
        self._last_ts = timestamp_ms

        near = False
        if face_center and face_size and face_size > 1e-6 and wrists:
            cx, cy = face_center
            thresh = self.near_frac * face_size
            for wx, wy in wrists:
                if math.hypot(wx - cx, wy - cy) <= thresh:
                    near = True
                    break

        if near:
            self.streak_ms += dt
        else:
            self.streak_ms = 0.0

        active = self.streak_ms >= self.min_duration_ms
        self.phone_suspected = active
        count = self._counter.update(timestamp_ms, active)
        return {
            "phone_suspected": active,
            "phone_streak_ms": round(self.streak_ms, 1),
            "phone_count_window": count,
        }

    def reset(self):
        self.streak_ms = 0.0
        self.phone_suspected = False
        self._last_ts = None
        self._counter.reset()


# Default implementation (YOLO can replace later)
PhoneDistractionDetector = HandNearFacePhoneHeuristic


def wrists_from_pose(pose_landmarks, img_w: float, img_h: float) -> list[tuple[float, float]]:
    """Extract left/right wrist (MediaPipe pose 15, 16) in pixel coords."""
    if not pose_landmarks:
        return []
    pts = pose_landmarks[0] if isinstance(pose_landmarks[0], (list, tuple)) else pose_landmarks
    out = []
    for idx in (15, 16):
        try:
            lm = pts[idx]
            vis = getattr(lm, "visibility", 1.0)
            if vis is not None and vis < 0.3:
                continue
            out.append((float(lm.x) * img_w, float(lm.y) * img_h))
        except (IndexError, TypeError, AttributeError):
            continue
    return out


def face_geometry_from_landmarks(face_landmarks, img_w: float, img_h: float):
    """Return (center_xy, face_size_px) or (None, None)."""
    if not face_landmarks:
        return None, None
    pts = face_landmarks[0] if face_landmarks and hasattr(face_landmarks[0], "__iter__") and not hasattr(face_landmarks[0], "x") else face_landmarks
    # handle list-of-lists from TransformedResult
    if pts and isinstance(pts, (list, tuple)) and pts and not hasattr(pts[0], "x"):
        pts = pts[0]
    try:
        xs = [float(lm.x) * img_w for lm in pts]
        ys = [float(lm.y) * img_h for lm in pts]
    except (TypeError, AttributeError):
        return None, None
    if not xs:
        return None, None
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    size = max(max(xs) - min(xs), max(ys) - min(ys))
    return (cx, cy), size
