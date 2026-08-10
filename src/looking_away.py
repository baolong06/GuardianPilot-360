"""
Looking-away detector — |yaw| sustained above threshold.

Output: looking_away (bool), looking_away_streak_ms, looking_away_count_window
"""
from __future__ import annotations

import math

from .frequency import EventFrequencyCounter


class LookingAwayDetector:
    def __init__(
        self,
        yaw_thresh_deg: float = 25.0,
        min_duration_sec: float = 1.0,
        window_sec: float = 60.0,
    ):
        self.yaw_thresh_deg = yaw_thresh_deg
        self.min_duration_ms = min_duration_sec * 1000.0
        self.streak_ms = 0.0
        self.looking_away = False
        self._counter = EventFrequencyCounter(window_sec=window_sec)
        self._last_ts: float | None = None

    def update(self, timestamp_ms: float, yaw: float | None) -> dict:
        if self._last_ts is None:
            dt = 0.0
        else:
            dt = max(0.0, timestamp_ms - self._last_ts)
        self._last_ts = timestamp_ms

        diverted = (
            yaw is not None
            and not (isinstance(yaw, float) and math.isnan(yaw))
            and abs(float(yaw)) >= self.yaw_thresh_deg
        )
        if diverted:
            self.streak_ms += dt
        else:
            self.streak_ms = 0.0

        active = self.streak_ms >= self.min_duration_ms
        self.looking_away = active
        count = self._counter.update(timestamp_ms, active)
        return {
            "looking_away": active,
            "looking_away_streak_ms": round(self.streak_ms, 1),
            "looking_away_count_window": count,
        }

    def reset(self):
        self.streak_ms = 0.0
        self.looking_away = False
        self._last_ts = None
        self._counter.reset()
