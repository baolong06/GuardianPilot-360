"""
Vehicle / trip context — speed + continuous driving time → risk multiplier.

Used to bias drowsiness scoring (high speed + fatigue escalates risk).
"""
from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class VehicleState:
    speed_kmh: float = 0.0
    driving_time_sec: float = 0.0
    risk_multiplier: float = 1.0
    trip_active: bool = False


class DrivingContext:
    """
    Mock CAN-friendly context.
    - set_speed() from /api/vehicle or can_sim
    - update() advances driving_time while speed > idle_kmh
    """

    def __init__(
        self,
        idle_kmh: float = 3.0,
        high_speed_kmh: float = 80.0,
        long_drive_sec: float = 2 * 3600,
    ):
        self.idle_kmh = idle_kmh
        self.high_speed_kmh = high_speed_kmh
        self.long_drive_sec = long_drive_sec
        self.speed_kmh = 0.0
        self.driving_time_sec = 0.0
        self.trip_started_at: float | None = None
        self._last_tick: float | None = None

    def set_speed(self, speed_kmh: float):
        self.speed_kmh = max(0.0, float(speed_kmh))
        if self.speed_kmh > self.idle_kmh and self.trip_started_at is None:
            self.trip_started_at = time.time()

    def update(self, now: float | None = None) -> VehicleState:
        now = now if now is not None else time.time()
        if self._last_tick is not None and self.speed_kmh > self.idle_kmh:
            self.driving_time_sec += max(0.0, now - self._last_tick)
        self._last_tick = now
        return self.snapshot()

    def risk_multiplier(self) -> float:
        """1.0 baseline; up to ~1.35 with high speed + long drive."""
        m = 1.0
        if self.speed_kmh >= self.high_speed_kmh:
            m += 0.15
        elif self.speed_kmh >= 50:
            m += 0.08
        if self.driving_time_sec >= self.long_drive_sec:
            m += 0.20
        elif self.driving_time_sec >= 3600:
            m += 0.10
        return round(min(m, 1.5), 3)

    def snapshot(self) -> VehicleState:
        return VehicleState(
            speed_kmh=round(self.speed_kmh, 1),
            driving_time_sec=round(self.driving_time_sec, 1),
            risk_multiplier=self.risk_multiplier(),
            trip_active=self.speed_kmh > self.idle_kmh or self.driving_time_sec > 0,
        )

    def reset(self):
        self.speed_kmh = 0.0
        self.driving_time_sec = 0.0
        self.trip_started_at = None
        self._last_tick = None

    def apply_to_score(self, score: float) -> float:
        """Bias drowsiness score upward under risky driving context."""
        return max(0.0, min(1.0, score * self.risk_multiplier()))
