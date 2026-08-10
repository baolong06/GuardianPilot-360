"""
Trip fatigue memory — aggregate PERCLOS / state / alert peaks for a session.
"""
from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field


@dataclass
class TripMemory:
    started_at: float = field(default_factory=time.time)
    samples: int = 0
    perclos_sum: float = 0.0
    perclos_peak: float = 0.0
    state_counts: Counter = field(default_factory=Counter)
    alert_peak: int = 0
    looking_away_events: int = 0
    phone_events: int = 0
    last_state: str = "NORMAL"

    def update(
        self,
        *,
        perclos: float = 0.0,
        drowsiness_state: str = "NORMAL",
        alert_level: int = 0,
        looking_away: bool = False,
        phone_suspected: bool = False,
    ):
        self.samples += 1
        self.perclos_sum += float(perclos)
        self.perclos_peak = max(self.perclos_peak, float(perclos))
        self.state_counts[drowsiness_state] += 1
        self.last_state = drowsiness_state
        self.alert_peak = max(self.alert_peak, int(alert_level))
        if looking_away:
            self.looking_away_events += 1
        if phone_suspected:
            self.phone_events += 1

    def summary(self, driving_time_sec: float | None = None) -> dict:
        elapsed = time.time() - self.started_at
        avg = self.perclos_sum / self.samples if self.samples else 0.0
        return {
            "trip_duration_sec": round(driving_time_sec if driving_time_sec is not None else elapsed, 1),
            "samples": self.samples,
            "perclos_avg": round(avg, 4),
            "perclos_peak": round(self.perclos_peak, 4),
            "state_distribution": dict(self.state_counts),
            "alert_peak": self.alert_peak,
            "last_state": self.last_state,
            "looking_away_frames": self.looking_away_events,
            "phone_suspected_frames": self.phone_events,
        }

    def reset(self):
        self.__init__()
