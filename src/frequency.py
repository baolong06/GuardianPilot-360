"""
Rolling-window event frequency counters (YHP-02 style).

- HeadNodCounter: đếm lần neck_alarm rising-edge trong cửa sổ (mặc định 60s)
- YawnCounter: đếm lần yawn_alarm rising-edge trong cửa sổ
"""
from __future__ import annotations

from collections import deque


class EventFrequencyCounter:
    """Đếm rising-edge events trong rolling window."""

    def __init__(self, window_sec: float = 60.0):
        self.window_ms = window_sec * 1000.0
        self._events: deque[float] = deque()  # timestamps of triggers
        self._prev_active = False

    def update(self, timestamp_ms: float, active: bool) -> int:
        """
        Gọi mỗi frame. Rising-edge (False→True) được ghi nhận là 1 event.
        Returns: số event trong window hiện tại.
        """
        if active and not self._prev_active:
            self._events.append(timestamp_ms)
        self._prev_active = active

        cutoff = timestamp_ms - self.window_ms
        while self._events and self._events[0] < cutoff:
            self._events.popleft()

        return len(self._events)

    def count(self) -> int:
        return len(self._events)

    def reset(self):
        self._events.clear()
        self._prev_active = False
