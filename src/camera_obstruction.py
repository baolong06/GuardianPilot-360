"""
Camera obstruction / mất mặt detection (PRD A6, R6).

Nếu trước đó đã từng thấy face, rồi n_faces == 0 liên tục > threshold_sec
→ camera_obstructed = True.
"""
from __future__ import annotations


class CameraObstructionDetector:
    def __init__(self, threshold_sec: float = 10.0):
        self.threshold_ms = threshold_sec * 1000.0
        self._ever_seen_face = False
        self._no_face_streak_ms = 0.0
        self._last_ts_ms: float | None = None
        self.camera_obstructed = False

    def update(self, face_found: bool, timestamp_ms: float) -> bool:
        if self._last_ts_ms is None:
            dt = 0.0
        else:
            dt = max(0.0, timestamp_ms - self._last_ts_ms)
        self._last_ts_ms = timestamp_ms

        if face_found:
            self._ever_seen_face = True
            self._no_face_streak_ms = 0.0
            self.camera_obstructed = False
        elif self._ever_seen_face:
            self._no_face_streak_ms += dt
            self.camera_obstructed = self._no_face_streak_ms >= self.threshold_ms
        else:
            # Chưa từng thấy face — không coi là bị che
            self.camera_obstructed = False

        return self.camera_obstructed

    def reset(self):
        self._ever_seen_face = False
        self._no_face_streak_ms = 0.0
        self._last_ts_ms = None
        self.camera_obstructed = False
