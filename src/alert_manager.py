"""
Alert Manager đa cấp — map DriverState → alert_level 0–4 + message + channels.

Channels (outline): sound / vibration / break_suggested
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from .scoring import DriverState


ALERT_MESSAGES: dict[int, str] = {
    0: "Bình thường — tài xế tỉnh táo",
    1: "Cảnh báo cấp 1 — Dấu hiệu mệt mỏi, hãy nghỉ ngơi sớm",
    2: "Cảnh báo cấp 2 — Buồn ngủ, cần tập trung hoặc dừng xe",
    3: "Cảnh báo cấp 3 — Ngủ gật! Đánh thức tài xế ngay",
    4: "Cảnh báo cấp 4 — NGUY HIỂM! Không phục hồi sau cảnh báo",
}


def channels_for_level(level: int) -> dict:
    """Map alert level → actuation channels for FE / HMI."""
    return {
        "sound": level >= 2,
        "vibration": level >= 3,
        "break_suggested": level >= 1,
    }


@dataclass
class AlertStatus:
    alert_level: int
    alert_message: str
    drowsiness_state: str
    changed: bool
    channels: dict = field(default_factory=lambda: channels_for_level(0))


OnLevelChange = Callable[[AlertStatus], None]


class AlertManager:
    def __init__(self, on_level_change: Optional[OnLevelChange] = None):
        self.alert_level: int = 0
        self._on_level_change = on_level_change

    def update(self, driver_state: DriverState | str | int) -> AlertStatus:
        if isinstance(driver_state, DriverState):
            new_level = int(driver_state)
            state_name = driver_state.name
        elif isinstance(driver_state, str):
            state = DriverState[driver_state.upper()]
            new_level = int(state)
            state_name = state.name
        else:
            new_level = int(driver_state)
            state_name = DriverState(new_level).name

        changed = new_level != self.alert_level
        self.alert_level = new_level

        status = AlertStatus(
            alert_level=new_level,
            alert_message=ALERT_MESSAGES.get(new_level, ALERT_MESSAGES[0]),
            drowsiness_state=state_name,
            changed=changed,
            channels=channels_for_level(new_level),
        )

        if changed and self._on_level_change is not None:
            self._on_level_change(status)

        return status

    def reset(self):
        prev = self.alert_level
        self.alert_level = 0
        if prev != 0 and self._on_level_change is not None:
            self._on_level_change(
                AlertStatus(
                    alert_level=0,
                    alert_message=ALERT_MESSAGES[0],
                    drowsiness_state=DriverState.NORMAL.name,
                    changed=True,
                    channels=channels_for_level(0),
                )
            )

    def __repr__(self) -> str:
        return f"AlertManager(level={self.alert_level})"
