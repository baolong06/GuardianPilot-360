"""
HITL threshold store — mutable runtime knobs for fusion/scoring/looking-away.
"""
from __future__ import annotations

import copy
import time
from typing import Any


# Defaults mirror fusion/scoring/looking_away constants.
#
# H5: `eye_closed_thresh` trước đây ghi 0.18 nhưng KHÔNG bao giờ được áp dụng
# (app.py quên đẩy xuống FusionState), nên giá trị chạy thật vẫn là 0.16 của
# fusion.EYE_CLOSED_THRESH. Nay knob đã có tác dụng thật (xem
# FusionState.apply_thresholds), nên default phải bằng đúng 0.16 — nếu để 0.18
# thì việc "sửa knob" sẽ vô tình làm hệ thống nhạy hơn hẳn ngay lần khởi động
# đầu tiên. Đây là thay đổi để KHỚP hành vi hiện hành, không phải tinh chỉnh.
_DEFAULTS: dict[str, Any] = {
    "eye_closed_thresh": 0.16,
    "eye_closed_on_sec": 0.8,
    "eye_closed_hard_sec": 1.2,
    "yaw_thresh_deg": 25.0,
    "looking_away_min_sec": 1.0,
    "phone_near_frac": 0.45,
    "phone_min_sec": 1.2,
    "fatigue_on": 0.40,
    "drowsy_on": 0.55,
    "microsleep_on": 0.75,
    "high_speed_kmh": 80.0,
    "long_drive_sec": 7200.0,
}

_current: dict[str, Any] = copy.deepcopy(_DEFAULTS)
_audit: list[dict[str, Any]] = []


def get_thresholds() -> dict[str, Any]:
    return copy.deepcopy(_current)


def get_defaults() -> dict[str, Any]:
    return copy.deepcopy(_DEFAULTS)


def update_thresholds(patch: dict[str, Any], actor: str = "engineer") -> dict[str, Any]:
    global _current
    applied = {}
    for k, v in patch.items():
        if k in _DEFAULTS:
            _current[k] = type(_DEFAULTS[k])(v)
            applied[k] = _current[k]
    if applied:
        _audit.append({
            "ts": time.time(),
            "actor": actor,
            "changes": applied,
        })
    return get_thresholds()


def reset_thresholds() -> dict[str, Any]:
    global _current
    _current = copy.deepcopy(_DEFAULTS)
    _audit.append({"ts": time.time(), "actor": "system", "changes": {"_reset": True}})
    return get_thresholds()


def audit_log(limit: int = 20) -> list[dict[str, Any]]:
    return list(_audit[-limit:])
