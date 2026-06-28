"""
guardian_pilot/actuation.py
------------------------------
Actuation Layer — Lớp hành động cuối cùng

Nhận DriverState từ Orchestrator → kích hoạt cảnh báo phù hợp.
Giai đoạn MVP: in console + ghi audit log file.
Giai đoạn Edge: hook CAN bus, loa, đèn LED, rung ghế.

Thiết kế theo kiến trúc mục 3 & 7.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import List

from .core.knowledge_graph import KnowledgeGraph
from .core.schema import ActuationEvent, AlertLevel, DriverState

logger = logging.getLogger("guardian_pilot.actuation")

# Màu ANSI cho console
_COLOR = {
    AlertLevel.NORMAL:         "\033[92m",   # xanh lá
    AlertLevel.MILD_WARNING:   "\033[93m",   # vàng
    AlertLevel.SEVERE_WARNING: "\033[91m",   # đỏ
    AlertLevel.EMERGENCY:      "\033[95m",   # tím/magenta
}
_RESET = "\033[0m"

# Icon theo mức
_ICON = {
    AlertLevel.NORMAL:         "✅",
    AlertLevel.MILD_WARNING:   "⚠️ ",
    AlertLevel.SEVERE_WARNING: "🚨",
    AlertLevel.EMERGENCY:      "🆘",
}


def _get_actions(level: AlertLevel) -> List[str]:
    """Ánh xạ alert level → danh sách hành động thực tế."""
    if level == AlertLevel.NORMAL:
        return ["system_idle"]
    elif level == AlertLevel.MILD_WARNING:
        return ["beep_soft_1x", "dashboard_amber"]
    elif level == AlertLevel.SEVERE_WARNING:
        return ["beep_loud_3x", "seat_vibrate", "dashboard_red"]
    elif level == AlertLevel.EMERGENCY:
        return ["siren_continuous", "seat_vibrate_strong",
                "dashboard_red_flash", "send_telemetry_emergency"]
    return []


class ActuationLayer:
    """
    Lớp hành động cuối — nhận callback từ Orchestrator.
    Ghi ActuationEvent vào KG và audit log file.
    """

    def __init__(
        self,
        kg: KnowledgeGraph,
        audit_log_path: str = "guardian_pilot_audit.log",
    ) -> None:
        self.kg             = kg
        self.audit_log_path = audit_log_path
        self._last_level    = AlertLevel.NORMAL

    def on_state_change(self, state: DriverState) -> None:
        """
        Được gọi bởi Orchestrator khi alert level thay đổi.
        Không gọi nếu level giống trước (Orchestrator đã lọc).
        """
        level   = state.current_alert_level
        actions = _get_actions(level)

        # In console (MVP)
        color = _COLOR.get(level, "")
        icon  = _ICON.get(level, "")
        print(
            f"\n{color}{'='*60}\n"
            f"{icon}  ALERT: {level.value}\n"
            f"  Reason:  {state.alert_reason}\n"
            f"  Conf:    {state.confidence:.2f}\n"
            f"  Active:  {[a.value for a in state.active_agents]}\n"
            f"  Actions: {actions}\n"
            f"{'='*60}{_RESET}"
        )

        # Ghi ActuationEvent vào KG
        event = ActuationEvent(
            alert_level = level,
            actions     = actions,
            reason      = state.alert_reason,
        )
        self.kg.log_actuation(event)

        # Ghi audit trail
        self._write_audit(state, actions)

        # Hook hành động thực tế (override trong production)
        self._execute_actions(level, actions)

        self._last_level = level

    def _execute_actions(self, level: AlertLevel, actions: List[str]) -> None:
        """
        MVP: chỉ log.
        Production: gọi CAN bus, audio driver, LED controller.
        """
        logger.info("ACTUATION | %s | actions=%s", level.value, actions)

    def _write_audit(self, state: DriverState, actions: List[str]) -> None:
        """Ghi audit trail dạng JSONL — 1 dòng = 1 event."""
        record = {
            "ts":          state.timestamp.isoformat(),
            "alert":       state.current_alert_level.value,
            "reason":      state.alert_reason,
            "confidence":  round(state.confidence, 3),
            "agents":      [a.value for a in state.active_agents],
            "actions":     actions,
        }
        try:
            with open(self.audit_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as e:
            logger.error("Không ghi được audit log: %s", e)
