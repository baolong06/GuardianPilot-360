"""
guardian_pilot/agents/orchestrator.py
----------------------------------------
Orchestrator Agent — Agent Chỉ huy

KHÔNG tự suy luận từ ảnh/sensor.
Chỉ đọc KG → áp 7 luật ưu tiên → ghi DriverState → trigger Actuation.

7 luật theo kiến trúc mục 5.6:
  RULE 1: M2 PATHOLOGICAL_PROXY + conf > 0.6 → EMERGENCY
  RULE 2: M1+M4 đều DROWSY + conf > 0.5 → SEVERE_WARNING
  RULE 3: Đúng 1 trong M1/M4 DROWSY + conf > 0.6 → MILD_WARNING + window
  RULE 4: M1 vs M4 mâu thuẫn → log ConflictEvent + MILD_WARNING
  RULE 5: M3 DISTRACTED + conf > 0.6 → max(current, MILD_WARNING)
  RULE 6: Bất kỳ agent offline → log vận hành + giảm max_achievable_confidence
  RULE 7: Mặc định → NORMAL
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple

from ..core.knowledge_graph import KnowledgeGraph
from ..core.schema import (
    AgentID, AlertLevel, ConflictEvent,
    DriverState, InputQuality, NormalizedLabel,
)

# Ngưỡng confidence trong từng Rule
R1_CONF_EMERGENCY  = 0.60
R2_CONF_SENSOR_FUSION = 0.50
R3_CONF_SINGLE     = 0.60
R5_CONF_DISTRACTED = 0.60

# Confirmation window cho RULE 3 (ms)
CONFIRMATION_WINDOW_MS = 2000.0

# Thứ tự severity (dùng để max() alert level)
_LEVEL_ORDER = {
    AlertLevel.NORMAL:         0,
    AlertLevel.MILD_WARNING:   1,
    AlertLevel.SEVERE_WARNING: 2,
    AlertLevel.EMERGENCY:      3,
}


def _max_level(a: AlertLevel, b: AlertLevel) -> AlertLevel:
    return a if _LEVEL_ORDER[a] >= _LEVEL_ORDER[b] else b


class OrchestratorAgent:
    """
    Agent Chỉ huy — đọc KG → áp luật → ghi DriverState.
    Không kế thừa PerceptionAgent (nó không inference model trực tiếp).
    """

    def __init__(
        self,
        kg: KnowledgeGraph,
        actuation_callback=None,
    ) -> None:
        self.kg = kg
        self._actuation_callback = actuation_callback
        self._lock = threading.Lock()

        # Tracking confirmation window (RULE 3)
        self._mild_warning_start: Optional[float] = None

    # ─────────────────────────────────────────
    #  Điểm vào chính — gọi mỗi tick
    # ─────────────────────────────────────────

    def tick(self) -> DriverState:
        """
        Một chu kỳ Orchestrator:
        1. Đọc PerceptionResult mới nhất từ KG
        2. Áp 7 luật → ra alert_level + reason
        3. Tính confidence tổng hợp
        4. Ghi DriverState mới vào KG
        5. Trigger Actuation nếu level thay đổi
        """
        with self._lock:
            perceptions = self._read_perceptions()
            old_state   = self.kg.get_driver_state()

            alert_level, reason, confidence = self._apply_rules(perceptions)

            active_agents = [
                aid for aid in perceptions
                if perceptions[aid] is not None
            ]

            new_state = DriverState(
                current_alert_level = alert_level,
                alert_reason        = reason,
                confidence          = confidence,
                active_agents       = active_agents,
            )
            self.kg.write_driver_state(new_state)

            # Trigger actuation chỉ khi level thay đổi
            if alert_level != old_state.current_alert_level:
                self._trigger_actuation(new_state)

            return new_state

    # ─────────────────────────────────────────
    #  7 Luật ưu tiên
    # ─────────────────────────────────────────

    def _apply_rules(self, perceptions: dict) -> Tuple[AlertLevel, str, float]:
        """
        Áp 7 luật theo thứ tự ưu tiên.
        Trả về (alert_level, reason, confidence).
        """
        m1 = perceptions.get(AgentID.M1_DROWSINESS)
        m2 = perceptions.get(AgentID.M2_MICROSLEEP)
        m3 = perceptions.get(AgentID.M3_DISTRACTED)
        m4 = perceptions.get(AgentID.M4_LANDMARK_GAZE)

        alive = self.kg.alive_agents()
        n_alive = len([a for a in alive if a != AgentID.ORCHESTRATOR])

        # ── RULE 6: Agent offline ─────────────────────────────────
        # (không dừng hệ thống, nhưng giảm max confidence)
        max_conf_multiplier = 1.0
        offline_agents = [
            aid for aid in [AgentID.M1_DROWSINESS, AgentID.M2_MICROSLEEP,
                            AgentID.M3_DISTRACTED, AgentID.M4_LANDMARK_GAZE]
            if not self.kg.get_health(aid).is_alive
        ]
        if offline_agents:
            # Mỗi agent offline → giảm 15% max confidence
            max_conf_multiplier = max(0.4, 1.0 - len(offline_agents) * 0.15)

        # ── RULE 1: EMERGENCY (M2 pathological) ───────────────────
        if (m2 is not None
                and m2.normalized_label == NormalizedLabel.PATHOLOGICAL_PROXY
                and m2.confidence > R1_CONF_EMERGENCY):
            return (
                AlertLevel.EMERGENCY,
                f"M2: Phát hiện trạng thái sinh lý nguy hiểm (conf={m2.confidence:.2f}). "
                "Ưu tiên khẩn cấp tuyệt đối.",
                m2.confidence * max_conf_multiplier,
            )

        # ── RULE 2: SEVERE_WARNING (M1+M4 đồng thuận DROWSY) ──────
        if (m1 is not None and m4 is not None
                and m1.normalized_label == NormalizedLabel.DROWSY
                and m4.normalized_label == NormalizedLabel.DROWSY
                and m1.confidence > R2_CONF_SENSOR_FUSION
                and m4.confidence > R2_CONF_SENSOR_FUSION):
            fused_conf = (m1.confidence + m4.confidence) / 2 * max_conf_multiplier
            return (
                AlertLevel.SEVERE_WARNING,
                f"Sensor fusion: M1(ảnh thô, conf={m1.confidence:.2f}) + "
                f"M4(landmark, conf={m4.confidence:.2f}) đều phát hiện BUỒN NGỦ.",
                fused_conf,
            )

        # ── RULE 4: Xung đột M1 vs M4 ─────────────────────────────
        if (m1 is not None and m4 is not None
                and m1.confidence > R2_CONF_SENSOR_FUSION
                and m4.confidence > R2_CONF_SENSOR_FUSION
                and m1.normalized_label != m4.normalized_label
                and {m1.normalized_label, m4.normalized_label}
                    == {NormalizedLabel.DROWSY, NormalizedLabel.ALERT}):

            conflict = ConflictEvent(
                agents_involved     = [AgentID.M1_DROWSINESS, AgentID.M4_LANDMARK_GAZE],
                conflicting_labels  = {
                    "M1": m1.normalized_label.value,
                    "M4": m4.normalized_label.value,
                },
                resolution_strategy = "trust_landmark_over_raw_image_on_lighting_uncertainty",
                final_label         = "MILD_WARNING",
            )
            self.kg.log_conflict(conflict)

            # Tin M4 hơn (landmark ít bị ảnh hưởng ánh sáng) nhưng không bỏ qua M1
            conf = max(m1.confidence, m4.confidence) * 0.6 * max_conf_multiplier
            return (
                AlertLevel.MILD_WARNING,
                f"XUNG ĐỘT: M1={m1.normalized_label.value}(conf={m1.confidence:.2f}) "
                f"vs M4={m4.normalized_label.value}(conf={m4.confidence:.2f}). "
                "Chọn MILD_WARNING (an toàn hơn). M4 landmark được ưu tiên.",
                conf,
            )

        # ── RULE 3: MILD_WARNING (đúng 1 agent DROWSY) ────────────
        drowsy_single = []
        if m1 is not None and m1.normalized_label == NormalizedLabel.DROWSY and m1.confidence > R3_CONF_SINGLE:
            drowsy_single.append(("M1", m1.confidence))
        if m4 is not None and m4.normalized_label == NormalizedLabel.DROWSY and m4.confidence > R3_CONF_SINGLE:
            drowsy_single.append(("M4", m4.confidence))

        if len(drowsy_single) == 1:
            agent_name, conf = drowsy_single[0]
            self._start_confirmation_window()
            return (
                AlertLevel.MILD_WARNING,
                f"{agent_name}: Cảnh báo DROWSY đơn lẻ (conf={conf:.2f}). "
                "Chờ xác nhận từ agent khác trong 2 giây.",
                conf * 0.7 * max_conf_multiplier,
            )

        # ── RULE 5: DISTRACTED (cộng dồn với alert hiện tại) ──────
        current_level = AlertLevel.NORMAL
        current_reason = "Tất cả chỉ số bình thường."
        current_conf   = 1.0 * max_conf_multiplier

        if (m3 is not None
                and m3.normalized_label == NormalizedLabel.DISTRACTED
                and m3.confidence > R5_CONF_DISTRACTED):
            current_level  = _max_level(current_level, AlertLevel.MILD_WARNING)
            current_reason = (
                f"M3: Phát hiện MẤT TẬP TRUNG (conf={m3.confidence:.2f}). "
                "Distracted và Drowsy là 2 trục độc lập — cộng dồn."
            )
            current_conf   = m3.confidence * max_conf_multiplier

        elif (m3 is not None
              and m3.normalized_label == NormalizedLabel.MILD_CONCERN
              and m3.confidence > R5_CONF_DISTRACTED):
            current_reason = (
                f"M3: Hành vi phân tâm nhẹ (conf={m3.confidence:.2f})."
            )

        # ── RULE 7: Default ────────────────────────────────────────
        # current_level đã là NORMAL hoặc được nâng lên bởi RULE 5
        return (current_level, current_reason, current_conf)

    # ─────────────────────────────────────────
    #  Helpers
    # ─────────────────────────────────────────

    def _read_perceptions(self) -> Dict:
        """Đọc PerceptionResult mới nhất từ KG cho tất cả Perception Agent."""
        return {
            AgentID.M1_DROWSINESS:    self.kg.get_latest_perception(AgentID.M1_DROWSINESS),
            AgentID.M2_MICROSLEEP:    self.kg.get_latest_perception(AgentID.M2_MICROSLEEP),
            AgentID.M3_DISTRACTED:    self.kg.get_latest_perception(AgentID.M3_DISTRACTED),
            AgentID.M4_LANDMARK_GAZE: self.kg.get_latest_perception(AgentID.M4_LANDMARK_GAZE),
        }

    def _start_confirmation_window(self) -> None:
        if self._mild_warning_start is None:
            self._mild_warning_start = time.monotonic()

    def _trigger_actuation(self, state: DriverState) -> None:
        """Gọi actuation callback khi alert level thay đổi."""
        if self._actuation_callback:
            self._actuation_callback(state)
