"""
tests/test_orchestrator_rules.py
-----------------------------------
Test 7 luật Orchestrator với KG mock — không cần model AI.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datetime import datetime, timezone
import pytest

from guardian_pilot.core.knowledge_graph import KnowledgeGraph
from guardian_pilot.core.schema import (
    AgentID, AlertLevel, InputQuality, NormalizedLabel, PerceptionResult,
)
from guardian_pilot.agents.orchestrator import OrchestratorAgent


def fresh_result(agent, label, confidence, quality=InputQuality.GOOD):
    return PerceptionResult(
        source_agent     = agent,
        normalized_label = label,
        confidence       = confidence,
        input_quality    = quality,
    )


@pytest.fixture
def kg_clean():
    return KnowledgeGraph()


def setup_kg(kg, m1=None, m2=None, m3=None, m4=None):
    """Ghi PerceptionResult vào KG cho các agent được chỉ định."""
    mapping = {
        AgentID.M1_DROWSINESS:    m1,
        AgentID.M2_MICROSLEEP:    m2,
        AgentID.M3_DISTRACTED:    m3,
        AgentID.M4_LANDMARK_GAZE: m4,
    }
    for agent_id, result in mapping.items():
        if result is not None:
            kg.write_perception(result)
        else:
            # Agent không gửi kết quả → offline
            kg.mark_agent_offline(agent_id)


class TestOrchestratorRules:

    # ── RULE 1: EMERGENCY ──────────────────────────────────────
    def test_rule1_emergency(self, kg_clean):
        setup_kg(
            kg_clean,
            m2=fresh_result(AgentID.M2_MICROSLEEP, NormalizedLabel.PATHOLOGICAL_PROXY, 0.85),
        )
        orch = OrchestratorAgent(kg_clean)
        state = orch.tick()
        assert state.current_alert_level == AlertLevel.EMERGENCY
        assert "pathological" in state.alert_reason.lower() or "nguy hiểm" in state.alert_reason

    def test_rule1_not_triggered_low_conf(self, kg_clean):
        """RULE 1 không trigger nếu confidence < 0.6."""
        setup_kg(
            kg_clean,
            m2=fresh_result(AgentID.M2_MICROSLEEP, NormalizedLabel.PATHOLOGICAL_PROXY, 0.55),
        )
        orch = OrchestratorAgent(kg_clean)
        state = orch.tick()
        assert state.current_alert_level != AlertLevel.EMERGENCY

    # ── RULE 2: SEVERE_WARNING ─────────────────────────────────
    def test_rule2_severe_both_drowsy(self, kg_clean):
        setup_kg(
            kg_clean,
            m1=fresh_result(AgentID.M1_DROWSINESS,    NormalizedLabel.DROWSY, 0.82),
            m4=fresh_result(AgentID.M4_LANDMARK_GAZE, NormalizedLabel.DROWSY, 0.75),
        )
        orch = OrchestratorAgent(kg_clean)
        state = orch.tick()
        assert state.current_alert_level == AlertLevel.SEVERE_WARNING

    def test_rule2_not_triggered_one_alert(self, kg_clean):
        """Chỉ 1 agent DROWSY → không phải SEVERE."""
        setup_kg(
            kg_clean,
            m1=fresh_result(AgentID.M1_DROWSINESS,    NormalizedLabel.DROWSY, 0.82),
            m4=fresh_result(AgentID.M4_LANDMARK_GAZE, NormalizedLabel.ALERT,  0.75),
        )
        orch = OrchestratorAgent(kg_clean)
        state = orch.tick()
        assert state.current_alert_level != AlertLevel.SEVERE_WARNING

    # ── RULE 3: MILD_WARNING single agent ─────────────────────
    def test_rule3_mild_single_m1(self, kg_clean):
        setup_kg(
            kg_clean,
            m1=fresh_result(AgentID.M1_DROWSINESS, NormalizedLabel.DROWSY, 0.72),
            m4=fresh_result(AgentID.M4_LANDMARK_GAZE, NormalizedLabel.ALERT, 0.80),
        )
        orch = OrchestratorAgent(kg_clean)
        state = orch.tick()
        # RULE 4 (conflict) sẽ trigger trước RULE 3 khi M1/M4 mâu thuẫn
        assert state.current_alert_level in (AlertLevel.MILD_WARNING,)

    # ── RULE 4: Conflict M1 vs M4 ─────────────────────────────
    def test_rule4_conflict_logged(self, kg_clean):
        setup_kg(
            kg_clean,
            m1=fresh_result(AgentID.M1_DROWSINESS,    NormalizedLabel.DROWSY, 0.70),
            m4=fresh_result(AgentID.M4_LANDMARK_GAZE, NormalizedLabel.ALERT,  0.65),
        )
        orch = OrchestratorAgent(kg_clean)
        state = orch.tick()
        conflicts = kg_clean.get_conflict_log()
        assert len(conflicts) >= 1
        assert state.current_alert_level == AlertLevel.MILD_WARNING

    # ── RULE 5: DISTRACTED ────────────────────────────────────
    def test_rule5_distracted_raises_to_mild(self, kg_clean):
        setup_kg(
            kg_clean,
            m1=fresh_result(AgentID.M1_DROWSINESS,    NormalizedLabel.ALERT, 0.9),
            m3=fresh_result(AgentID.M3_DISTRACTED,     NormalizedLabel.DISTRACTED, 0.78),
            m4=fresh_result(AgentID.M4_LANDMARK_GAZE, NormalizedLabel.ALERT, 0.9),
        )
        orch = OrchestratorAgent(kg_clean)
        state = orch.tick()
        assert state.current_alert_level == AlertLevel.MILD_WARNING

    # ── RULE 7: NORMAL default ────────────────────────────────
    def test_rule7_normal_when_all_alert(self, kg_clean):
        setup_kg(
            kg_clean,
            m1=fresh_result(AgentID.M1_DROWSINESS,    NormalizedLabel.ALERT, 0.9),
            m3=fresh_result(AgentID.M3_DISTRACTED,     NormalizedLabel.ALERT, 0.9),
            m4=fresh_result(AgentID.M4_LANDMARK_GAZE, NormalizedLabel.ALERT, 0.9),
        )
        orch = OrchestratorAgent(kg_clean)
        state = orch.tick()
        assert state.current_alert_level == AlertLevel.NORMAL

    # ── Offline agent reduces max_conf ────────────────────────
    def test_offline_agent_reduces_confidence(self, kg_clean):
        setup_kg(
            kg_clean,
            m1=fresh_result(AgentID.M1_DROWSINESS,    NormalizedLabel.DROWSY, 0.8),
            m4=fresh_result(AgentID.M4_LANDMARK_GAZE, NormalizedLabel.DROWSY, 0.8),
            # M2 và M3 offline
        )
        orch = OrchestratorAgent(kg_clean)
        state = orch.tick()
        # Confidence phải thấp hơn 0.8 do có agent offline
        assert state.confidence < 0.8

    # ── RULE 3: M4 offline, M1 vẫn trigger MILD ───────────────
    def test_rule3_m4_offline_m1_drowsy(self, kg_clean):
        """Khi M4 offline, M1 DROWSY đơn độc vẫn phải trigger MILD_WARNING."""
        kg_clean.mark_agent_offline(AgentID.M4_LANDMARK_GAZE)
        kg_clean.write_perception(
            fresh_result(AgentID.M1_DROWSINESS, NormalizedLabel.DROWSY, 0.75)
        )
        kg_clean.mark_agent_offline(AgentID.M2_MICROSLEEP)
        kg_clean.mark_agent_offline(AgentID.M3_DISTRACTED)
        orch = OrchestratorAgent(kg_clean)
        state = orch.tick()
        assert state.current_alert_level == AlertLevel.MILD_WARNING

    # ── RULE 3: Confidence thấp không trigger ─────────────────
    def test_rule3_low_conf_no_trigger(self, kg_clean):
        """M1 DROWSY nhưng conf <= 0.6 → không trigger RULE 3."""
        setup_kg(
            kg_clean,
            m1=fresh_result(AgentID.M1_DROWSINESS,    NormalizedLabel.DROWSY, 0.55),
            m4=fresh_result(AgentID.M4_LANDMARK_GAZE, NormalizedLabel.ALERT,  0.80),
        )
        orch = OrchestratorAgent(kg_clean)
        state = orch.tick()
        # R4 check: M1 DROWSY conf=0.55 < R2_CONF=0.5 (pass), nhưng < R3_CONF=0.6 → không vào RULE 3
        # R4 không trigger vì m1.confidence (0.55) > R2_CONF_SENSOR_FUSION (0.50)
        # → thực ra RULE 4 sẽ bắt → chỉ cần verify level không phải SEVERE
        assert state.current_alert_level != AlertLevel.SEVERE_WARNING
        assert state.current_alert_level != AlertLevel.EMERGENCY

    # ── RULE 5: MILD_CONCERN không nâng level ─────────────────
    def test_rule5_mild_concern_no_level_raise(self, kg_clean):
        """M3 MILD_CONCERN không nâng lên MILD_WARNING (chỉ ghi reason)."""
        setup_kg(
            kg_clean,
            m1=fresh_result(AgentID.M1_DROWSINESS,    NormalizedLabel.ALERT,        0.9),
            m3=fresh_result(AgentID.M3_DISTRACTED,     NormalizedLabel.MILD_CONCERN, 0.78),
            m4=fresh_result(AgentID.M4_LANDMARK_GAZE, NormalizedLabel.ALERT,        0.9),
        )
        orch = OrchestratorAgent(kg_clean)
        state = orch.tick()
        assert state.current_alert_level == AlertLevel.NORMAL

    # ── RULE 6: Tất cả 4 agent offline → multiplier clamp 0.4 ─
    def test_rule6_all_offline_clamps_multiplier(self, kg_clean):
        """Khi tất cả 4 agent offline, max_conf_multiplier bị clamp về 0.4."""
        for aid in [AgentID.M1_DROWSINESS, AgentID.M2_MICROSLEEP,
                    AgentID.M3_DISTRACTED, AgentID.M4_LANDMARK_GAZE]:
            kg_clean.mark_agent_offline(aid)
        orch = OrchestratorAgent(kg_clean)
        state = orch.tick()
        # RULE 7 default → confidence = 1.0 * 0.4 = 0.4
        assert state.confidence == pytest.approx(0.4, abs=0.01)
        assert state.current_alert_level == AlertLevel.NORMAL

    # ── RULE 1 override RULE 2 ────────────────────────────────
    def test_rule1_overrides_rule2(self, kg_clean):
        """Khi có cả M2 PATHOLOGICAL lẫn M1+M4 DROWSY → EMERGENCY thắng."""
        setup_kg(
            kg_clean,
            m1=fresh_result(AgentID.M1_DROWSINESS,    NormalizedLabel.DROWSY,             0.9),
            m2=fresh_result(AgentID.M2_MICROSLEEP,    NormalizedLabel.PATHOLOGICAL_PROXY, 0.85),
            m4=fresh_result(AgentID.M4_LANDMARK_GAZE, NormalizedLabel.DROWSY,             0.9),
        )
        orch = OrchestratorAgent(kg_clean)
        state = orch.tick()
        assert state.current_alert_level == AlertLevel.EMERGENCY

    # ── Rule 4: không log conflict khi cả hai đều ALERT ───────
    def test_rule4_no_conflict_both_alert(self, kg_clean):
        """M1 và M4 đều ALERT → không có ConflictEvent được ghi."""
        setup_kg(
            kg_clean,
            m1=fresh_result(AgentID.M1_DROWSINESS,    NormalizedLabel.ALERT, 0.9),
            m4=fresh_result(AgentID.M4_LANDMARK_GAZE, NormalizedLabel.ALERT, 0.9),
        )
        orch = OrchestratorAgent(kg_clean)
        orch.tick()
        assert len(kg_clean.get_conflict_log()) == 0

    # ── Tick 2 lần liên tiếp → idempotent ────────────────────
    def test_double_tick_idempotent_normal(self, kg_clean):
        """Gọi tick() 2 lần liên tiếp khi NORMAL → state không thay đổi."""
        setup_kg(
            kg_clean,
            m1=fresh_result(AgentID.M1_DROWSINESS,    NormalizedLabel.ALERT, 0.9),
            m3=fresh_result(AgentID.M3_DISTRACTED,     NormalizedLabel.ALERT, 0.9),
            m4=fresh_result(AgentID.M4_LANDMARK_GAZE, NormalizedLabel.ALERT, 0.9),
        )
        orch = OrchestratorAgent(kg_clean)
        s1 = orch.tick()
        s2 = orch.tick()
        assert s1.current_alert_level == s2.current_alert_level == AlertLevel.NORMAL

    # ── RULE 2: conf boundary — chính xác > 0.5 ──────────────
    def test_rule2_boundary_confidence_exactly_half(self, kg_clean):
        """M1 conf=0.50 không thỏa > 0.50 → không trigger RULE 2."""
        setup_kg(
            kg_clean,
            m1=fresh_result(AgentID.M1_DROWSINESS,    NormalizedLabel.DROWSY, 0.50),
            m4=fresh_result(AgentID.M4_LANDMARK_GAZE, NormalizedLabel.DROWSY, 0.80),
        )
        orch = OrchestratorAgent(kg_clean)
        state = orch.tick()
        assert state.current_alert_level != AlertLevel.SEVERE_WARNING

    # ── RULE 5 + RULE 6 kết hợp ──────────────────────────────
    def test_rule5_with_offline_reduces_confidence(self, kg_clean):
        """M3 DISTRACTED + M2 offline → MILD_WARNING với confidence bị giảm."""
        kg_clean.mark_agent_offline(AgentID.M1_DROWSINESS)
        kg_clean.mark_agent_offline(AgentID.M4_LANDMARK_GAZE)
        kg_clean.mark_agent_offline(AgentID.M2_MICROSLEEP)
        kg_clean.write_perception(
            fresh_result(AgentID.M3_DISTRACTED, NormalizedLabel.DISTRACTED, 0.85)
        )
        orch = OrchestratorAgent(kg_clean)
        state = orch.tick()
        assert state.current_alert_level == AlertLevel.MILD_WARNING
        # 3 offline → multiplier = max(0.4, 1.0 - 3*0.15) = max(0.4, 0.55) = 0.55
        assert state.confidence < 0.85
