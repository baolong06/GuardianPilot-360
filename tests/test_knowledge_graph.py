"""
tests/test_knowledge_graph.py
--------------------------------
Unit test cho KnowledgeGraph — không cần model AI.
Chạy: python -m pytest tests/ -v
"""

import time
from datetime import datetime, timezone, timedelta

import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from guardian_pilot.core.knowledge_graph import KnowledgeGraph
from guardian_pilot.core.schema import (
    AgentID, AlertLevel, ConflictEvent, DriverState,
    InputQuality, NormalizedLabel, PerceptionResult,
)


@pytest.fixture
def kg():
    return KnowledgeGraph()


def make_result(
    agent: AgentID,
    label: NormalizedLabel = NormalizedLabel.ALERT,
    confidence: float = 0.8,
    quality: InputQuality = InputQuality.GOOD,
    age_ms: float = 0.0,          # 0 = fresh
) -> PerceptionResult:
    ts = datetime.now(timezone.utc) - timedelta(milliseconds=age_ms)
    return PerceptionResult(
        source_agent     = agent,
        timestamp        = ts,
        normalized_label = label,
        confidence       = confidence,
        input_quality    = quality,
    )


# ── Test: write + read PerceptionResult ──────────────────────

class TestPerceptionReadWrite:
    def test_write_and_read_fresh(self, kg):
        result = make_result(AgentID.M1_DROWSINESS, NormalizedLabel.DROWSY, 0.9)
        kg.write_perception(result)
        got = kg.get_latest_perception(AgentID.M1_DROWSINESS)
        assert got is not None
        assert got.normalized_label == NormalizedLabel.DROWSY
        assert got.confidence == pytest.approx(0.9)

    def test_stale_result_returns_none(self, kg):
        # Age 600ms > FRESH_THRESHOLD_MS=500ms
        result = make_result(AgentID.M1_DROWSINESS, age_ms=600)
        kg.write_perception(result)
        got = kg.get_latest_perception(AgentID.M1_DROWSINESS, max_age_ms=500)
        assert got is None

    def test_empty_history_returns_none(self, kg):
        got = kg.get_latest_perception(AgentID.M3_DISTRACTED)
        assert got is None

    def test_history_maxlen(self, kg):
        # Ghi 35 kết quả → chỉ giữ 30
        for i in range(35):
            kg.write_perception(make_result(AgentID.M4_LANDMARK_GAZE))
        history = kg.get_recent_perceptions(AgentID.M4_LANDMARK_GAZE, n=50)
        assert len(history) == 30


# ── Test: DriverState ─────────────────────────────────────────

class TestDriverState:
    def test_default_state(self, kg):
        state = kg.get_driver_state()
        assert state.current_alert_level == AlertLevel.NORMAL

    def test_write_and_read_state(self, kg):
        new_state = DriverState(
            current_alert_level = AlertLevel.SEVERE_WARNING,
            alert_reason        = "Test SEVERE",
            confidence          = 0.75,
        )
        kg.write_driver_state(new_state)
        got = kg.get_driver_state()
        assert got.current_alert_level == AlertLevel.SEVERE_WARNING
        assert got.confidence == pytest.approx(0.75)


# ── Test: AgentHealth ─────────────────────────────────────────

class TestAgentHealth:
    def test_initial_alive(self, kg):
        health = kg.get_health(AgentID.M1_DROWSINESS)
        assert health.is_alive is True
        assert health.consecutive_failures == 0

    def test_mark_failure_three_times(self, kg):
        kg.mark_agent_failure(AgentID.M3_DISTRACTED)
        kg.mark_agent_failure(AgentID.M3_DISTRACTED)
        assert kg.get_health(AgentID.M3_DISTRACTED).is_alive is True
        kg.mark_agent_failure(AgentID.M3_DISTRACTED)
        assert kg.get_health(AgentID.M3_DISTRACTED).is_alive is False

    def test_mark_offline(self, kg):
        kg.mark_agent_offline(AgentID.M2_MICROSLEEP)
        assert kg.get_health(AgentID.M2_MICROSLEEP).is_alive is False

    def test_write_perception_resets_failures(self, kg):
        # Failure → write perception → health alive lại
        kg.mark_agent_failure(AgentID.M1_DROWSINESS)
        kg.mark_agent_failure(AgentID.M1_DROWSINESS)
        result = make_result(AgentID.M1_DROWSINESS)
        kg.write_perception(result)
        health = kg.get_health(AgentID.M1_DROWSINESS)
        assert health.is_alive is True
        assert health.consecutive_failures == 0

    def test_alive_agents_excludes_offline(self, kg):
        kg.mark_agent_offline(AgentID.M2_MICROSLEEP)
        alive = kg.alive_agents()
        assert AgentID.M2_MICROSLEEP not in alive
        assert AgentID.M1_DROWSINESS in alive


# ── Test: Conflict log ────────────────────────────────────────

class TestConflictLog:
    def test_log_and_retrieve(self, kg):
        event = ConflictEvent(
            agents_involved    = [AgentID.M1_DROWSINESS, AgentID.M4_LANDMARK_GAZE],
            conflicting_labels = {"M1": "DROWSY", "M4": "ALERT"},
            resolution_strategy= "trust_landmark",
            final_label        = "MILD_WARNING",
        )
        kg.log_conflict(event)
        log = kg.get_conflict_log()
        assert len(log) == 1
        assert log[0].resolution_strategy == "trust_landmark"


# ── Test: Snapshot ────────────────────────────────────────────

class TestSnapshot:
    def test_snapshot_has_required_keys(self, kg):
        snap = kg.snapshot()
        for key in ("timestamp", "alert_level", "confidence", "alive_agents", "perceptions"):
            assert key in snap

    def test_snapshot_perception_reflects_latest(self, kg):
        result = make_result(AgentID.M1_DROWSINESS, NormalizedLabel.DROWSY, 0.88)
        kg.write_perception(result)
        snap = kg.snapshot()
        m1_snap = snap["perceptions"]["M1_Drowsiness"]
        assert m1_snap is not None
        assert m1_snap["label"] == "DROWSY"
