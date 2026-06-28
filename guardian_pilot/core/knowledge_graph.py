"""
guardian_pilot/core/knowledge_graph.py
----------------------------------------
Blackboard / Knowledge Graph in-memory dùng networkx (MVP).
Thread-safe với RLock — nhiều agent đọc/ghi đồng thời an toàn.

Schema node:
  PerceptionResult  — 1 node / agent / frame, giữ lịch sử N=30
  DriverState       — 1 node duy nhất
  AgentHealth       — 1 node / agent
  ConflictEvent     — thêm mới mỗi khi xung đột
  ActuationEvent    — thêm mới mỗi khi actuation kích hoạt
"""

from __future__ import annotations

import threading
import time
from collections import deque
from datetime import datetime, timezone
from typing import Dict, List, Optional

import networkx as nx

from .schema import (
    AgentID, AgentHealth, AlertLevel, ActuationEvent,
    ConflictEvent, DriverState, PerceptionResult,
)

HISTORY_PER_AGENT   = 30      # giữ 30 PerceptionResult gần nhất / agent
FRESH_THRESHOLD_MS  = 500.0   # node cũ hơn 500ms bị coi là stale


class KnowledgeGraph:
    """
    Blackboard dùng chung giữa tất cả agent.
    Nội tại là một networkx DiGraph + các deque buffer per-agent.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._graph = nx.DiGraph()

        # --- DriverState (singleton) ---
        self._driver_state = DriverState()
        self._graph.add_node("driver_state", data=self._driver_state)

        # --- PerceptionResult history per agent ---
        self._perception_history: Dict[AgentID, deque[PerceptionResult]] = {
            aid: deque(maxlen=HISTORY_PER_AGENT) for aid in AgentID
        }

        # --- AgentHealth per agent ---
        self._agent_health: Dict[AgentID, AgentHealth] = {
            aid: AgentHealth(agent_id=aid) for aid in AgentID
        }

        # --- Conflict & Actuation logs ---
        self._conflict_log:   List[ConflictEvent]   = []
        self._actuation_log:  List[ActuationEvent]  = []

    # ──────────────────────────────────────────
    #  WRITE — Perception Result
    # ──────────────────────────────────────────

    def write_perception(self, result: PerceptionResult) -> None:
        """Agent ghi kết quả mới nhất vào KG."""
        with self._lock:
            agent = result.source_agent
            self._perception_history[agent].append(result)

            # Cập nhật AgentHealth
            health = self._agent_health[agent]
            health.is_alive = True
            health.last_successful_inference = result.timestamp
            health.consecutive_failures = 0

            # Graph edge: PerceptionResult → DriverState (semantic)
            node_id = f"{agent.value}_{result.timestamp.isoformat()}"
            self._graph.add_node(node_id, data=result)
            self._graph.add_edge(node_id, "driver_state", rel="CONTRIBUTES_TO")

    # ──────────────────────────────────────────
    #  WRITE — DriverState (Orchestrator ghi)
    # ──────────────────────────────────────────

    def write_driver_state(self, state: DriverState) -> None:
        with self._lock:
            self._driver_state = state
            self._graph.nodes["driver_state"]["data"] = state

    # ──────────────────────────────────────────
    #  WRITE — AgentHealth
    # ──────────────────────────────────────────

    def mark_agent_failure(self, agent_id: AgentID) -> None:
        with self._lock:
            health = self._agent_health[agent_id]
            health.consecutive_failures += 1
            if health.consecutive_failures >= 3:
                health.is_alive = False

    def mark_agent_offline(self, agent_id: AgentID, reason: str = "") -> None:
        """Agent tự khai báo offline (ví dụ: không có sensor)."""
        with self._lock:
            self._agent_health[agent_id].is_alive = False

    def mark_agent_degraded(self, agent_id: AgentID) -> None:
        with self._lock:
            self._agent_health[agent_id].degraded_mode = True

    # ──────────────────────────────────────────
    #  WRITE — Conflict / Actuation events
    # ──────────────────────────────────────────

    def log_conflict(self, event: ConflictEvent) -> None:
        with self._lock:
            self._conflict_log.append(event)

    def log_actuation(self, event: ActuationEvent) -> None:
        with self._lock:
            self._actuation_log.append(event)

    # ──────────────────────────────────────────
    #  READ — Latest PerceptionResult per agent
    # ──────────────────────────────────────────

    def get_latest_perception(
        self, agent_id: AgentID, max_age_ms: float = FRESH_THRESHOLD_MS
    ) -> Optional[PerceptionResult]:
        """Trả về PerceptionResult mới nhất còn 'tươi', hoặc None."""
        with self._lock:
            history = self._perception_history[agent_id]
            if not history:
                return None
            result = history[-1]
            return result if result.is_fresh(max_age_ms) else None

    def get_recent_perceptions(
        self, agent_id: AgentID, n: int = 5
    ) -> List[PerceptionResult]:
        """Lấy n PerceptionResult gần nhất (kể cả stale) để phân tích trend."""
        with self._lock:
            history = list(self._perception_history[agent_id])
            return history[-n:] if history else []

    # ──────────────────────────────────────────
    #  READ — DriverState
    # ──────────────────────────────────────────

    def get_driver_state(self) -> DriverState:
        with self._lock:
            return self._driver_state

    # ──────────────────────────────────────────
    #  READ — AgentHealth
    # ──────────────────────────────────────────

    def get_health(self, agent_id: AgentID) -> AgentHealth:
        with self._lock:
            return self._agent_health[agent_id]

    def get_all_health(self) -> Dict[AgentID, AgentHealth]:
        with self._lock:
            return dict(self._agent_health)

    def alive_agents(self) -> List[AgentID]:
        with self._lock:
            return [
                aid for aid, h in self._agent_health.items()
                if h.is_alive and aid != AgentID.ORCHESTRATOR
            ]

    # ──────────────────────────────────────────
    #  READ — Logs
    # ──────────────────────────────────────────

    def get_conflict_log(self, last_n: int = 10) -> List[ConflictEvent]:
        with self._lock:
            return self._conflict_log[-last_n:]

    def get_actuation_log(self, last_n: int = 10) -> List[ActuationEvent]:
        with self._lock:
            return self._actuation_log[-last_n:]

    # ──────────────────────────────────────────
    #  Utility
    # ──────────────────────────────────────────

    def snapshot(self) -> dict:
        """Trả về snapshot dạng dict dùng cho logging / debugging."""
        with self._lock:
            ds = self._driver_state
            alive = self.alive_agents()
            latest = {
                aid.value: (
                    {
                        "label": r.normalized_label.value,
                        "conf":  round(r.confidence, 3),
                        "quality": r.input_quality.value,
                        "age_ms": round(
                            (datetime.now(timezone.utc) - r.timestamp).total_seconds() * 1000, 1
                        ),
                    }
                    if (r := self._perception_history[aid][-1] if self._perception_history[aid] else None)
                    else None
                )
                for aid in AgentID if aid != AgentID.ORCHESTRATOR
            }
            return {
                "timestamp":     ds.timestamp.isoformat(),
                "alert_level":   ds.current_alert_level.value,
                "alert_reason":  ds.alert_reason,
                "confidence":    round(ds.confidence, 3),
                "alive_agents":  [a.value for a in alive],
                "perceptions":   latest,
                "conflicts":     len(self._conflict_log),
            }
