"""
guardian_pilot/core/schema.py
------------------------------
Định nghĩa tất cả kiểu dữ liệu dùng chung giữa các agent.
Không import bất kỳ model AI nào tại đây.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


# ─────────────────────────────────────────────────────────
#  Enums
# ─────────────────────────────────────────────────────────

class AgentID(str, Enum):
    M1_DROWSINESS     = "M1_Drowsiness"
    M2_MICROSLEEP     = "M2_Microsleep"
    M3_DISTRACTED     = "M3_Distracted"
    M4_LANDMARK_GAZE  = "M4_LandmarkGaze"
    ORCHESTRATOR      = "Orchestrator"


class NormalizedLabel(str, Enum):
    ALERT             = "ALERT"
    MILD_CONCERN      = "MILD_CONCERN"
    DROWSY            = "DROWSY"
    DISTRACTED        = "DISTRACTED"
    MICROSLEEP        = "MICROSLEEP"
    PATHOLOGICAL_PROXY = "PATHOLOGICAL_PROXY"
    UNKNOWN           = "UNKNOWN"


class AlertLevel(str, Enum):
    NORMAL            = "NORMAL"
    MILD_WARNING      = "MILD_WARNING"
    SEVERE_WARNING    = "SEVERE_WARNING"
    EMERGENCY         = "EMERGENCY"


class InputQuality(str, Enum):
    GOOD     = "GOOD"
    DEGRADED = "DEGRADED"
    MISSING  = "MISSING"


# ─────────────────────────────────────────────────────────
#  KG Node dataclasses
# ─────────────────────────────────────────────────────────

def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class PerceptionResult:
    """Node: kết quả từ 1 Perception Agent tại 1 thời điểm."""
    source_agent:        AgentID
    timestamp:           datetime               = field(default_factory=_now)
    raw_output:          Dict[str, Any]         = field(default_factory=dict)
    normalized_label:    NormalizedLabel        = NormalizedLabel.UNKNOWN
    confidence:          float                  = 0.0          # [0, 1]
    input_quality:       InputQuality           = InputQuality.GOOD
    processing_latency_ms: float               = 0.0

    def is_fresh(self, max_age_ms: float = 500.0) -> bool:
        """Kiểm tra node còn 'tươi' (< max_age_ms ms)."""
        age = (datetime.now(timezone.utc) - self.timestamp).total_seconds() * 1000
        return age < max_age_ms


@dataclass
class DriverState:
    """Node duy nhất, cập nhật liên tục — quyết định cuối của Orchestrator."""
    timestamp:           datetime               = field(default_factory=_now)
    current_alert_level: AlertLevel             = AlertLevel.NORMAL
    alert_reason:        str                    = "Khởi tạo hệ thống"
    confidence:          float                  = 1.0
    active_agents:       List[AgentID]         = field(default_factory=list)


@dataclass
class AgentHealth:
    """Node theo dõi sức khoẻ vận hành của từng agent."""
    agent_id:                    AgentID
    is_alive:                    bool       = True
    last_successful_inference:   Optional[datetime] = None
    consecutive_failures:        int        = 0
    degraded_mode:               bool       = False   # đang dùng fallback


@dataclass
class ConflictEvent:
    """Node ghi nhận xung đột giữa 2+ agent."""
    timestamp:           datetime               = field(default_factory=_now)
    agents_involved:     List[AgentID]         = field(default_factory=list)
    conflicting_labels:  Dict[str, str]        = field(default_factory=dict)
    resolution_strategy: str                   = ""
    final_label:         str                   = ""


@dataclass
class ActuationEvent:
    """Node ghi nhận hành động đã kích hoạt."""
    timestamp:   datetime   = field(default_factory=_now)
    alert_level: AlertLevel = AlertLevel.NORMAL
    actions:     List[str]  = field(default_factory=list)
    reason:      str        = ""
