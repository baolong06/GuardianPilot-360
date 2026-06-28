"""
guardian_pilot/agents/base_agent.py
--------------------------------------
Abstract base class cho mọi Perception Agent.
Triển khai chính xác interface trong kiến trúc mục 5.1.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Optional

from ..core.knowledge_graph import KnowledgeGraph
from ..core.schema import (
    AgentID, InputQuality, NormalizedLabel, PerceptionResult,
)


class ModelInput:
    """Container chứa input đã được preprocess."""
    def __init__(self, data: Any, quality: InputQuality = InputQuality.GOOD):
        self.data    = data
        self.quality = quality


class RawOutput:
    """Container chứa output thô của model.predict()."""
    def __init__(self, data: Any):
        self.data = data


class PerceptionAgent(ABC):
    """
    Template pattern cho tất cả Perception Agent (M1–M4).

    Pipeline chuẩn mỗi tick:
      preprocess() → infer() → normalize() → estimate_confidence() → write KG
    """

    def __init__(self, agent_id: AgentID, kg: KnowledgeGraph) -> None:
        self.agent_id = agent_id
        self.kg       = kg

    # ─────────────────────────────────────────
    #  Abstract interface — phải implement
    # ─────────────────────────────────────────

    @abstractmethod
    def preprocess(self, raw_frame: Any) -> Optional[ModelInput]:
        """
        Biến frame thô thành input đúng format model cần.
        Trả về None nếu input quality kém (mặt không detect được, v.v.).
        """

    @abstractmethod
    def infer(self, model_input: ModelInput) -> RawOutput:
        """Gọi model.predict(), trả về output thô."""

    @abstractmethod
    def normalize(self, raw_output: RawOutput) -> NormalizedLabel:
        """Map output thô về NormalizedLabel chuẩn."""

    @abstractmethod
    def estimate_confidence(
        self, raw_output: RawOutput, input_quality: InputQuality
    ) -> float:
        """
        Tính độ tin cậy — KHÔNG chỉ dựa vào softmax score.
        Xét cả input_quality: DEGRADED → giảm confidence dù model
        vẫn cho kết quả.
        """

    # ─────────────────────────────────────────
    #  Pipeline chuẩn — không override
    # ─────────────────────────────────────────

    def run(self, raw_frame: Any) -> PerceptionResult:
        """
        Chạy toàn bộ pipeline và ghi kết quả vào KG.
        Mọi exception đều được bắt → agent báo failure, không crash.
        """
        t_start = time.perf_counter()
        try:
            model_input = self.preprocess(raw_frame)

            if model_input is None:
                result = self._emit_degraded_result(reason="preprocess_failed")
            else:
                raw_out  = self.infer(model_input)
                label    = self.normalize(raw_out)
                conf     = self.estimate_confidence(raw_out, model_input.quality)
                latency  = (time.perf_counter() - t_start) * 1000

                result = PerceptionResult(
                    source_agent       = self.agent_id,
                    raw_output         = self._raw_to_dict(raw_out),
                    normalized_label   = label,
                    confidence         = conf,
                    input_quality      = model_input.quality,
                    processing_latency_ms = latency,
                )

            self.kg.write_perception(result)
            return result

        except Exception as exc:  # noqa: BLE001
            self.kg.mark_agent_failure(self.agent_id)
            error_result = self._emit_error_result(str(exc))
            # Vẫn ghi vào KG để Orchestrator biết agent đang có vấn đề
            self.kg.write_perception(error_result)
            return error_result

    # ─────────────────────────────────────────
    #  Helpers
    # ─────────────────────────────────────────

    def _emit_degraded_result(self, reason: str = "") -> PerceptionResult:
        return PerceptionResult(
            source_agent       = self.agent_id,
            raw_output         = {"degraded_reason": reason},
            normalized_label   = NormalizedLabel.UNKNOWN,
            confidence         = 0.0,
            input_quality      = InputQuality.DEGRADED,
            processing_latency_ms = 0.0,
        )

    def _emit_error_result(self, error: str) -> PerceptionResult:
        return PerceptionResult(
            source_agent       = self.agent_id,
            raw_output         = {"error": error},
            normalized_label   = NormalizedLabel.UNKNOWN,
            confidence         = 0.0,
            input_quality      = InputQuality.MISSING,
            processing_latency_ms = 0.0,
        )

    @staticmethod
    def _raw_to_dict(raw_out: RawOutput) -> dict:
        """Chuyển RawOutput về dict để lưu vào KG."""
        if isinstance(raw_out.data, dict):
            return raw_out.data
        if hasattr(raw_out.data, "tolist"):  # numpy array
            return {"values": raw_out.data.tolist()}
        return {"raw": str(raw_out.data)}
