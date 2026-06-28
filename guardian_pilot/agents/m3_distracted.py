"""
guardian_pilot/agents/m3_distracted.py
-----------------------------------------
Agent M3 — Distracted Driver Detection (DBMNet + Baseline fallback)
Model chính:    dbmnet_full_task3.keras
Model fallback: baseline_ghostnetlike_task3.keras
Input: ảnh toàn cảnh cabin 224×224, giá trị pixel [0,255]
Output: 10-class hành vi mất tập trung

Đặc thù (kiến trúc mục 5.4):
  - Cơ chế fallback nội bộ: DBMNet confidence < threshold → dùng baseline
  - Nếu 2 model đồng thuận → confidence × 1.1
  - Nếu 2 model mâu thuẫn → confidence × 0.7 + log ConflictEvent
  - Nếu DBMNet lỗi runtime (OOM, exception) → tự động fallback hoàn toàn
"""

from __future__ import annotations

import os
from typing import Any, Optional

import numpy as np

from ..core.knowledge_graph import KnowledgeGraph
from ..core.schema import AgentID, ConflictEvent, InputQuality, NormalizedLabel
from .base_agent import ModelInput, PerceptionAgent, RawOutput

# 10 class State Farm distracted driver (chuẩn Task 3)
DISTRACTED_CLASSES = {
    0: ("safe_driving",        NormalizedLabel.ALERT),
    1: ("texting_right",       NormalizedLabel.DISTRACTED),
    2: ("phone_right",         NormalizedLabel.DISTRACTED),
    3: ("texting_left",        NormalizedLabel.DISTRACTED),
    4: ("phone_left",          NormalizedLabel.DISTRACTED),
    5: ("radio",               NormalizedLabel.MILD_CONCERN),
    6: ("drinking",            NormalizedLabel.MILD_CONCERN),
    7: ("reaching_behind",     NormalizedLabel.DISTRACTED),
    8: ("hair_makeup",         NormalizedLabel.MILD_CONCERN),
    9: ("talking_passenger",   NormalizedLabel.MILD_CONCERN),
}

DBMNET_CONFIDENCE_THRESHOLD = 0.50   # dưới ngưỡng này → gọi baseline


class M3DistractedAgent(PerceptionAgent):
    """
    Agent M3 với dual-model fallback theo đúng kiến trúc mục 5.4.
    """

    def __init__(
        self,
        kg: KnowledgeGraph,
        dbmnet_path: str,
        baseline_path: str,
    ) -> None:
        super().__init__(AgentID.M3_DISTRACTED, kg)
        self.dbmnet_path   = dbmnet_path
        self.baseline_path = baseline_path
        self._dbmnet    = None
        self._baseline  = None

    def _load_models(self) -> None:
        import tensorflow as tf  # noqa: PLC0415

        if self._dbmnet is None:
            if not os.path.exists(self.dbmnet_path):
                raise FileNotFoundError(f"[M3] DBMNet không tìm thấy: {self.dbmnet_path}")
            self._dbmnet = tf.keras.models.load_model(self.dbmnet_path, compile=False)
            print(f"[M3] ✓ DBMNet loaded: {self.dbmnet_path}")

        if self._baseline is None:
            if not os.path.exists(self.baseline_path):
                raise FileNotFoundError(f"[M3] Baseline không tìm thấy: {self.baseline_path}")
            self._baseline = tf.keras.models.load_model(self.baseline_path, compile=False)
            print(f"[M3] ✓ Baseline loaded: {self.baseline_path}")

    def _load_baseline_only(self) -> None:
        """Dùng khi DBMNet lỗi runtime → chỉ load baseline."""
        import tensorflow as tf  # noqa: PLC0415
        if self._baseline is None and os.path.exists(self.baseline_path):
            self._baseline = tf.keras.models.load_model(self.baseline_path, compile=False)
            self.kg.mark_agent_degraded(AgentID.M3_DISTRACTED)
            print("[M3] ⚠  DBMNet lỗi — chuyển sang baseline hoàn toàn.")

    # ─────────────────────────────────────────
    #  Pipeline steps
    # ─────────────────────────────────────────

    def preprocess(self, raw_frame: Any) -> Optional[ModelInput]:
        """
        raw_frame: np.ndarray BGR, bất kỳ kích thước.
        Output: ảnh 224×224 giá trị [0,255] (uint8).
        """
        import cv2  # noqa: PLC0415
        if raw_frame is None:
            return None

        frame_rgb = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
        resized   = cv2.resize(frame_rgb, (224, 224))
        # Giữ nguyên [0,255] — M3 không normalize như M1
        tensor    = np.expand_dims(resized.astype(np.float32), axis=0)

        # Đánh giá chất lượng (kiểm tra ảnh quá tối/quá sáng)
        mean_val  = float(np.mean(resized))
        quality   = (
            InputQuality.DEGRADED if (mean_val < 20 or mean_val > 235)
            else InputQuality.GOOD
        )
        return ModelInput(data={"tensor": tensor}, quality=quality)

    def infer(self, model_input: ModelInput) -> RawOutput:
        """
        Dual-model inference với fallback logic (mục 5.4):
        1. DBMNet chạy trước
        2. confidence < threshold → gọi baseline để xác nhận
        3. Nếu đồng thuận → boost confidence; mâu thuẫn → penalize + log
        """
        self._load_models()
        tensor = model_input.data["tensor"]

        # Chạy DBMNet chính
        dbm_out = self._dbmnet.predict(tensor, verbose=0)
        dbm_prob    = float(np.max(dbm_out[0]))
        dbm_class   = int(np.argmax(dbm_out[0]))

        if dbm_prob >= DBMNET_CONFIDENCE_THRESHOLD:
            # DBMNet đủ tự tin → không cần baseline
            return RawOutput(data={
                "class":       dbm_class,
                "confidence":  dbm_prob,
                "source":      "dbmnet",
                "all_probs":   dbm_out[0].tolist(),
            })

        # DBMNet không tự tin → gọi baseline
        bl_out   = self._baseline.predict(tensor, verbose=0)
        bl_class = int(np.argmax(bl_out[0]))

        if bl_class == dbm_class:
            # Đồng thuận → tăng nhẹ tin cậy
            final_conf = min(1.0, dbm_prob * 1.1)
            source     = "dbmnet+baseline_agree"
        else:
            # Mâu thuẫn nội bộ → giảm tin cậy + ghi ConflictEvent
            final_conf = dbm_prob * 0.7
            source     = "dbmnet+baseline_conflict"
            conflict   = ConflictEvent(
                agents_involved     = [AgentID.M3_DISTRACTED],
                conflicting_labels  = {
                    "dbmnet":   DISTRACTED_CLASSES[dbm_class][0],
                    "baseline": DISTRACTED_CLASSES.get(bl_class, (str(bl_class),))[0],
                },
                resolution_strategy = "trust_dbmnet_penalize_confidence",
                final_label         = DISTRACTED_CLASSES[dbm_class][0],
            )
            self.kg.log_conflict(conflict)

        return RawOutput(data={
            "class":       dbm_class,
            "confidence":  final_conf,
            "source":      source,
            "all_probs":   dbm_out[0].tolist(),
        })

    def normalize(self, raw_output: RawOutput) -> NormalizedLabel:
        cls = raw_output.data.get("class", 0)
        return DISTRACTED_CLASSES.get(cls, (None, NormalizedLabel.UNKNOWN))[1]

    def estimate_confidence(
        self, raw_output: RawOutput, input_quality: InputQuality
    ) -> float:
        conf = float(raw_output.data.get("confidence", 0.0))
        if input_quality == InputQuality.DEGRADED:
            conf *= 0.6
        if input_quality == InputQuality.MISSING:
            return 0.0
        return min(1.0, conf)
