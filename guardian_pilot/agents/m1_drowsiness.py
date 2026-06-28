"""
guardian_pilot/agents/m1_drowsiness.py
-----------------------------------------
Agent M1 — Drowsiness Detection (DCNN)
Model: dcnn_drowsiness_task1_baseline.keras
Input: ảnh mặt 96×96, normalize [0,1]
Output: P(Drowsy) — xác suất buồn ngủ tức thời

Đặc thù (kiến trúc mục 5.2):
  - Tín hiệu nhanh, mỗi frame, độ trễ thấp — "cảm biến cảnh báo sớm"
  - estimate_confidence PHẢI giảm mạnh nếu ảnh quá tối/quá sáng
    (kiểm tra histogram trước khi đưa vào model)
"""

from __future__ import annotations

import os
from typing import Any, Optional

import numpy as np

from ..core.knowledge_graph import KnowledgeGraph
from ..core.schema import AgentID, InputQuality, NormalizedLabel
from .base_agent import ModelInput, PerceptionAgent, RawOutput

# Ngưỡng confidence để phân loại
DROWSY_THRESHOLD = 0.50   # P(Drowsy) ≥ 0.50 → DROWSY
MILD_THRESHOLD   = 0.35   # [0.35, 0.50) → MILD_CONCERN

# Ngưỡng ánh sáng (kiểm tra histogram trung bình pixel 0-1)
DARK_THRESHOLD   = 0.15   # ảnh quá tối (mean < 0.15)
BRIGHT_THRESHOLD = 0.85   # ảnh quá sáng (mean > 0.85)

# Penalty hệ số confidence khi ánh sáng kém
LIGHTING_PENALTY = 0.45   # nhân confidence với hệ số này


class M1DrowsinessAgent(PerceptionAgent):
    """
    Agent M1 bọc model DCNN phát hiện buồn ngủ tức thời.
    Lazy-load model: chỉ load khi lần đầu gọi infer().
    """

    def __init__(self, kg: KnowledgeGraph, model_path: str) -> None:
        super().__init__(AgentID.M1_DROWSINESS, kg)
        self.model_path = model_path
        self._model     = None   # lazy load

    def _load_model(self) -> None:
        if self._model is not None:
            return
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"[M1] Không tìm thấy model tại: {self.model_path}"
            )
        # Import tensorflow chỉ khi cần — không block startup
        import tensorflow as tf  # noqa: PLC0415
        self._model = tf.keras.models.load_model(self.model_path, compile=False)
        print(f"[M1] ✓ Model loaded: {self.model_path}")

    # ─────────────────────────────────────────
    #  Pipeline steps
    # ─────────────────────────────────────────

    def preprocess(self, raw_frame: Any) -> Optional[ModelInput]:
        """
        raw_frame: np.ndarray BGR hoặc RGB, bất kỳ kích thước.
        Bước 1: detect face / crop 96×96
        Bước 2: normalize [0,1]
        Bước 3: kiểm tra chất lượng ánh sáng
        """
        import cv2  # noqa: PLC0415

        if raw_frame is None:
            return None

        # Convert BGR → RGB nếu cần
        if len(raw_frame.shape) == 3 and raw_frame.shape[2] == 3:
            frame_rgb = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
        else:
            frame_rgb = raw_frame

        # Resize về 96×96
        face_img = cv2.resize(frame_rgb, (96, 96))
        face_norm = face_img.astype(np.float32) / 255.0

        # Đánh giá chất lượng ánh sáng
        mean_brightness = float(np.mean(face_norm))
        if mean_brightness < DARK_THRESHOLD:
            quality = InputQuality.DEGRADED
        elif mean_brightness > BRIGHT_THRESHOLD:
            quality = InputQuality.DEGRADED
        else:
            quality = InputQuality.GOOD

        # Shape (1, 96, 96, 3) cho batch inference
        input_tensor = np.expand_dims(face_norm, axis=0)
        return ModelInput(
            data    = {"tensor": input_tensor, "brightness": mean_brightness},
            quality = quality,
        )

    def infer(self, model_input: ModelInput) -> RawOutput:
        self._load_model()
        tensor    = model_input.data["tensor"]
        pred      = self._model.predict(tensor, verbose=0)
        # Hỗ trợ output shape (1,1) hoặc (1,2)
        if pred.shape[-1] == 1:
            drowsy_prob = float(pred[0, 0])
        else:
            drowsy_prob = float(pred[0, 1])   # class index 1 = Drowsy
        return RawOutput(data={"drowsy_prob": drowsy_prob})

    def normalize(self, raw_output: RawOutput) -> NormalizedLabel:
        prob = raw_output.data["drowsy_prob"]
        if prob >= DROWSY_THRESHOLD:
            return NormalizedLabel.DROWSY
        elif prob >= MILD_THRESHOLD:
            return NormalizedLabel.MILD_CONCERN
        return NormalizedLabel.ALERT

    def estimate_confidence(
        self, raw_output: RawOutput, input_quality: InputQuality
    ) -> float:
        """
        Confidence = softmax score, NHƯNG:
        - Nếu lighting degraded → nhân penalty mạnh (ảnh tối/sáng quá dễ fool DCNN)
        - Nếu input missing → 0.0
        """
        prob = raw_output.data.get("drowsy_prob", 0.5)

        # Distance từ ranh giới quyết định (0.5) = confidence cơ bản
        base_conf = abs(prob - 0.5) * 2.0   # scale về [0,1]

        if input_quality == InputQuality.MISSING:
            return 0.0
        if input_quality == InputQuality.DEGRADED:
            return base_conf * LIGHTING_PENALTY

        return min(1.0, base_conf)
