"""
guardian_pilot/agents/m2_microsleep.py
-----------------------------------------
Agent M2 — Microsleep Detection (CNN_16s)
Model: cnn_16s_best.keras
Input: tín hiệu EEG/EOG 16s window — KHÔNG phải camera
Output: 4-class: Wake(0) / MSE(1) / MSEc(2) / ED(3)

Đặc thù (kiến trúc mục 5.3):
  - Tín hiệu "chậm nhưng sâu" — chỉ tin cậy khi có sensor EEG/EOG
  - Nếu KHÔNG có sensor → tự báo is_alive=False ngay khi khởi động
    (KHÔNG giả lập input — Orchestrator cần biết M2 không tham gia)
  - Class ED (3) → PATHOLOGICAL_PROXY — trigger RULE 1 EMERGENCY
"""

from __future__ import annotations

import os
from typing import Any, Optional

import numpy as np

from ..core.knowledge_graph import KnowledgeGraph
from ..core.schema import AgentID, InputQuality, NormalizedLabel
from .base_agent import ModelInput, PerceptionAgent, RawOutput

# Mapping class index → NormalizedLabel
CLASS_MAP = {
    0: NormalizedLabel.ALERT,              # Wake
    1: NormalizedLabel.MICROSLEEP,         # MSE  (Microsleep Episode)
    2: NormalizedLabel.MICROSLEEP,         # MSEc (Microsleep confirmed)
    3: NormalizedLabel.PATHOLOGICAL_PROXY, # ED   (Extreme Drowsiness / pathological)
}

# EEG/EOG sample rate (thường 128 Hz → 16s = 2048 samples)
DEFAULT_SAMPLE_RATE = 128
WINDOW_SECONDS      = 16


class M2MicrosleepAgent(PerceptionAgent):
    """
    Agent M2 — EEG/EOG microsleep detection.

    sensor_available=False: agent tự khai báo offline ngay khi khởi động.
    Đây là thiết kế bắt buộc — không được chạy với dữ liệu giả lập.
    """

    def __init__(
        self,
        kg: KnowledgeGraph,
        model_path: str,
        sensor_available: bool = False,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
    ) -> None:
        super().__init__(AgentID.M2_MICROSLEEP, kg)
        self.model_path      = model_path
        self.sensor_available = sensor_available
        self.sample_rate     = sample_rate
        self.window_size     = WINDOW_SECONDS * sample_rate
        self._model          = None

        # Nếu không có sensor → offline ngay lập tức
        if not sensor_available:
            kg.mark_agent_offline(
                AgentID.M2_MICROSLEEP,
                reason="EEG/EOG sensor không được lắp đặt"
            )
            print(
                "[M2] ⚠  Sensor EEG/EOG không khả dụng. "
                "Agent M2 offline — Orchestrator sẽ bỏ qua M2."
            )

    def _load_model(self) -> None:
        if self._model is not None:
            return
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"[M2] Không tìm thấy model tại: {self.model_path}"
            )
        import tensorflow as tf  # noqa: PLC0415
        self._model = tf.keras.models.load_model(self.model_path, compile=False)
        print(f"[M2] ✓ Model loaded: {self.model_path}")

    # ─────────────────────────────────────────
    #  Pipeline steps
    # ─────────────────────────────────────────

    def preprocess(self, raw_signal: Any) -> Optional[ModelInput]:
        """
        raw_signal: np.ndarray shape (channels, samples) hoặc (samples, channels)
        Kiểm tra đủ 16s window, chuẩn hóa đơn giản.
        """
        if not self.sensor_available:
            return None   # Không xử lý nếu offline

        if raw_signal is None:
            return None

        signal = np.array(raw_signal, dtype=np.float32)

        # Đảm bảo đúng shape (window_size, channels)
        if signal.ndim == 1:
            signal = signal.reshape(-1, 1)
        if signal.shape[0] < self.window_size:
            return None   # Chưa đủ 16s dữ liệu

        # Lấy 16s cuối
        window = signal[-self.window_size:, :]

        # Z-score normalization per channel
        mean = window.mean(axis=0, keepdims=True)
        std  = window.std(axis=0, keepdims=True) + 1e-8
        window_norm = (window - mean) / std

        tensor = np.expand_dims(window_norm, axis=0)  # (1, 2048, channels)
        return ModelInput(data={"tensor": tensor}, quality=InputQuality.GOOD)

    def infer(self, model_input: ModelInput) -> RawOutput:
        self._load_model()
        pred    = self._model.predict(model_input.data["tensor"], verbose=0)
        class_probs = pred[0]   # shape (4,)
        pred_class  = int(np.argmax(class_probs))
        return RawOutput(data={
            "class_probs": class_probs.tolist(),
            "pred_class":  pred_class,
            "max_prob":    float(np.max(class_probs)),
        })

    def normalize(self, raw_output: RawOutput) -> NormalizedLabel:
        pred_class = raw_output.data["pred_class"]
        return CLASS_MAP.get(pred_class, NormalizedLabel.UNKNOWN)

    def estimate_confidence(
        self, raw_output: RawOutput, input_quality: InputQuality
    ) -> float:
        """Confidence = xác suất softmax của class được chọn."""
        if input_quality != InputQuality.GOOD:
            return 0.0
        return float(raw_output.data.get("max_prob", 0.0))

    # ─────────────────────────────────────────
    #  Public: feed EEG/EOG signal
    # ─────────────────────────────────────────

    def process_signal(self, eeg_signal: np.ndarray) -> None:
        """Entry point thay vì run(frame) — nhận tín hiệu EEG/EOG."""
        self.run(eeg_signal)
