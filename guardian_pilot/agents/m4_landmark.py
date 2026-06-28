"""
guardian_pilot/agents/m4_landmark.py
---------------------------------------
Agent M4 — Landmark & Gaze Detection (MLP/LSTM)
Model chính:  lstm_landmark_task4_fixed.keras  (temporal, online monitoring)
Model phụ:    mlp_landmark_task4_fixed.keras   (1 frame, nhanh hơn)
Preprocessor: face_landmarker.task (MediaPipe)
Scaler:       landmark_scaler_task4.pkl (StandardScaler — PHẢI dùng đúng file đã fit)
Input:        Sequence 15 frame × 1440-dim landmark
Output:       Binary: Alert(0) / Drowsy(1)

Đặc thù (kiến trúc mục 5.5):
  - LSTM làm chính (xét tinh thần temporal/online monitoring)
  - MLP làm tín hiệu phụ đối chiếu nhanh (1 frame, không cần đủ 15 frame)
  - PHẢI dùng đúng scaler đã lưu, KHÔNG fit lại (lệch phân phối → sai hoàn toàn)
  - Input là landmark hình học → ít bị ảnh hưởng ánh sáng hơn M1
"""

from __future__ import annotations

import os
import pickle
from collections import deque
from typing import Any, Optional

import numpy as np

from ..core.knowledge_graph import KnowledgeGraph
from ..core.schema import AgentID, InputQuality, NormalizedLabel
from .base_agent import ModelInput, PerceptionAgent, RawOutput

LANDMARK_DIM      = 1440    # 468 face landmarks × 3 coords (x, y, z)
SEQUENCE_LENGTH   = 15      # LSTM cần 15 frame buffer
DROWSY_THRESHOLD  = 0.50    # P(Drowsy) ≥ 0.50 → DROWSY


class M4LandmarkAgent(PerceptionAgent):
    """
    Agent M4 — landmark-based alertness detection.
    Dùng MediaPipe FaceLandmarker để extract landmarks từ frame.
    """

    def __init__(
        self,
        kg: KnowledgeGraph,
        lstm_path: str,
        mlp_path: str,
        scaler_path: str,
        landmarker_path: str,
    ) -> None:
        super().__init__(AgentID.M4_LANDMARK_GAZE, kg)
        self.lstm_path       = lstm_path
        self.mlp_path        = mlp_path
        self.scaler_path     = scaler_path
        self.landmarker_path = landmarker_path

        self._lstm_model   = None
        self._mlp_model    = None
        self._scaler       = None
        self._landmarker   = None

        # Buffer sequence 15 frame cho LSTM
        self._frame_buffer: deque = deque(maxlen=SEQUENCE_LENGTH)

        # Validate scaler ngay khi khởi động
        self._validate_scaler()

    def _validate_scaler(self) -> None:
        """
        Bắt buộc check scaler tồn tại trước khi khởi động.
        Nếu không load được → agent tự khai offline (mục 7, bảng lỗi).
        """
        if not os.path.exists(self.scaler_path):
            self.kg.mark_agent_offline(
                AgentID.M4_LANDMARK_GAZE,
                reason="landmark_scaler_task4.pkl không tìm thấy"
            )
            raise RuntimeError(
                f"[M4] CRITICAL: Scaler không tồn tại tại {self.scaler_path}. "
                "Không được dùng input chưa chuẩn hóa — agent M4 offline."
            )

    def _load_resources(self) -> None:
        """Lazy-load tất cả resources (models + scaler + landmarker)."""
        import tensorflow as tf  # noqa: PLC0415

        if self._scaler is None:
            with open(self.scaler_path, "rb") as f:
                self._scaler = pickle.load(f)
            print(f"[M4] ✓ Scaler loaded: {self.scaler_path}")

        if self._lstm_model is None:
            self._lstm_model = tf.keras.models.load_model(self.lstm_path, compile=False)
            print(f"[M4] ✓ LSTM loaded: {self.lstm_path}")

        if self._mlp_model is None:
            self._mlp_model = tf.keras.models.load_model(self.mlp_path, compile=False)
            print(f"[M4] ✓ MLP loaded: {self.mlp_path}")

        if self._landmarker is None:
            self._load_landmarker()

    def _load_landmarker(self) -> None:
        """Load MediaPipe FaceLandmarker."""
        try:
            import mediapipe as mp  # noqa: PLC0415
            BaseOptions = mp.tasks.BaseOptions
            FaceLandmarker = mp.tasks.vision.FaceLandmarker
            FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
            VisionRunningMode = mp.tasks.vision.RunningMode

            options = FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=self.landmarker_path),
                running_mode=VisionRunningMode.IMAGE,
                num_faces=1,
            )
            self._landmarker = FaceLandmarker.create_from_options(options)
            print(f"[M4] ✓ FaceLandmarker loaded: {self.landmarker_path}")
        except ImportError:
            print("[M4] ⚠  mediapipe chưa cài. Dùng fallback zero-vector.")
            self._landmarker = "FALLBACK"

    def _extract_landmarks(self, frame_rgb: np.ndarray) -> Optional[np.ndarray]:
        """
        Dùng MediaPipe để extract 1440-dim landmark vector.
        Trả về None nếu không detect được mặt.
        """
        if self._landmarker == "FALLBACK":
            # Không có mediapipe → không thể extract landmark thật
            return None

        import mediapipe as mp  # noqa: PLC0415
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame_rgb.astype(np.uint8),
        )
        result = self._landmarker.detect(mp_image)

        if not result.face_landmarks:
            return None   # Không detect được mặt

        # Flatten 468 landmarks × 3 coords = 1440-dim vector
        lm_list = result.face_landmarks[0]
        vec = np.array(
            [[lm.x, lm.y, lm.z] for lm in lm_list], dtype=np.float32
        ).flatten()

        if vec.shape[0] != LANDMARK_DIM:
            return None

        return vec

    # ─────────────────────────────────────────
    #  Pipeline steps
    # ─────────────────────────────────────────

    def preprocess(self, raw_frame: Any) -> Optional[ModelInput]:
        """
        1. Extract MediaPipe landmarks từ frame
        2. Scale với landmark_scaler_task4.pkl (StandardScaler)
        3. Thêm vào buffer sequence 15 frame
        4. Trả về ModelInput khi đủ 15 frame (cho LSTM)
           hoặc 1 frame (cho MLP nhanh)
        """
        import cv2  # noqa: PLC0415
        self._load_resources()

        if raw_frame is None:
            return None

        frame_rgb = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
        landmarks = self._extract_landmarks(frame_rgb)

        if landmarks is None:
            return None   # Không detect được mặt → DEGRADED

        # Chuẩn hóa bằng scaler đã fit (KHÔNG fit lại)
        lm_scaled = self._scaler.transform(landmarks.reshape(1, -1)).flatten()

        # Thêm vào buffer
        self._frame_buffer.append(lm_scaled)

        # Cung cấp cả sequence (LSTM) và single frame (MLP)
        has_sequence = len(self._frame_buffer) >= SEQUENCE_LENGTH
        data = {
            "single_frame": lm_scaled,
            "sequence":     np.array(list(self._frame_buffer)) if has_sequence else None,
            "buffer_len":   len(self._frame_buffer),
        }
        return ModelInput(data=data, quality=InputQuality.GOOD)

    def infer(self, model_input: ModelInput) -> RawOutput:
        """
        LSTM là chính nếu đủ 15 frame buffer.
        MLP là phụ (đối chiếu nhanh 1 frame).
        """
        self._load_resources()
        data = model_input.data
        results = {}

        # MLP: luôn chạy (1 frame)
        single = data["single_frame"].reshape(1, -1)
        mlp_pred = self._mlp_model.predict(single, verbose=0)[0]
        results["mlp_drowsy_prob"] = float(
            mlp_pred[1] if mlp_pred.shape[0] > 1 else mlp_pred[0]
        )

        # LSTM: chạy khi đủ 15 frame
        if data["sequence"] is not None:
            seq = data["sequence"].reshape(1, SEQUENCE_LENGTH, LANDMARK_DIM)
            lstm_pred = self._lstm_model.predict(seq, verbose=0)[0]
            results["lstm_drowsy_prob"] = float(
                lstm_pred[1] if lstm_pred.shape[0] > 1 else lstm_pred[0]
            )
            results["primary"] = "lstm"
        else:
            results["primary"] = "mlp"

        return RawOutput(data=results)

    def normalize(self, raw_output: RawOutput) -> NormalizedLabel:
        """Dùng LSTM làm chính nếu có, fallback MLP."""
        data     = raw_output.data
        primary  = data.get("primary", "mlp")
        key      = "lstm_drowsy_prob" if primary == "lstm" else "mlp_drowsy_prob"
        prob     = data.get(key, 0.0)
        return NormalizedLabel.DROWSY if prob >= DROWSY_THRESHOLD else NormalizedLabel.ALERT

    def estimate_confidence(
        self, raw_output: RawOutput, input_quality: InputQuality
    ) -> float:
        """
        LSTM: cao hơn (xét sequence temporal).
        MLP phụ: lower weight.
        Landmark ít bị ảnh hưởng ánh sáng → không penalty lighting.
        """
        if input_quality == InputQuality.MISSING:
            return 0.0

        data    = raw_output.data
        primary = data.get("primary", "mlp")

        if primary == "lstm":
            prob = data.get("lstm_drowsy_prob", 0.5)
            # LSTM được coi tin cậy hơn → weight cao hơn
            base_conf = abs(prob - 0.5) * 2.0
            return min(1.0, base_conf * 1.1)
        else:
            prob      = data.get("mlp_drowsy_prob", 0.5)
            base_conf = abs(prob - 0.5) * 2.0
            return min(1.0, base_conf * 0.85)   # MLP đơn frame kém hơn LSTM
