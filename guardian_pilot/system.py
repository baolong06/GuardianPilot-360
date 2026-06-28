"""
guardian_pilot/system.py
--------------------------
GuardianPilot360System — điểm khởi tạo duy nhất của toàn hệ thống.

Wires tất cả components lại với nhau:
  KnowledgeGraph ← shared bởi tất cả agent
  M1, M2, M3, M4 ← Perception Agents
  OrchestratorAgent ← đọc KG + áp luật
  ActuationLayer ← nhận callback từ Orchestrator
  FrameDispatcher ← điều phối frame → agents → orchestrator

Usage:
  system = GuardianPilot360System.from_model_dir("E:/Khởi nghiệp/Model")
  system.run_on_video("test_video.mp4")     # offline test
  system.run_on_camera(camera_index=0)      # real-time
"""

from __future__ import annotations

import logging
import os
import time

from .actuation  import ActuationLayer
from .agents.m1_drowsiness import M1DrowsinessAgent
from .agents.m2_microsleep import M2MicrosleepAgent
from .agents.m3_distracted import M3DistractedAgent
from .agents.m4_landmark   import M4LandmarkAgent
from .agents.orchestrator  import OrchestratorAgent
from .core.knowledge_graph import KnowledgeGraph
from .dispatcher           import FrameDispatcher

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("guardian_pilot.system")


class GuardianPilot360System:
    """Facade — khởi tạo và vận hành toàn bộ hệ thống GuardianPilot 360."""

    def __init__(
        self,
        kg:           KnowledgeGraph,
        dispatcher:   FrameDispatcher,
        actuation:    ActuationLayer,
    ) -> None:
        self.kg         = kg
        self.dispatcher = dispatcher
        self.actuation  = actuation
        self._running   = False

    # ─────────────────────────────────────────
    #  Factory
    # ─────────────────────────────────────────

    @classmethod
    def from_model_dir(
        cls,
        base_dir: str,
        sensor_eeg_available: bool = False,
        audit_log: str = "guardian_pilot_audit.log",
    ) -> "GuardianPilot360System":
        """
        Tạo hệ thống từ thư mục chứa model files.
        Đường dẫn tương đối so với base_dir.
        """
        def p(*parts: str) -> str:
            return os.path.join(base_dir, *parts)

        # ── Knowledge Graph ──────────────────
        kg = KnowledgeGraph()

        # ── Actuation Layer ──────────────────
        actuation = ActuationLayer(kg, audit_log_path=audit_log)

        # ── Perception Agents ────────────────
        m1 = M1DrowsinessAgent(
            kg         = kg,
            model_path = p("task_1", "dcnn_drowsiness_task1_baseline.keras"),
        )
        m2 = M2MicrosleepAgent(
            kg               = kg,
            model_path       = p("Task 2 — Reproduce Automatic Detection of Microsl",
                                  "cnn_16s_best.keras"),
            sensor_available = sensor_eeg_available,
        )
        m3 = M3DistractedAgent(
            kg            = kg,
            dbmnet_path   = p("Task_3", "dbmnet_full_task3.keras"),
            baseline_path = p("Task_3", "baseline_ghostnetlike_task3.keras"),
        )
        m4 = M4LandmarkAgent(
            kg               = kg,
            lstm_path        = p("Task_4", "lstm_landmark_task4_fixed.keras"),
            mlp_path         = p("Task_4", "mlp_landmark_task4_fixed.keras"),
            scaler_path      = p("Task_4", "landmark_scaler_task4.pkl"),
            landmarker_path  = p("Task_4", "face_landmarker.task"),
        )

        # ── Orchestrator ─────────────────────
        orchestrator = OrchestratorAgent(
            kg                  = kg,
            actuation_callback  = actuation.on_state_change,
        )

        # ── Frame Dispatcher ─────────────────
        dispatcher = FrameDispatcher(
            kg           = kg,
            m1           = m1,
            m2           = m2,
            m3           = m3,
            m4           = m4,
            orchestrator = orchestrator,
        )

        logger.info("✓ GuardianPilot 360 khởi tạo thành công.")
        logger.info("  Sensor EEG/EOG: %s", "CÓ" if sensor_eeg_available else "KHÔNG")

        return cls(kg=kg, dispatcher=dispatcher, actuation=actuation)

    # ─────────────────────────────────────────
    #  Run modes
    # ─────────────────────────────────────────

    def run_on_video(
        self,
        video_path: str,
        target_fps: float = 15.0,
        display: bool = True,
    ) -> None:  # noqa: C901
        """
        Chạy hệ thống trên file video (offline test — Giai đoạn 1).
        target_fps: số frame/giây xử lý (không cần xử lý mọi frame).
        """
        import cv2  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Không mở được video: {video_path}")

        original_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_skip   = max(1, int(original_fps / target_fps))
        frame_idx    = 0
        self._running = True

        logger.info("▶  Chạy trên video: %s (%.0f fps → xử lý %.0f fps)",
                    video_path, original_fps, target_fps)

        while self._running:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            if frame_idx % frame_skip != 0:
                continue

            snapshot = self.dispatcher.dispatch_frame(frame)

            if display:
                self._overlay_status(frame, snapshot)
                cv2.imshow("GuardianPilot 360", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        cap.release()
        cv2.destroyAllWindows()
        self.dispatcher.shutdown()
        logger.info("■  Video processing hoàn tất.")
        self._print_summary()

    def run_on_camera(
        self,
        camera_index: int = 0,
        target_fps: float = 15.0,
        display: bool = True,
    ) -> None:
        """
        Chạy real-time trên camera (Giai đoạn 2).
        """
        import cv2  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415
        cap = cv2.VideoCapture(camera_index)
        cap.set(cv2.CAP_PROP_FPS, 30)
        if not cap.isOpened():
            raise ValueError(f"Không mở được camera index={camera_index}")

        tick_interval = 1.0 / target_fps
        self._running  = True
        logger.info("▶  Real-time camera mode (fps=%.0f). Nhấn Q để dừng.", target_fps)

        while self._running:
            t0 = time.perf_counter()
            ret, frame = cap.read()
            if not ret:
                logger.warning("Camera frame drop.")
                continue

            snapshot = self.dispatcher.dispatch_frame(frame)

            if display:
                self._overlay_status(frame, snapshot)
                cv2.imshow("GuardianPilot 360 — Live", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            # Rate limiting — không vượt quá target_fps
            elapsed = time.perf_counter() - t0
            sleep_t = tick_interval - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

        cap.release()
        cv2.destroyAllWindows()
        self.dispatcher.shutdown()
        logger.info("■  Camera mode dừng.")

    def stop(self) -> None:
        self._running = False

    # ─────────────────────────────────────────
    #  Display overlay
    # ─────────────────────────────────────────

    @staticmethod
    def _overlay_status(frame, snapshot: dict) -> None:
        """Vẽ status overlay lên frame (cho display mode)."""
        import cv2  # noqa: PLC0415
        level   = snapshot.get("alert_level", "NORMAL")
        conf    = snapshot.get("confidence", 0.0)
        reason  = snapshot.get("alert_reason", "")[:60]

        color_map = {
            "NORMAL":         (0, 200, 0),
            "MILD_WARNING":   (0, 200, 255),
            "SEVERE_WARNING": (0, 80, 255),
            "EMERGENCY":      (0, 0, 255),
        }
        color = color_map.get(level, (200, 200, 200))

        # Background rect
        cv2.rectangle(frame, (0, 0), (640, 70), (20, 20, 20), -1)
        cv2.putText(frame, f"{level} (conf={conf:.2f})",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, reason,
                    (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    # ─────────────────────────────────────────
    #  Summary
    # ─────────────────────────────────────────

    def _print_summary(self) -> None:
        snapshot = self.kg.snapshot()
        conflicts = len(self.kg.get_conflict_log())
        actuations = len(self.kg.get_actuation_log())
        print("\n" + "="*60)
        print("  GUARDIAN PILOT 360 — SESSION SUMMARY")
        print("="*60)
        print(f"  Final alert level : {snapshot['alert_level']}")
        print(f"  Final confidence  : {snapshot['confidence']}")
        print(f"  Alive agents      : {snapshot['alive_agents']}")
        print(f"  Total conflicts   : {conflicts}")
        print(f"  Total actuations  : {actuations}")
        print("="*60)
