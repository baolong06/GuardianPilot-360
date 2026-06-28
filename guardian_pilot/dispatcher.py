"""
guardian_pilot/dispatcher.py
-------------------------------
Frame Dispatcher — lõi xử lý phân phối frame tới các agent.

Nhiệm vụ (theo kiến trúc mục 3):
  - Nhận 1 frame từ camera
  - Tạo 3 bản preprocessing song song cho M1 / M3 / M4
    (M2 xử lý tín hiệu EEG riêng, không từ camera)
  - Chạy M1 / M3 / M4 song song bằng ThreadPoolExecutor
  - Trigger Orchestrator sau khi ít nhất M1+M4 đã xong
  - Đảm bảo: agent chậm nhất không block toàn hệ thống

Latency mục tiêu: <150ms/tick (mục 6 kiến trúc)
"""

from __future__ import annotations

import concurrent.futures
import logging
import time
from typing import Callable, List, Optional

from .agents.m1_drowsiness import M1DrowsinessAgent
from .agents.m2_microsleep import M2MicrosleepAgent
from .agents.m3_distracted import M3DistractedAgent
from .agents.m4_landmark   import M4LandmarkAgent
from .agents.orchestrator  import OrchestratorAgent
from .core.knowledge_graph import KnowledgeGraph
from .core.schema          import AgentID

logger = logging.getLogger("guardian_pilot.dispatcher")


class FrameDispatcher:
    """
    Điều phối frame từ camera tới các Perception Agent và Orchestrator.
    Dùng ThreadPoolExecutor để chạy M1/M3/M4 song song.
    """

    def __init__(
        self,
        kg: KnowledgeGraph,
        m1: M1DrowsinessAgent,
        m2: Optional[M2MicrosleepAgent],
        m3: M3DistractedAgent,
        m4: M4LandmarkAgent,
        orchestrator: OrchestratorAgent,
        max_workers: int = 3,
        tick_timeout_ms: float = 140.0,   # budget cho M1+M3+M4
    ) -> None:
        self.kg           = kg
        self.m1           = m1
        self.m2           = m2
        self.m3           = m3
        self.m4           = m4
        self.orchestrator = orchestrator
        self.tick_timeout = tick_timeout_ms / 1000.0

        # ThreadPool cho inference song song
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="gp_agent",
        )

        self._tick_count = 0

    # ─────────────────────────────────────────
    #  Public: xử lý 1 frame camera
    # ─────────────────────────────────────────

    def dispatch_frame(self, camera_frame) -> dict:
        """
        Điểm vào chính — gọi mỗi khi có frame mới từ camera.
        camera_frame: np.ndarray BGR (OpenCV format)
        Trả về snapshot KG sau khi Orchestrator xử lý.
        """
        t_start = time.perf_counter()
        self._tick_count += 1

        # Submit M1, M3, M4 song song
        futures = {
            "M1": self._executor.submit(self.m1.run, camera_frame),
            "M3": self._executor.submit(self.m3.run, camera_frame),
            "M4": self._executor.submit(self.m4.run, camera_frame),
        }

        # Chờ M1 và M4 (2 agent quan trọng nhất cho RULE 2) trong budget time
        self._wait_priority_agents(
            futures={"M1": futures["M1"], "M4": futures["M4"]},
            timeout=self.tick_timeout,
        )

        # M3 có thể chậm hơn (DBMNet nặng) — cho thêm thời gian nhỏ
        self._wait_with_timeout(futures["M3"], timeout=0.020)  # +20ms

        # Trigger Orchestrator với dữ liệu mới nhất có sẵn
        # (không chờ agent nào chậm trễ — dùng timestamp KG để biết data tươi không)
        state = self.orchestrator.tick()

        elapsed_ms = (time.perf_counter() - t_start) * 1000
        logger.debug(
            "Tick #%d | %.1fms | %s (conf=%.2f)",
            self._tick_count, elapsed_ms,
            state.current_alert_level.value, state.confidence,
        )

        if elapsed_ms > 150:
            logger.warning("⚠  Tick #%d quá ngưỡng 150ms: %.1fms", self._tick_count, elapsed_ms)

        return self.kg.snapshot()

    # ─────────────────────────────────────────
    #  Public: feed EEG/EOG signal (M2)
    # ─────────────────────────────────────────

    def dispatch_eeg_signal(self, eeg_signal) -> None:
        """
        Xử lý tín hiệu EEG/EOG cho M2.
        Chạy trên thread riêng, hoàn toàn độc lập với camera tick.
        """
        if self.m2 is not None and self.kg.get_health(AgentID.M2_MICROSLEEP).is_alive:
            self._executor.submit(self.m2.process_signal, eeg_signal)

    # ─────────────────────────────────────────
    #  Helpers
    # ─────────────────────────────────────────

    def _wait_priority_agents(self, futures: dict, timeout: float) -> None:
        """Chờ các agent ưu tiên trong thời gian cho phép."""
        done, not_done = concurrent.futures.wait(
            futures.values(),
            timeout=timeout,
        )
        if not_done:
            agent_names = [k for k, f in futures.items() if f in not_done]
            logger.warning(
                "Agent %s chưa xong trong %.0fms — Orchestrator chạy với data cũ nhất.",
                agent_names, timeout * 1000,
            )

    @staticmethod
    def _wait_with_timeout(future, timeout: float) -> None:
        try:
            future.result(timeout=timeout)
        except (concurrent.futures.TimeoutError, Exception):
            pass

    # ─────────────────────────────────────────
    #  Cleanup
    # ─────────────────────────────────────────

    def shutdown(self) -> None:
        """Dừng ThreadPool an toàn."""
        self._executor.shutdown(wait=False, cancel_futures=True)
        logger.info("FrameDispatcher shutdown.")
