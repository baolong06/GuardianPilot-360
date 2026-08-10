"""
System metrics + lightweight watchdog (SYS-04 / SYS-05).

- Metrics: CPU%, RAM%, optional GPU%, process RSS, uptime
- Watchdog: theo dõi thời gian kể từ lần inference gần nhất;
  nếu > stale_sec → log warning và gọi callback reload (nếu có)
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Optional

logger = logging.getLogger(__name__)

_PROCESS_START = time.time()


def collect_metrics() -> dict:
    """Thu thập CPU/RAM/(GPU nếu có)/uptime. Không raise nếu thiếu psutil."""
    metrics: dict = {
        "uptime_sec": round(time.time() - _PROCESS_START, 1),
        "cpu_percent": None,
        "ram_percent": None,
        "ram_used_mb": None,
        "process_rss_mb": None,
        "gpu_percent": None,
        "gpu_mem_percent": None,
        "temp_c": None,
    }
    try:
        import psutil

        metrics["cpu_percent"] = psutil.cpu_percent(interval=0.0)
        vm = psutil.virtual_memory()
        metrics["ram_percent"] = vm.percent
        metrics["ram_used_mb"] = round(vm.used / (1024 * 1024), 1)
        proc = psutil.Process()
        metrics["process_rss_mb"] = round(proc.memory_info().rss / (1024 * 1024), 1)

        # Nhiệt độ (không phải máy nào cũng có)
        try:
            temps = psutil.sensors_temperatures()
            if temps:
                # lấy sensor đầu tiên có reading
                for entries in temps.values():
                    if entries:
                        metrics["temp_c"] = entries[0].current
                        break
        except (AttributeError, OSError):
            pass
    except ImportError:
        metrics["note"] = "psutil not installed — install for SYS-04 metrics"

    # GPU optional (nvidia-smi via pynvml nếu có)
    try:
        import pynvml  # type: ignore

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        metrics["gpu_percent"] = util.gpu
        metrics["gpu_mem_percent"] = round(100.0 * mem.used / mem.total, 1)
        pynvml.nvmlShutdown()
    except Exception:
        pass

    return metrics


class InferenceWatchdog:
    """
    Thread giám sát: nếu không có kết quả inference mới trong stale_sec giây
    → warning + optional reload callback.
    """

    def __init__(
        self,
        stale_sec: float = 5.0,
        check_interval_sec: float = 1.0,
        on_stale: Optional[Callable[[], None]] = None,
    ):
        self.stale_sec = stale_sec
        self.check_interval_sec = check_interval_sec
        self.on_stale = on_stale
        self._last_inference_ts = time.time()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._stale_count = 0
        self._armed = False  # chỉ giám sát sau lần inference đầu

    def heartbeat(self):
        """Gọi sau mỗi lần analyze thành công."""
        with self._lock:
            self._last_inference_ts = time.time()
            self._armed = True

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name="inference-watchdog", daemon=True
        )
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

    def status(self) -> dict:
        with self._lock:
            age = time.time() - self._last_inference_ts
            return {
                "armed": self._armed,
                "last_inference_age_sec": round(age, 2),
                "stale_sec": self.stale_sec,
                "stale_count": self._stale_count,
                "running": bool(self._thread and self._thread.is_alive()),
            }

    def _loop(self):
        while not self._stop.wait(self.check_interval_sec):
            with self._lock:
                armed = self._armed
                age = time.time() - self._last_inference_ts
            if not armed:
                continue
            if age > self.stale_sec:
                self._stale_count += 1
                logger.warning(
                    "Watchdog: no inference for %.1fs (threshold %.1fs) — stale #%d",
                    age,
                    self.stale_sec,
                    self._stale_count,
                )
                if self.on_stale is not None:
                    try:
                        self.on_stale()
                    except Exception:
                        logger.exception("Watchdog on_stale callback failed")
