"""Unit tests cho metrics + watchdog."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.metrics import collect_metrics, InferenceWatchdog


def test_collect_metrics_shape():
    m = collect_metrics()
    assert "uptime_sec" in m
    assert m["uptime_sec"] >= 0
    # psutil có thể có hoặc không — không fail nếu thiếu
    print(f"PASS test_collect_metrics_shape: cpu={m.get('cpu_percent')}, ram={m.get('ram_percent')}")


def test_watchdog_heartbeat_and_stale():
    calls = []
    wd = InferenceWatchdog(
        stale_sec=0.2,
        check_interval_sec=0.05,
        on_stale=lambda: calls.append(time.time()),
    )
    wd.start()
    wd.heartbeat()
    time.sleep(0.15)
    assert wd.status()["armed"] is True
    assert len(calls) == 0  # chưa stale
    time.sleep(0.25)
    assert len(calls) >= 1
    wd.stop()
    print(f"PASS test_watchdog_heartbeat_and_stale: stale_calls={len(calls)}")


def test_watchdog_not_armed_before_heartbeat():
    calls = []
    wd = InferenceWatchdog(
        stale_sec=0.1,
        check_interval_sec=0.05,
        on_stale=lambda: calls.append(1),
    )
    wd.start()
    time.sleep(0.25)
    assert len(calls) == 0
    wd.stop()
    print("PASS test_watchdog_not_armed_before_heartbeat")


if __name__ == "__main__":
    test_collect_metrics_shape()
    test_watchdog_heartbeat_and_stale()
    test_watchdog_not_armed_before_heartbeat()
    print("\nAll metrics tests passed.")
