#!/usr/bin/env python3
"""
Mock CAN bus speed simulator — prints / optionally POSTs speed samples.

Usage:
  python tools/can_sim.py                  # print loop
  python tools/can_sim.py --post http://127.0.0.1:5000/api/vehicle
  python tools/can_sim.py --profile highway
"""
from __future__ import annotations

import argparse
import json
import math
import time
import urllib.request


def speed_profile(t: float, profile: str) -> float:
    if profile == "idle":
        return 0.0
    if profile == "city":
        return max(0.0, 35 + 15 * math.sin(t / 8.0))
    if profile == "highway":
        return max(0.0, 95 + 8 * math.sin(t / 20.0))
    # mixed
    cycle = t % 120
    if cycle < 20:
        return 0.0
    if cycle < 60:
        return 40 + 10 * math.sin(cycle / 5.0)
    return 90 + 5 * math.sin(cycle / 10.0)


def post_speed(url: str, speed: float):
    data = json.dumps({"speed_kmh": round(speed, 1)}).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=2) as resp:
        return resp.read()


def main():
    p = argparse.ArgumentParser(description="Mock CAN speed simulator")
    p.add_argument("--profile", default="mixed", choices=["idle", "city", "highway", "mixed"])
    p.add_argument("--hz", type=float, default=2.0)
    p.add_argument("--post", default=None, help="POST to /api/vehicle")
    p.add_argument("--duration", type=float, default=0, help="0 = forever")
    args = p.parse_args()

    t0 = time.time()
    interval = 1.0 / max(args.hz, 0.1)
    while True:
        t = time.time() - t0
        if args.duration > 0 and t >= args.duration:
            break
        spd = speed_profile(t, args.profile)
        print(f"[can_sim] t={t:6.1f}s  speed={spd:5.1f} km/h")
        if args.post:
            try:
                post_speed(args.post, spd)
            except Exception as exc:
                print(f"  post failed: {exc}")
        time.sleep(interval)


if __name__ == "__main__":
    main()
