# -*- coding: utf-8 -*-
"""
Test API thật - gửi frame webcam hiện tại qua /api/analyze_lite nhiều lần
để xem model có cùng bias như test local không.

Vì không thể gửi ảnh mặt người thật qua script, ta test bằng cách:
1. Đo inference latency
2. Gửi frame đen/ngẫu nhiên để xem fallback behavior
3. Đo FPS ổn định không
"""
import os, sys, time, base64
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import cv2
import requests

API = "http://127.0.0.1:5000"
S = requests.Session()


# 1. Init
r = S.post(f"{API}/api/init").json()
print(f"[init] ok={r['ok']} load_mode={r['load_mode']} "
      f"rule_only={r['rule_only_mode']}\n")

# 2. Reset
S.post(f"{API}/api/reset")

# 3. Tạo 10 frame giả ngẫu nhiên (320x240) và gửi
print("=" * 60)
print("Latency test - 10 frames:")
print("=" * 60)
latencies = []
for i in range(10):
    img = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 75])
    data_url = "data:image/jpeg;base64," + base64.b64encode(buf).decode()
    t0 = time.time()
    r = S.post(f"{API}/api/analyze", json={"image": data_url})
    lat = (time.time() - t0) * 1000
    latencies.append(lat)
    d = r.json()
    print(f"  [{i}] {lat:5.1f}ms face={d['face_found']} "
          f"alarm={d['alarm_on']} state={d['drowsiness_state']} "
          f"ema={d['ema_prob']:.3f}")

print(f"\n  Mean latency: {np.mean(latencies):.1f}ms")
print(f"  Max latency:  {np.max(latencies):.1f}ms")
print(f"  Min latency:  {np.min(latencies):.1f}ms")
print(f"  → {'✅ Fast enough for live' if np.mean(latencies) < 200 else '⚠ Latency cao'}")

# 4. Check FPS
print()
print("=" * 60)
print("Server metrics:")
print("=" * 60)
m = S.get(f"{API}/api/metrics").json()
print(f"  CPU={m['cpu_percent']}%  RAM={m['ram_percent']}%  "
      f"inference_fps={m['inference_fps']}  uptime={m['uptime_sec']}s")
print(f"  Watchdog: armed={m['watchdog']['armed']} "
      f"last_inference_age={m['watchdog']['last_inference_age_sec']:.1f}s")
print(f"  → {'⚠ Watchdog armed (stale)' if m['watchdog']['armed'] else '✅ OK'}")
