# -*- coding: utf-8 -*-
"""
Diagnostic: kiểm tra API + phân tích logic FusionState qua unit-test trực tiếp.
"""
import sys, json, time, base64
import os
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
# Force UTF-8 stdout cho Windows
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import cv2
import requests

API = "http://127.0.0.1:5000"
SESSION = requests.Session()


def reset():
    SESSION.post(f"{API}/api/reset").raise_for_status()


def draw_frame(ear: float, mar: float = 0.3, neck_tilt: float = 0.0,
               pitch: float = 0.0, yaw: float = 0.0, roll: float = 0.0,
               h: int = 240, w: int = 320) -> str:
    """
    Vẽ frame giả lập với:
      - Face mesh 478 điểm (để MediaPipe detect được)
      - Mắt trái + phải có EAR như yêu cầu
      - Pose 33 điểm (nose, vai)
    Vì MediaPipe yêu cầu landmarks thật trong ảnh, cách này KHÔNG hoạt động
    trực tiếp với analyze_lite (nó vẫn chạy MediaPipe trên ảnh).

    Thay vào đó, ta chỉ test API status / fusion state qua các test khác.
    """
    img = np.full((h, w, 3), 64, dtype=np.uint8)
    # ... sẽ không dùng cách này
    return ""


def analyze_dummy(ear: float, mar: float = 0.3, neck_tilt: float = 0.0,
                  pitch: float = 0.0, yaw: float = 0.0, roll: float = 0.0):
    """
    Gọi /api/analyze với 1 frame bất kỳ để có timestamp, sau đó check
    response — không cách nào bypass MediaPipe. Trả về dict rỗng.

    Cách thực sự: dùng test trực tiếp FusionState (xem test_fusion_state.py).
    """
    img = np.zeros((240, 320, 3), dtype=np.uint8)
    cv2.putText(img, f"EAR={ear:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    data_url = "data:image/jpeg;base64," + base64.b64encode(buf).decode()
    r = SESSION.post(f"{API}/api/analyze", json={"image": data_url})
    return r.json()


def print_header(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def main():
    # ── 0. Init ──────────────────────────────────────────────────────────
    r = SESSION.post(f"{API}/api/init", json={}).json()
    print(f"[init] ok={r.get('ok')} load_mode={r.get('load_mode')} "
          f"rule_only_mode={r.get('rule_only_mode')}")
    if not r.get("ok"):
        sys.exit(1)

    # ── 1. Status check ────────────────────────────────────────────────
    print_header("1. SERVER STATUS")
    s = SESSION.get(f"{API}/api/status").json()
    print(json.dumps(s, indent=2, ensure_ascii=False))

    # ── 2. Analyze một frame trống (không có mặt) ──────────────────────
    print_header("2. FRAME KHÔNG CÓ MẶT")
    img = np.zeros((240, 320, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    data_url = "data:image/jpeg;base64," + base64.b64encode(buf).decode()
    r = SESSION.post(f"{API}/api/analyze",
                     json={"image": data_url}).json()
    print(f"  ok={r['ok']} face_found={r['face_found']} "
          f"alarm_on={r['alarm_on']} state={r['drowsiness_state']}")

    # ── 3. Gửi 5 frames liên tiếp với frame đen ─────────────────────────
    print_header("3. 5 FRAMES ĐEN LIÊN TIẾP")
    for i in range(5):
        r = SESSION.post(f"{API}/api/analyze",
                         json={"image": data_url}).json()
        print(f"  [{i}] face={r['face_found']} alarm={r['alarm_on']} "
              f"state={r['drowsiness_state']} lvl={r['alert_level']}")

    # ── 4. Metrics ─────────────────────────────────────────────────────
    print_header("4. METRICS")
    m = SESSION.get(f"{API}/api/metrics").json()
    print(f"  CPU={m['cpu_percent']}% RAM={m['ram_percent']}% "
          f"inference_fps={m['inference_fps']} uptime={m['uptime_sec']}s")
    print(f"  Watchdog: {m['watchdog']}")


if __name__ == "__main__":
    main()
