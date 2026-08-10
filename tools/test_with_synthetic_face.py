# -*- coding: utf-8 -*-
"""
Tạo ảnh test có "khuôn mặt giả lập" + test pipeline end-to-end qua API.

Vì không thể lấy webcam thật của user trong Playwright browser, ta:
1. Tạo 3 ảnh test (eyes open, eyes closed, head nod)
2. Gửi qua /api/analyze_lite để xác nhận:
   - MediaPipe có detect được face không
   - EAR/MAR/pose values hợp lý không
   - Alarm/state phản ứng đúng không
3. Đây là cách duy nhất test được mà không cần webcam thật
"""
import os, sys, base64, time
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


def make_eyes_open_face(h=480, w=640):
    """Tạo ảnh với 1 oval đại diện cho mặt + 2 mắt mở (vẽ bằng ellipse đen)."""
    img = np.full((h, w, 3), 200, dtype=np.uint8)  # nền xám sáng
    # Mặt oval
    cv2.ellipse(img, (w//2, h//2), (180, 220), 0, 0, 360, (180, 200, 220), -1)
    # Mắt trái (open) - ellipse rộng
    cv2.ellipse(img, (w//2 - 70, h//2 - 40), (35, 18), 0, 0, 360, (255, 255, 255), -1)
    cv2.ellipse(img, (w//2 - 70, h//2 - 40), (15, 15), 0, 0, 360, (50, 50, 50), -1)
    # Mắt phải (open)
    cv2.ellipse(img, (w//2 + 70, h//2 - 40), (35, 18), 0, 0, 360, (255, 255, 255), -1)
    cv2.ellipse(img, (w//2 + 70, h//2 - 40), (15, 15), 0, 0, 360, (50, 50, 50), -1)
    # Miệng
    cv2.ellipse(img, (w//2, h//2 + 80), (60, 8), 0, 0, 360, (80, 80, 80), 2)
    # Mũi
    cv2.circle(img, (w//2, h//2 + 20), 8, (100, 100, 100), -1)
    return img


def make_eyes_closed_face(h=480, w=640):
    """Mặt với mắt nhắm (EAR thấp)."""
    img = np.full((h, w, 3), 200, dtype=np.uint8)
    cv2.ellipse(img, (w//2, h//2), (180, 220), 0, 0, 360, (180, 200, 220), -1)
    # Mắt nhắm = đường ngang
    cv2.line(img, (w//2 - 105, h//2 - 40), (w//2 - 35, h//2 - 40), (50, 50, 50), 3)
    cv2.line(img, (w//2 + 35, h//2 - 40), (w//2 + 105, h//2 - 40), (50, 50, 50), 3)
    cv2.ellipse(img, (w//2, h//2 + 80), (60, 8), 0, 0, 360, (80, 80, 80), 2)
    cv2.circle(img, (w//2, h//2 + 20), 8, (100, 100, 100), -1)
    return img


def make_head_nod_face(h=480, w=640, nod=0):
    """Mặt với đầu nghiêng xuống (head nod)."""
    img = np.full((h, w, 3), 200, dtype=np.uint8)
    # Mặt dịch xuống + thu nhỏ (giả lập cú gật)
    cy = h//2 + nod
    rx, ry = max(80, 180 - nod), max(80, 220 - nod*2)
    cv2.ellipse(img, (w//2, cy), (rx, ry), 0, 0, 360, (180, 200, 220), -1)
    cv2.ellipse(img, (w//2 - 70, cy - 40), (35, 18), 0, 0, 360, (255, 255, 255), -1)
    cv2.ellipse(img, (w//2 - 70, cy - 40), (15, 15), 0, 0, 360, (50, 50, 50), -1)
    cv2.ellipse(img, (w//2 + 70, cy - 40), (35, 18), 0, 0, 360, (255, 255, 255), -1)
    cv2.ellipse(img, (w//2 + 70, cy - 40), (15, 15), 0, 0, 360, (50, 50, 50), -1)
    cv2.ellipse(img, (w//2, cy + 80), (60, 8), 0, 0, 360, (80, 80, 80), 2)
    cv2.circle(img, (w//2, cy + 20), 8, (100, 100, 100), -1)
    return img


def encode(img):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


def send(path, label):
    with open(path, "rb") as f:
        data = f.read()
    data_url = "data:image/jpeg;base64," + base64.b64encode(data).decode()
    r = S.post(f"{API}/api/analyze_lite", json={"image": data_url}).json()
    print(f"\n{'='*60}\n  {label}\n{'='*60}")
    print(f"  ok={r['ok']} face_found={r['face_found']}")
    if r.get("face_found"):
        feat = r.get("features", {}) or {}
        print(f"  EAR_avg={feat.get('ear_avg')}  MAR={feat.get('mar')}  "
              f"neck={feat.get('neck_tilt')}  pitch={feat.get('pitch')}  "
              f"yaw={feat.get('yaw')}  roll={feat.get('roll')}")
        print(f"  p_mlp={r.get('p_mlp_drowsy')}  p_lstm={r.get('p_lstm_drowsy')}  "
              f"ema={r.get('ema_prob')}  alarm={r.get('alarm_on')}  "
              f"state={r.get('drowsiness_state')}")
        print(f"  neck_alarm={r.get('neck_alarm')}  eye_alarm={r.get('eye_alarm')}  "
              f"yawn_alarm={r.get('yawn_alarm')}")
        print(f"  ear_smooth={r.get('ear_smooth')}  eyes_open={r.get('eyes_open_streak_ms')}ms  "
              f"eyes_close={r.get('eye_closed_streak_ms')}ms")
    else:
        print(f"  ⚠ MediaPipe KHÔNG detect được mặt từ ảnh giả lập")
        print(f"  → Đây là behavior bình thường: ảnh vẽ tay quá đơn giản,")
        print(f"    MediaPipe cần ảnh người thật mới detect được")


# 1. Init + reset
r = S.post(f"{API}/api/init").json()
print(f"[init] {r}")
S.post(f"{API}/api/reset")

# 2. Tạo ảnh test
os.makedirs("data/_test_imgs", exist_ok=True)
cv2.imwrite("data/_test_imgs/eyes_open.jpg", make_eyes_open_face())
cv2.imwrite("data/_test_imgs/eyes_closed.jpg", make_eyes_closed_face())
for i, nod in enumerate([0, 30, 60, 90, 120]):
    cv2.imwrite(f"data/_test_imgs/head_nod_{i}_{nod}.jpg", make_head_nod_face(nod=nod))

# 3. Send
send("data/_test_imgs/eyes_open.jpg", "TEST A: Mắt mở (ảnh vẽ tay)")
send("data/_test_imgs/eyes_closed.jpg", "TEST B: Mắt nhắm (ảnh vẽ tay)")
send("data/_test_imgs/head_nod_4_120.jpg", "TEST C: Head nod mạnh (ảnh vẽ tay)")

# 4. Send 1 sequence để test EMA + hysteresis
print("\n" + "="*60)
print("  SEQUENCE TEST: 20 frames mắt mở → 15 frames nhắm → 10 frames m�")
print("="*60)
S.post(f"{API}/api/reset")
import time as _t
for i in range(20):
    img = make_eyes_open_face()
    r = S.post(f"{API}/api/analyze_lite", json={"image": encode(img)}).json()
    if i % 5 == 0:
        print(f"  [open {i:>2}] face={r['face_found']} "
              f"alarm={r.get('alarm_on')} state={r.get('drowsiness_state')} "
              f"ema={r.get('ema_prob')}")
    _t.sleep(0.05)
for i in range(15):
    img = make_eyes_closed_face()
    r = S.post(f"{API}/api/analyze_lite", json={"image": encode(img)}).json()
    if i % 3 == 0:
        print(f"  [CLOSE{i:>2}] face={r['face_found']} "
              f"alarm={r.get('alarm_on')} state={r.get('drowsiness_state')} "
              f"ema={r.get('ema_prob')} eye_alarm={r.get('eye_alarm')}")
    _t.sleep(0.05)
for i in range(10):
    img = make_eyes_open_face()
    r = S.post(f"{API}/api/analyze_lite", json={"image": encode(img)}).json()
    if i % 2 == 0:
        print(f"  [OPEN {i:>2}] face={r['face_found']} "
              f"alarm={r.get('alarm_on')} state={r.get('drowsiness_state')} "
              f"ema={r.get('ema_prob')}")
    _t.sleep(0.05)
