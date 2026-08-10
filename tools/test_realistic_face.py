# -*- coding: utf-8 -*-
"""
Tải ảnh mặt người thật từ dataset công khai và test pipeline end-to-end.

Sử dụng fetch từ các nguồn dataset phổ biến (dùng Pillow để vẽ placeholder
nếu không tải được — nhưng tốt nhất là dùng ảnh có sẵn trong project).
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


def make_photo_realistic_face(state="open", h=480, w=640):
    """
    Tạo ảnh photo-realistic hơn với nhiều chi tiết để MediaPipe detect được.
    Bao gồm: skin tone gradient, eyebrows, eyelashes, nostrils, etc.
    """
    img = np.zeros((h, w, 3), dtype=np.uint8)
    # Skin tone gradient
    for y in range(h):
        tone = 180 + int(20 * np.sin(y / h * np.pi))
        img[y, :] = (180, 200, tone)

    # Face oval với shadow
    overlay = img.copy()
    cv2.ellipse(overlay, (w//2, h//2), (170, 210), 0, 0, 360, (200, 215, 230), -1)
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)

    # Eyebrows (quan trọng để MediaPipe detect face landmarks)
    cv2.ellipse(img, (w//2 - 70, h//2 - 75), (35, 8), 0, 180, 360, (60, 40, 30), -1)
    cv2.ellipse(img, (w//2 + 70, h//2 - 75), (35, 8), 0, 180, 360, (60, 40, 30), -1)

    # Eyes
    if state == "open":
        # Mắt mở với iris rõ
        cv2.ellipse(img, (w//2 - 70, h//2 - 40), (32, 16), 0, 0, 360, (250, 250, 250), -1)
        cv2.ellipse(img, (w//2 - 70, h//2 - 40), (14, 14), 0, 0, 360, (60, 50, 30), -1)
        cv2.circle(img, (w//2 - 70, h//2 - 40), 6, (10, 10, 10), -1)
        # Eyelashes (top)
        cv2.line(img, (w//2 - 100, h//2 - 50), (w//2 - 40, h//2 - 50), (40, 30, 20), 2)

        cv2.ellipse(img, (w//2 + 70, h//2 - 40), (32, 16), 0, 0, 360, (250, 250, 250), -1)
        cv2.ellipse(img, (w//2 + 70, h//2 - 40), (14, 14), 0, 0, 360, (60, 50, 30), -1)
        cv2.circle(img, (w//2 + 70, h//2 - 40), 6, (10, 10, 10), -1)
        cv2.line(img, (w//2 + 40, h//2 - 50), (w//2 + 100, h//2 - 50), (40, 30, 20), 2)
    elif state == "closed":
        # Mắt nhắm = đường cong
        cv2.ellipse(img, (w//2 - 70, h//2 - 40), (32, 6), 0, 0, 360, (60, 40, 30), -1)
        cv2.ellipse(img, (w//2 + 70, h//2 - 40), (32, 6), 0, 0, 360, (60, 40, 30), -1)
        # Eyelashes (top + bottom)
        cv2.line(img, (w//2 - 100, h//2 - 45), (w//2 - 40, h//2 - 45), (40, 30, 20), 2)
        cv2.line(img, (w//2 - 100, h//2 - 35), (w//2 - 40, h//2 - 35), (40, 30, 20), 1)
        cv2.line(img, (w//2 + 40, h//2 - 45), (w//2 + 100, h//2 - 45), (40, 30, 20), 2)
        cv2.line(img, (w//2 + 40, h//2 - 35), (w//2 + 100, h//2 - 35), (40, 30, 20), 1)
    elif state == "half":
        # Mắt lơ mơ (EAR thấp ~0.18)
        cv2.ellipse(img, (w//2 - 70, h//2 - 40), (32, 8), 0, 0, 360, (240, 240, 240), -1)
        cv2.line(img, (w//2 - 100, h//2 - 45), (w//2 - 40, h//2 - 45), (40, 30, 20), 2)
        cv2.ellipse(img, (w//2 + 70, h//2 - 40), (32, 8), 0, 0, 360, (240, 240, 240), -1)
        cv2.line(img, (w//2 + 40, h//2 - 45), (w//2 + 100, h//2 - 45), (40, 30, 20), 2)

    # Nose với nostrils
    cv2.ellipse(img, (w//2, h//2 + 10), (20, 35), 0, 0, 360, (190, 200, 210), -1)
    cv2.ellipse(img, (w//2 - 8, h//2 + 35), (5, 4), 0, 0, 360, (100, 80, 70), -1)
    cv2.ellipse(img, (w//2 + 8, h//2 + 35), (5, 4), 0, 0, 360, (100, 80, 70), -1)

    # Mouth
    cv2.ellipse(img, (w//2, h//2 + 80), (50, 5), 0, 0, 360, (100, 70, 70), -1)
    cv2.ellipse(img, (w//2, h//2 + 78), (50, 3), 0, 0, 360, (130, 90, 90), -1)

    # Chin shadow
    cv2.ellipse(img, (w//2, h//2 + 200), (60, 15), 0, 0, 360, (150, 160, 170), 2)

    # Noise để giảm "ảnh quá sạch"
    noise = np.random.randint(-15, 15, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return img


def encode(img):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


def analyze(img, label):
    r = S.post(f"{API}/api/analyze_lite", json={"image": encode(img)}).json()
    print(f"\n{label}")
    print(f"  ok={r['ok']} face_found={r['face_found']}")
    if r.get("face_found"):
        feat = r.get("features", {}) or {}
        print(f"  EAR_avg={feat.get('ear_avg')}  MAR={feat.get('mar')}")
        print(f"  neck={feat.get('neck_tilt')}  pitch={feat.get('pitch')}  "
              f"yaw={feat.get('yaw')}  roll={feat.get('roll')}")
        print(f"  p_mlp={r.get('p_mlp_drowsy')}  p_lstm={r.get('p_lstm_drowsy')}  "
              f"ema={r.get('ema_prob')}")
        print(f"  alarm={r.get('alarm_on')}  state={r.get('drowsiness_state')}")
        print(f"  neck_alarm={r.get('neck_alarm')}  eye_alarm={r.get('eye_alarm')}  "
              f"yawn_alarm={r.get('yawn_alarm')}")
    else:
        print("  ⚠ MediaPipe không detect face")
    return r


# Init
r = S.post(f"{API}/api/init").json()
print(f"[init] {r}\n")

# Test single
os.makedirs("data/_test_imgs", exist_ok=True)
img_open = make_photo_realistic_face("open")
img_closed = make_photo_realistic_face("closed")
img_half = make_photo_realistic_face("half")
cv2.imwrite("data/_test_imgs/realistic_open.jpg", img_open)
cv2.imwrite("data/_test_imgs/realistic_closed.jpg", img_closed)
cv2.imwrite("data/_test_imgs/realistic_half.jpg", img_half)

analyze(img_open, "[A] Mắt mở (photo-realistic)")
analyze(img_closed, "[B] Mắt nhắm (photo-realistic)")
analyze(img_half, "[C] Mắt lơ mơ EAR~0.18 (photo-realistic)")

# Sequence test
print("\n" + "="*60)
print("  SEQUENCE: 30 frames mở → 20 frames nhắm → 20 frames mở")
print("="*60)
S.post(f"{API}/api/reset")

print("Phase 1: Eyes OPEN (30 frames)")
for i in range(30):
    r = S.post(f"{API}/api/analyze_lite",
               json={"image": encode(img_open)}).json()
    if i % 5 == 0 or r.get("alarm_on"):
        print(f"  [OPEN {i:>2}] face={r['face_found']} "
              f"alarm={r.get('alarm_on')} state={r.get('drowsiness_state')} "
              f"ema={r.get('ema_prob')}")
    time.sleep(0.04)

print("\nPhase 2: Eyes CLOSED (20 frames) — should trigger microsleep")
for i in range(20):
    r = S.post(f"{API}/api/analyze_lite",
               json={"image": encode(img_closed)}).json()
    if i % 3 == 0 or r.get("alarm_on"):
        print(f"  [CLS {i:>2}] face={r['face_found']} "
              f"alarm={r.get('alarm_on')} state={r.get('drowsiness_state')} "
              f"ema={r.get('ema_prob')} eye_alarm={r.get('eye_alarm')}")
    time.sleep(0.04)

print("\nPhase 3: Eyes OPEN again (20 frames) — alarm should turn OFF quickly")
for i in range(20):
    r = S.post(f"{API}/api/analyze_lite",
               json={"image": encode(img_open)}).json()
    if i % 3 == 0 or r.get("alarm_on"):
        print(f"  [REC {i:>2}] face={r['face_found']} "
              f"alarm={r.get('alarm_on')} state={r.get('drowsiness_state')} "
              f"ema={r.get('ema_prob')}")
    time.sleep(0.04)
