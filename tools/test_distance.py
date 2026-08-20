"""Test pipeline debug output for face sizes at various distances."""
import sys
import cv2
import base64
import requests
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def img_to_b64(path, max_w=640):
    img = cv2.imread(path)
    if img is None:
        return None
    h, w = img.shape[:2]
    if w != max_w:
        scale = max_w / w
        img = cv2.resize(img, (max_w, int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


# Test với ảnh gốc ở các kích thước khác nhau
# M1: path tương đối so với repo thay vì ổ đĩa của một máy cá nhân.
src = str(ROOT / "data" / "_test_imgs" / "realistic_open.jpg")
img_full = cv2.imread(src)
print(f"Original: {img_full.shape}")

for scale, label in [(1.0, "100% (NEAR)"), (0.75, "75%"),
                       (0.5, "50% (DISTANT 2x)"), (0.35, "35% (FAR 3x)"),
                       (0.25, "25% (VERY FAR 4x)")]:
    h, w = img_full.shape[:2]
    new_w = int(w * scale)
    new_h = int(h * scale)
    small = cv2.resize(img_full, (new_w, new_h), interpolation=cv2.INTER_AREA)
    # Pad về 640x480
    canvas = cv2.resize(small, (640, 480), interpolation=cv2.INTER_LINEAR)
    ok, buf = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 90])
    data_url = "data:image/jpeg;base64," + base64.b64encode(buf).decode()

    r = requests.post(
        "http://127.0.0.1:5001/api/analyze_lite",
        json={"image": data_url},
        timeout=30,
    )
    j = r.json()
    print(f"\n[{label}] scale={scale:.2f}, canvas={canvas.shape}")
    print(f"  face_found={j.get('face_found')}, "
          f"alarm_on={j.get('alarm_on')}, "
          f"ema_prob={j.get('ema_prob')}")
    if j.get("features"):
        f = j["features"]
        print(f"  EAR={f.get('ear_avg')}, MAR={f.get('mar')}, "
              f"pitch={f.get('pitch')}, neck={f.get('neck_tilt')}")
    # face_landmarks count
    fl = j.get("face_landmarks")
    if fl:
        print(f"  face_landmarks: {len(fl)//2} points")
    else:
        print(f"  face_landmarks: None")
