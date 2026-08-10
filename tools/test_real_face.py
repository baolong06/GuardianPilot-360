# -*- coding: utf-8 -*-
"""
Tải ảnh mặt người thật từ Unsplash/Pexels CDN và test pipeline.
Mục đích: xác nhận MediaPipe detect được face sau khi fix num_threads bug.
"""
import os, sys, base64, time
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import requests
import numpy as np
import cv2

API = "http://127.0.0.1:5000"
S = requests.Session()


def analyze_bytes(img_bytes, label):
    data_url = "data:image/jpeg;base64," + base64.b64encode(img_bytes).decode()
    r = S.post(f"{API}/api/analyze_lite", json={"image": data_url})
    print(f"\n{'='*60}\n{label}\n{'='*60}")
    print(f"HTTP: {r.status_code}")
    if r.status_code != 200:
        print(f"Body: {r.text[:200]}")
        return
    d = r.json()
    print(f"ok={d['ok']} face_found={d['face_found']}")
    if d.get("face_found"):
        feat = d.get("features") or {}
        print(f"EAR_avg={feat.get('ear_avg')}  MAR={feat.get('mar')}")
        print(f"neck={feat.get('neck_tilt')}  pitch={feat.get('pitch')}  "
              f"yaw={feat.get('yaw')}  roll={feat.get('roll')}")
        print(f"p_mlp={d.get('p_mlp_drowsy')}  p_lstm={d.get('p_lstm_drowsy')}  "
              f"ema={d.get('ema_prob')}")
        print(f"alarm={d.get('alarm_on')}  state={d.get('drowsiness_state')}")
        print(f"neck_alarm={d.get('neck_alarm')}  eye_alarm={d.get('eye_alarm')}")
    else:
        print("  ⚠ MediaPipe không detect face")


# Try multiple sources for a face image
sources = [
    # WIDER FACE dataset sample (small face)
    ("https://raw.githubusercontent.com/wywu/ABN/master/data/example.png",
     "WIDER Face sample"),
    # Public test image
    ("https://thispersondoesnotexist.com/",
     "ThisPersonDoesNotExist"),
    # Pexels portrait
    ("https://images.pexels.com/photos/220453/pexels-photo-220453.jpeg?w=400",
     "Pexels portrait"),
    # Public face test (Wikimedia)
    ("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c9/Smiling_girl.jpg/320px-Smiling_girl.jpg",
     "Wikimedia Smiling girl"),
]

for url, label in sources:
    try:
        print(f"\n>>> Downloading: {label}")
        r = requests.get(url, timeout=10,
                         headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code != 200 or len(r.content) < 1000:
            print(f"   Skip (status={r.status_code}, size={len(r.content)})")
            continue
        print(f"   Got {len(r.content)} bytes")
        analyze_bytes(r.content, label)
    except Exception as e:
        print(f"   Error: {e}")
