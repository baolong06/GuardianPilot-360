# -*- coding: utf-8 -*-
"""
Test với ảnh mặt người thật từ các nguồn đáng tin cậy.
"""
import os, sys, base64
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
        print(f"Body: {r.text[:300]}")
        return None
    d = r.json()
    print(f"ok={d['ok']} face_found={d['face_found']}")
    if d.get("face_found"):
        feat = d.get("features") or {}
        print(f"✅ DETECTED!")
        print(f"  EAR_avg={feat.get('ear_avg')}  MAR={feat.get('mar')}")
        print(f"  neck={feat.get('neck_tilt')}  pitch={feat.get('pitch')}  "
              f"yaw={feat.get('yaw')}  roll={feat.get('roll')}")
        print(f"  p_mlp={d.get('p_mlp_drowsy')}  p_lstm={d.get('p_lstm_drowsy')}  "
              f"ema={d.get('ema_prob')}")
        print(f"  alarm={d.get('alarm_on')}  state={d.get('drowsiness_state')}")
    else:
        print(f"  ⚠ MediaPipe không detect face (nhưng server không crash)")
    return d


# Nguồn ảnh face công khai
sources = [
    # Pexels portraits (CC0)
    ("https://images.pexels.com/photos/2379004/pexels-photo-2379004.jpeg?w=640",
     "Pexels face 1"),
    ("https://images.pexels.com/photos/1222271/pexels-photo-1222271.jpeg?w=640",
     "Pexels face 2"),
    ("https://images.pexels.com/photos/614810/pexels-photo-614810.jpeg?w=640",
     "Pexels face 3"),
    # LFW (Labeled Faces in the Wild)
    ("https://upload.wikimedia.org/wikipedia/commons/8/85/Elon_Musk_2015.jpg",
     "Wikimedia portrait"),
]

for url, label in sources:
    try:
        print(f"\n>>> {label}")
        r = requests.get(url, timeout=15,
                         headers={"User-Agent": "Mozilla/5.0 (test)"})
        if r.status_code != 200 or len(r.content) < 5000:
            print(f"   Skip (status={r.status_code}, size={len(r.content)})")
            continue
        print(f"   Got {len(r.content)} bytes")
        analyze_bytes(r.content, label)
    except Exception as e:
        print(f"   Error: {e}")
