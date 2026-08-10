# -*- coding: utf-8 -*-
"""Debug script — reproduce error locally to see traceback."""
import os, sys, base64
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import cv2
import requests

# Load same model the server uses
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.pipeline import run_holistic
from src.landmarks import extract_features
from src.model_loader import load_drowsiness_bundle

bundle = load_drowsiness_bundle(Path(__file__).parent.parent)
model_path = str(bundle["holistic_task"])
print(f"Model: {model_path}\n")

# Download a face image
url = "https://images.pexels.com/photos/2379004/pexels-photo-2379004.jpeg?w=640"
print(f"Downloading {url}")
r = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
print(f"Got {len(r.content)} bytes\n")

img_arr = np.frombuffer(r.content, dtype=np.uint8)
img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
print(f"Image shape: {img.shape}")

# Run pipeline (catch error)
print("\n--- run_holistic() ---")
try:
    result = run_holistic(img, model_path)
    print(f"  Result: {type(result).__name__}")
    if result:
        print(f"  face_landmarks type: {type(result.face_landmarks).__name__}")
        if result.face_landmarks:
            print(f"  face_landmarks[0] type: {type(result.face_landmarks[0]).__name__}")
            if isinstance(result.face_landmarks[0], list):
                print(f"  face_landmarks[0][0] type: {type(result.face_landmarks[0][0]).__name__}")
                if hasattr(result.face_landmarks[0][0], 'x'):
                    print(f"  face_landmarks[0][0].x = {result.face_landmarks[0][0].x}")
            elif hasattr(result.face_landmarks[0], 'x'):
                print(f"  face_landmarks[0].x = {result.face_landmarks[0].x}")
        print(f"  pose_landmarks type: {type(result.pose_landmarks).__name__}")
except Exception as e:
    import traceback
    traceback.print_exc()

# Run extract_features
print("\n--- extract_features() ---")
try:
    feat = extract_features(result, img.shape[1], img.shape[0])
    print(f"  features: {feat}")
except Exception as e:
    import traceback
    traceback.print_exc()
