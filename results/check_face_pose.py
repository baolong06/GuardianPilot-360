"""Check face_landmarks structure on every frame sampled."""
import cv2
import sys
from pathlib import Path

ROOT = Path(r"E:/KhoiNghiep/GuardianPilot")
sys.path.insert(0, str(ROOT))

from src.pipeline import run_holistic

VIDEO = r"E:/KhoiNghiep/GuardianPilot/Drowsiness Detection - Google Chrome 2026-07-30 05-48-18.mp4"
MODEL = r"E:/KhoiNghiep/GuardianPilot/models/holistic_landmarker.task"

cap = cv2.VideoCapture(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Video: {w}x{h} {fps:.2f}fps {total}frames ({total/fps:.1f}s)")

frame_idx = 0
face_with_inner = 0
face_empty = 0
pose_with_inner = 0
pose_empty = 0
checked = 0

while True:
    ok, frame = cap.read()
    if not ok:
        break
    if frame_idx > 100:  # chỉ check 100 frames
        break
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_result = run_holistic(rgb, MODEL)
    f = mp_result.face_landmarks
    p = mp_result.pose_landmarks
    print(f"DEBUG f={type(f).__name__} p={type(p).__name__} p_val={str(p)[:80]}")
    if f_inner > 0:
        face_with_inner += 1
    else:
        face_empty += 1
    if p_inner > 0:
        pose_with_inner += 1
    else:
        pose_empty += 1
    checked += 1
    frame_idx += 1

cap.release()
print(f"Checked: {checked} frames")
print(f"  face_with_landmarks: {face_with_inner} ({100*face_with_inner/checked:.1f}%)")
print(f"  face_empty (no face detected): {face_empty} ({100*face_empty/checked:.1f}%)")
print(f"  pose_with_landmarks: {pose_with_inner} ({100*pose_with_inner/checked:.1f}%)")
print(f"  pose_empty (no pose detected): {pose_empty} ({100*pose_empty/checked:.1f}%)")
