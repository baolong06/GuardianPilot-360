"""Debug: in ra cấu trúc pose_landmarks thực sự."""
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
print(f"FPS={fps:.2f} total={total}")

frame_idx = 0
checked = 0

while True:
    ok, frame = cap.read()
    if not ok:
        break
    # Kiểm tra mỗi frame (không skip) lần đầu để debug
    if frame_idx % 30 != 0:
        frame_idx += 1
        continue
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_result = run_holistic(rgb, MODEL)

    # Debug chi tiết
    face_attr = mp_result.face_landmarks
    pose_attr = mp_result.pose_landmarks

    f_type = type(face_attr).__name__
    f_len = len(face_attr) if hasattr(face_attr, '__len__') else 'NA'
    if f_len and f_len != 'NA':
        # Nếu là list of lists, in ra cấu trúc
        try:
            inner_len = len(face_attr[0])
            print(f"  f={frame_idx:4d} t={frame_idx/fps:.2f}s face=type={f_type} outer_len={f_len} inner_len={inner_len}")
        except (TypeError, IndexError):
            print(f"  f={frame_idx:4d} t={frame_idx/fps:.2f}s face=type={f_type} len={f_len} (cannot get inner)")

    p_type = type(pose_attr).__name__
    p_len = len(pose_attr) if hasattr(pose_attr, '__len__') else 'NA'
    if p_len != 'NA':
        try:
            inner_len = len(list(pose_attr[0]))
            print(f"                              pose=type={p_type} outer_len={p_len} inner_len={inner_len}")
        except (TypeError, IndexError):
            print(f"                              pose=type={p_type} len={p_len} (cannot get inner)")
    frame_idx += 1
    checked += 1

cap.release()
