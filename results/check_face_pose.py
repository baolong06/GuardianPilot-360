"""Check face_landmarks structure on every frame sampled.

M1: bo hard-code "E:/KhoiNghiep/GuardianPilot"; dung path tuong doi + argparse.

Usage:
  python results/check_face_pose.py --video path/to/video.mp4
"""
import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline import run_holistic


def _resolve_model() -> str:
    """Tìm holistic_landmarker.task theo đúng thứ tự ưu tiên mà app.py dùng."""
    from src.model_loader import model_search_roots, resolve_artifact
    found = resolve_artifact("holistic_landmarker.task", model_search_roots(ROOT))
    if found is None:
        raise SystemExit(
            "Khong tim thay holistic_landmarker.task trong models/compatible, "
            "models/ hay results/. Chay: python tools/convert_models.py --in-place"
        )
    return str(found)


def _parse_args(desc: str):
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("--video", type=Path, required=True,
                        help="Duong dan video can phan tich")
    args = parser.parse_args()
    if not args.video.is_file():
        raise SystemExit(f"Khong tim thay video: {args.video}")
    return args


_args = _parse_args("Check face_landmarks structure tren tung frame")
VIDEO = str(_args.video)
MODEL = _resolve_model()

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
