"""Check pose vs face detection rate trên video.

M1: bo hard-code "E:/KhoiNghiep/GuardianPilot"; dung path tuong doi + argparse.

Usage:
  python results/check_detection.py --video path/to/video.mp4
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


_args = _parse_args("Check pose vs face detection rate tren video")
VIDEO = str(_args.video)
MODEL = _resolve_model()

cap = cv2.VideoCapture(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"FPS={fps:.2f} total={total} {w}x{h}")

frame_idx = 0
face_count = 0
pose_count = 0
checked = 0

while True:
    ok, frame = cap.read()
    if not ok:
        break
    if frame_idx % 5 != 0:
        frame_idx += 1
        continue
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_result = run_holistic(rgb, MODEL)
    has_face = bool(mp_result.face_landmarks)
    has_pose = bool(mp_result.pose_landmarks)
    if has_face:
        face_count += 1
        n_face_lm = len(mp_result.face_landmarks[0]) if mp_result.face_landmarks else 0
    else:
        n_face_lm = 0
    if has_pose:
        pose_count += 1
        n_pose_lm = len(list(mp_result.pose_landmarks[0])) if mp_result.pose_landmarks else 0
    else:
        n_pose_lm = 0
    # In mỗi 30 frames
    if frame_idx % 30 == 0:
        print(f"  f={frame_idx:4d} t={frame_idx/fps:.2f}s face={has_face}({n_face_lm}lm) pose={has_pose}({n_pose_lm}lm)")
    frame_idx += 1
    checked += 1

cap.release()
print(f"\nChecked: {checked}")
print(f"Face detected: {face_count} ({100*face_count/checked:.1f}%)")
print(f"Pose detected: {pose_count} ({100*pose_count/checked:.1f}%)")
