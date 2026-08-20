"""Debug: in ra cấu trúc pose_landmarks thực sự.

M1: bo hard-code "E:/KhoiNghiep/GuardianPilot"; dung path tuong doi + argparse.

Usage:
  python results/debug_holistic.py --video path/to/video.mp4
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


_args = _parse_args("Debug: in ra cau truc pose_landmarks thuc su")
VIDEO = str(_args.video)
MODEL = _resolve_model()

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
