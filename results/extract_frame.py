"""Lưu 1 frame giữa video ra PNG để kiểm tra bằng mắt.

M1: bỏ hard-code `E:/KhoiNghiep/GuardianPilot`.

Usage:
  python results/extract_frame.py --video path/to/video.mp4
"""
import argparse
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
_parser = argparse.ArgumentParser(description="Trích 1 frame giữa video")
_parser.add_argument("--video", type=Path, required=True)
_parser.add_argument("--out", type=Path,
                     default=ROOT / "results" / "video_mid_frame.png")
_args = _parser.parse_args()
VIDEO = str(_args.video)
if not _args.video.is_file():
    raise SystemExit(f"Không tìm thấy video: {VIDEO}")
cap = cv2.VideoCapture(VIDEO)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
# Frame giữa video
cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
ok, frame = cap.read()
if ok:
    out = str(_args.out)
    cv2.imwrite(out, frame)
    print(f"Saved mid frame to {out}")
    print(f"Frame shape: {frame.shape}, mean brightness: {frame.mean():.1f}")
else:
    print("Cannot read frame")
cap.release()
