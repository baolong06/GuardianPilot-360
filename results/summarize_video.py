"""Đọc CSV và phân tích neck_tilt.

M1: bỏ hard-code `E:/KhoiNghiep/GuardianPilot`.

Usage:
  python results/summarize_video.py [--csv results/video_nod_analysis.csv]
"""
import argparse
import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

_parser = argparse.ArgumentParser(description="Phân tích neck_tilt từ CSV")
_parser.add_argument("--csv", type=Path,
                     default=ROOT / "results" / "video_nod_analysis.csv")
_args = _parser.parse_args()
CSV = _args.csv
if not Path(CSV).is_file():
    raise SystemExit(f"Không tìm thấy CSV: {CSV}")
rows = []
with open(CSV) as f:
    r = csv.reader(f)
    header = next(r)
    for row in r:
        rows.append(row)

print(f"Total rows: {len(rows)}")
print(f"Header: {header}")

# Parse
neck_vals = []
ear_vals = []
for row in rows:
    f_idx = int(row[0])
    t_sec = float(row[1])
    neck = float(row[2]) if row[2] else None
    ear = float(row[3]) if row[3] else None
    delta = float(row[4])
    alarm_8 = int(row[5])
    alarm_15 = int(row[6])
    neck_vals.append((f_idx, t_sec, neck, delta, alarm_8, alarm_15))
    ear_vals.append((f_idx, t_sec, ear))

valid_neck = [x for x in neck_vals if x[2] is not None]
print(f"\nValid neck_tilt: {len(valid_neck)} / {len(neck_vals)} frames ({100*len(valid_neck)/len(neck_vals):.1f}%)")

if valid_neck:
    arr = np.array([x[2] for x in valid_neck])
    deltas = np.array([x[3] for x in valid_neck])
    print(f"\nneck_tilt stats (valid frames only):")
    print(f"  min={arr.min():.2f}  max={arr.max():.2f}  mean={arr.mean():.2f}  std={arr.std():.2f}")
    print(f"\ndelta (neck - baseline) stats:")
    print(f"  min={deltas.min():.2f}  max={deltas.max():.2f}  mean={deltas.mean():.2f}  std={deltas.std():.2f}")

# EAR analysis
valid_ear = [x for x in ear_vals if x[2] is not None]
if valid_ear:
    ear_arr = np.array([x[2] for x in valid_ear])
    print(f"\nEAR stats:")
    print(f"  min={ear_arr.min():.3f}  max={ear_arr.max():.3f}  mean={ear_arr.mean():.3f}")

# Tìm frames có delta lớn nhất
sorted_by_delta = sorted(valid_neck, key=lambda x: -x[3])[:20]
print(f"\nTop 20 frames by delta (largest neck deviation):")
print(f"{'frame':>6} {'t_sec':>7} {'neck':>8} {'delta':>8} {'alarm_8':>8}")
for f_idx, t_sec, neck, delta, a8, a15 in sorted_by_delta:
    print(f"{f_idx:>6} {t_sec:>7.2f} {neck:>8.2f} {delta:>8.2f} {a8:>8}")

# Frames có ear thấp
sorted_by_ear = sorted(valid_ear, key=lambda x: x[2])[:10]
print(f"\nTop 10 frames by lowest EAR:")
for f_idx, t_sec, ear in sorted_by_ear:
    print(f"  frame={f_idx} t={t_sec:.2f} EAR={ear:.3f}")
