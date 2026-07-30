"""Phân tích video: trích neck_tilt + EAR + alarm signal theo từng frame."""
import cv2
import numpy as np
import math
import sys
import csv
from pathlib import Path

ROOT = Path(r"E:/KhoiNghiep/GuardianPilot")
sys.path.insert(0, str(ROOT))

from src.landmarks import extract_features

# Import pipeline (holistic + ROI)
from src.pipeline import run_holistic

VIDEO = r"E:/KhoiNghiep/GuardianPilot/Drowsiness Detection - Google Chrome 2026-07-30 05-48-18.mp4"
OUT_CSV = r"E:/KhoiNghiep/GuardianPilot/results/video_nod_analysis.csv"
MODEL_PATH = r"E:/KhoiNghiep/GuardianPilot/models/holistic_landmarker.task"

cap = cv2.VideoCapture(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"FPS={fps:.2f}  total={total}  {w}x{h}  duration={total/fps:.1f}s")

Path(r"E:/KhoiNghiep/GuardianPilot/results").mkdir(parents=True, exist_ok=True)

neck_vals = []
ear_vals = []
results = []
neck_alarm_count = 0

with open(OUT_CSV, "w", newline="") as f:
    wri = csv.writer(f)
    wri.writerow(["frame", "t_sec", "neck_tilt", "ear_avg", "neck_delta_abs", "neck_alarm_8", "neck_alarm_15"])

    frame_idx = 0
    # Tham số mô phỏng như fusion: baseline chậm, EMA, alarm-on
    EMA_ALPHA = 0.5
    NECK_BASELINE_ALPHA = 0.05
    HYSTERESIS_ON = 0.55
    HYSTERESIS_OFF = 0.30
    MIN_ON_SEC = 0.2
    neck_baseline = None
    ema_prob = 0.0
    time_above_on_ms = 0.0
    alarm_on = False
    last_alarm_change_frame = -1
    last_alarm_state = False

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # Giảm xuống mỗi 2 frame để nhanh
        if frame_idx % 2 != 0:
            frame_idx += 1
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_result = run_holistic(rgb, MODEL_PATH)
        feat = extract_features(mp_result, w, h)
        neck_tilt = float("nan")
        ear_avg = float("nan")
        if feat is not None:
            neck_tilt = feat.get("neck_tilt", float("nan"))
            ear_avg = feat.get("ear_avg", float("nan"))
            neck_vals.append(neck_tilt if not math.isnan(neck_tilt) else None)

        # Mô phỏng fusion logic
        dt_ms = (1000 / fps) * 2  # 2 frame interval
        neck_alarm = False
        if not math.isnan(neck_tilt):
            if neck_baseline is None:
                neck_baseline = neck_tilt
            elif not alarm_on:
                neck_baseline = (
                    (1 - NECK_BASELINE_ALPHA) * neck_baseline
                    + NECK_BASELINE_ALPHA * neck_tilt
                )
            if abs(neck_tilt - neck_baseline) > 8.0:
                neck_alarm = True
                neck_alarm_count += 1
        neck_delta = abs(neck_tilt - neck_baseline) if (neck_baseline is not None and not math.isnan(neck_tilt)) else 0.0

        combined = 0.70 if neck_alarm else 0.0
        ema_prob = (1 - EMA_ALPHA) * ema_prob + EMA_ALPHA * combined
        if ema_prob >= HYSTERESIS_ON:
            time_above_on_ms += dt_ms
        else:
            time_above_on_ms = 0.0
        if not alarm_on and time_above_on_ms >= MIN_ON_SEC * 1000:
            alarm_on = True
            last_alarm_change_frame = frame_idx
        if alarm_on != last_alarm_state:
            print(f"[frame {frame_idx} t={frame_idx/fps:.1f}s] ALARM={alarm_on} neck={neck_tilt:.2f} baseline={neck_baseline:.2f} delta={neck_delta:.2f} ema={ema_prob:.3f}")
            last_alarm_state = alarm_on

        wri.writerow([
            frame_idx,
            round(frame_idx / fps, 3),
            round(neck_tilt, 3) if not math.isnan(neck_tilt) else "",
            round(ear_avg, 3) if not math.isnan(ear_avg) else "",
            round(neck_delta, 3),
            int(neck_alarm),
            int(abs(neck_tilt - neck_baseline) > 15.0) if (neck_baseline is not None and not math.isnan(neck_tilt)) else 0,
        ])
        frame_idx += 1

cap.release()
print(f"\nDone. neck_alarm frames (delta>8): {neck_alarm_count}")
print(f"CSV: {OUT_CSV}")

# Thống kê neck_tilt
valid = [v for v in neck_vals if v is not None]
if valid:
    arr = np.array(valid)
    print(f"\nneck_tilt stats: min={arr.min():.1f} max={arr.max():.1f} mean={arr.mean():.1f} std={arr.std():.1f}")
    # Tìm các "cú gật" (delta từ baseline)
    print(f"Frames with valid neck: {len(valid)}/{frame_idx//2} ({100*len(valid)/(frame_idx//2):.1f}%)")
