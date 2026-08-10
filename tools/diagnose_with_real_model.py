# -*- coding: utf-8 -*-
"""
Diagnostic thật — dùng model thật từ src/model_loader.py, kiểm tra
các trường hợp mà trong production dễ gây cảnh báo sai:
  - EAR mở nhưng model MLP output cao (model bias)
  - LSTM kẹt giá trị
  - Tương tác giữa các signal
"""
import os, sys
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model_loader import load_drowsiness_bundle
from src.fusion import FusionState

from pathlib import Path
BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
bundle = load_drowsiness_bundle(BASE_DIR)
mlp = bundle["mlp_model"]
lstm = bundle["lstm_model"]
mlp_scaler = bundle["mlp_scaler"]
lstm_scaler = bundle["lstm_scaler"]
print(f"[model] load_mode={bundle['load_mode']}")
print()


def feat(ear=0.3, mar=0.3, neck=0.0, pitch=0.0, yaw=0.0, roll=0.0,
         mouth_aspect=0.3):
    return {
        "ear_left": ear, "ear_right": ear, "ear_avg": ear,
        "mar": mar, "mouth_aspect": mouth_aspect,
        "pitch": pitch, "yaw": yaw, "roll": roll,
        "neck_tilt": neck,
    }


# ── Test 1: Mắt m� 5s — kiểm tra model có bias về phía drowsy không ─────
print("=" * 70)
print("TEST 1: Mắt mở (EAR=0.30) liên tục 5s — model có bias không?")
print("=" * 70)
fs = FusionState()
seq = [(i * 100, feat(ear=0.30)) for i in range(50)]
for t, f in seq:
    out = fs.update(f, mlp, lstm, mlp_scaler, lstm_scaler, timestamp_ms=t)
final = out
print(f"  Sau 5s: ema={final['ema_prob']:.3f} alarm={final['alarm_on']} "
      f"state={final['drowsiness_state']} score={final['drowsiness_score']:.3f} "
      f"p_mlp={final['p_mlp_drowsy']:.3f} p_lstm={final['p_lstm_drowsy']:.3f}")


# ── Test 2: LSTM window đầy → LSTM output có đúng không ────────────────
print()
print("=" * 70)
print("TEST 2: LSTM window đầy với EAR cao → LSTM có hợp lý không?")
print("=" * 70)
fs = FusionState()
seq = [(i * 100, feat(ear=0.32)) for i in range(30)]
for t, f in seq:
    out = fs.update(f, mlp, lstm, mlp_scaler, lstm_scaler, timestamp_ms=t)
print(f"  Frame 30 (window đầy): p_lstm={out['p_lstm_drowsy']:.3f}")
print(f"  → Nếu p_lstm > 0.3 khi EAR=0.32, model LSTM bị bias.")


# ── Test 3: EAR=0.18 (threshold) liên tục 5s — model phản ứng thế nào? ─
print()
print("=" * 70)
print("TEST 3: EAR=0.18 (threshold) liên tục 5s — có bị false-alarm?")
print("=" * 70)
fs = FusionState()
seq = [(i * 100, feat(ear=0.18)) for i in range(50)]
prev_state = None
transitions = []
for t, f in seq:
    out = fs.update(f, mlp, lstm, mlp_scaler, lstm_scaler, timestamp_ms=t)
    if out["drowsiness_state"] != prev_state:
        transitions.append((t, out["drowsiness_state"], out["ema_prob"]))
        prev_state = out["drowsiness_state"]
print("  Transitions:")
for t, state, ema in transitions:
    print(f"    t={t}ms → state={state} (ema={ema:.3f})")


# ── Test 4: Đóng mở mắt xen kẽ (chớp mắt giả lập) ──────────────────
print()
print("=" * 70)
print("TEST 4: Chớp mắt 150ms (EAR=0.05) — false alarm?")
print("=" * 70)
fs = FusionState()
seq = []
for i in range(20):
    seq.append((i * 100, feat(ear=0.30)))
seq.append((2100, feat(ear=0.05)))  # 1 frame nhắm
for i in range(20):
    seq.append(((22 + i) * 100, feat(ear=0.30)))
any_alarm = False
max_ema = 0.0
for t, f in seq:
    out = fs.update(f, mlp, lstm, mlp_scaler, lstm_scaler, timestamp_ms=t)
    if out["alarm_on"]:
        any_alarm = True
    max_ema = max(max_ema, out["ema_prob"])
print(f"  Any alarm? {any_alarm} (max_ema={max_ema:.3f})")
print(f"  → {'❌ False alarm!' if any_alarm else '✅ OK'}")


# ── Test 5: Microsleep thật 1.5s — alarm latency bao nhiêu? ───────────
print()
print("=" * 70)
print("TEST 5: Microsleep thật (EAR=0.05 liên tục 1.5s)")
print("=" * 70)
fs = FusionState()
seq = []
for i in range(20):
    seq.append((i * 100, feat(ear=0.30)))
close_start = None
for i in range(15):
    if i == 0:
        close_start = (20 + i) * 100
    seq.append(((20 + i) * 100, feat(ear=0.05)))
alarm_at = None
for t, f in seq:
    out = fs.update(f, mlp, lstm, mlp_scaler, lstm_scaler, timestamp_ms=t)
    if out["alarm_on"] and alarm_at is None and close_start is not None and t >= close_start:
        alarm_at = t - close_start
if alarm_at is not None:
    print(f"  ✅ Alarm sau {alarm_at}ms nhắm mắt")
else:
    print(f"  ❌ KHÔNG alarm!")


# ── Test 6: Gật gù mạnh (neck tilt 25° trong 300ms) — neck_alarm có bật? ─
print()
print("=" * 70)
print("TEST 6: Cú gật 300ms (neck tilt 0 → 25 → 0)")
print("=" * 70)
fs = FusionState()
seq = []
for i in range(20):
    seq.append((i * 100, feat(neck=0.0)))
# cú gật 300ms
for i, nt in enumerate([8, 16, 24, 24, 16, 8, 0]):
    seq.append(((20 + i) * 100, feat(neck=float(nt))))
for i in range(20):
    seq.append(((27 + i) * 100, feat(neck=0.0)))
neck_alarm_count = 0
for t, f in seq:
    out = fs.update(f, mlp, lstm, mlp_scaler, lstm_scaler, timestamp_ms=t)
    if out["neck_alarm"]:
        neck_alarm_count += 1
print(f"  neck_alarm triggered trong {neck_alarm_count} frames")
print(f"  → {'✅ OK' if neck_alarm_count >= 1 else '❌ Không phát hiện gật gù!'}")


# ── Test 7: Ngáp 1.5s (MAR=0.6, mouth_aspect=0.7) ────────────────────
print()
print("=" * 70)
print("TEST 7: Ngáp 1.5s")
print("=" * 70)
fs = FusionState()
seq = []
for i in range(20):
    seq.append((i * 100, feat(mar=0.3, mouth_aspect=0.3)))
for i in range(15):
    seq.append(((20 + i) * 100, feat(mar=0.6, mouth_aspect=0.7)))
yawn_count = 0
for t, f in seq:
    out = fs.update(f, mlp, lstm, mlp_scaler, lstm_scaler, timestamp_ms=t)
    if out["yawn_alarm"]:
        yawn_count += 1
print(f"  yawn_alarm triggered trong {yawn_count} frames")
print(f"  → {'✅ OK' if yawn_count >= 1 else '❌ Không phát hiện ngáp!'}")


# ── Test 8: Stress — EAR dao động 0.16↔0.22 liên tục 10s ────────────
print()
print("=" * 70)
print("TEST 8: Stress test — EAR dao động quanh threshold 10s")
print("=" * 70)
fs = FusionState()
seq = []
for i in range(100):
    ear = 0.16 if i % 2 == 0 else 0.22
    seq.append((i * 100, feat(ear=ear)))
flips = 0
alarm_periods = 0
prev_alarm = False
for t, f in seq:
    out = fs.update(f, mlp, lstm, mlp_scaler, lstm_scaler, timestamp_ms=t)
    if out["alarm_on"] != prev_alarm:
        flips += 1
        if out["alarm_on"]:
            alarm_periods += 1
        prev_alarm = out["alarm_on"]
print(f"  alarm flip {flips} lần ({alarm_periods} periods alarm ON)")
print(f"  → {'⚠ Flap!' if flips > 5 else '✅ OK'}")


# ── Test 9: Stress — đầu quay sang trái/phải (yaw lớn) liên tục 10s ─
print()
print("=" * 70)
print("TEST 9: Stress test — Yaw dao động ±40° 10s (looking_away)")
print("=" * 70)
fs = FusionState()
seq = []
for i in range(100):
    yaw = -40 if i % 2 == 0 else 40
    seq.append((i * 100, feat(yaw=float(yaw))))
looking_away_periods = 0
prev = False
for t, f in seq:
    out = fs.update(f, mlp, lstm, mlp_scaler, lstm_scaler, timestamp_ms=t)
    if out.get("looking_away") and not prev:
        looking_away_periods += 1
    prev = out.get("looking_away", False)
print(f"  looking_away periods: {looking_away_periods}")
print(f"  → {'✅' if looking_away_periods >= 1 else '�'}")


# ── Test 10: Stress — gật đầu nhiều lần liên tiếp ───────────────────
print()
print("=" * 70)
print("TEST 10: Stress — gật đầu 3 lần liên tiếp (mỗi cú 300ms)")
print("=" * 70)
fs = FusionState()
seq = []
for i in range(20):
    seq.append((i * 100, feat()))
nod_frames = [8, 16, 24, 24, 16, 8, 0]  # 1 cú gật
t = 20
for cycle in range(3):
    for nt in nod_frames:
        seq.append((t * 100, feat(neck=float(nt))))
        t += 1
    t += 5  # nghỉ giữa các cú
for i in range(30):
    seq.append((t * 100, feat()))
    t += 1

nod_count = 0
in_alarm = False
for tt, f in seq:
    out = fs.update(f, mlp, lstm, mlp_scaler, lstm_scaler, timestamp_ms=tt)
    if out["neck_alarm"] and not in_alarm:
        nod_count += 1
        in_alarm = True
    if not out["neck_alarm"]:
        in_alarm = False
print(f"  Phát hiện {nod_count}/3 cú gật")
print(f"  → {'✅' if nod_count >= 2 else '❌ Miss gật gù!'}")
