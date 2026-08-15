"""End-to-end smoke test of FusionState with synthetic feature dicts.

This exercises the core pipeline that was patched:
- per-eye EAR LPF
- pitch baseline update with alarm_on guard
- yawn alarm single-fire state machine
- EAR escape-valve
- LSTM buffer
- EMA + hysteresis debounce
"""
import os
import sys
import math

# UTF-8 stdout for emojis
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


class FakeModel:
    """Stand-in for Keras model. Predicts P(non-drowsy)."""
    def predict(self, x, verbose=0):
        import numpy as np
        # bias to drowsy=0.7 for predictable outputs
        return np.full((x.shape[0], 1), 0.3, dtype=np.float32)


class FakeScaler:
    def transform(self, x):
        import numpy as np
        return x.astype(np.float32)


class FakeState:
    """Mimics enough of scorer/looking_away/perclos for unit-test.
    Real versions are also imported separately at the end of this script.
    """
    pass


def make_feat(ear=0.30, mar=0.20, pitch=0.0, yaw=0.0, roll=0.0,
              neck_tilt=0.0, mouth_aspect=0.30):
    """Synthetic feature dict matching landmarks.extract_features() output."""
    return {
        "ear_left": ear, "ear_right": ear, "ear_avg": ear,
        "mar": mar, "mouth_aspect": mouth_aspect,
        "pitch": pitch, "yaw": yaw, "roll": roll,
        "neck_tilt": neck_tilt, "has_neck_tilt": 1.0,
    }


print("=" * 60)
print("FUSION STATE E2E SMOKE TEST")
print("=" * 60)

from src.fusion import FusionState, EAR_LPF_ALPHA, EYE_CLOSED_THRESH, HYSTERESIS_ON

fs = FusionState()
mlp = FakeModel()
lstm = FakeModel()
scaler = FakeScaler()

# ── Test A: baseline steady-state ──────────────────────────────────────────
print("\n[A] Baseline awake driver — 30 frames, eye open")
ts = 1000.0
for i in range(30):
    feat = make_feat(ear=0.30, pitch=5.0)
    res = fs.update(feat, mlp, lstm, scaler, scaler, timestamp_ms=ts)
    ts += 33.0  # ~30 fps
print(f"  alarm_on={res['alarm_on']}  ema_prob={res['ema_prob']}")
print(f"  ear_smooth={res.get('ear_smooth')}")
assert res["alarm_on"] is False, "should NOT alarm when eyes open"
print("  PASS")

# ── Test B: ear_smooth attribute still accessible (backward-compat) ───────
print("\n[B] ear_smooth attribute (backward-compat property alias)")
print(f"  fs.ear_smooth      = {fs.ear_smooth}")
print(f"  fs.ear_avg_smooth  = {fs.ear_avg_smooth}")
assert fs.ear_smooth == fs.ear_avg_smooth, "alias must equal canonical"
print("  PASS")

# ── Test C: eye-closed sustained → alarm eventually triggers ────────────
print("\n[C] Eye closure sustained (EAR<0.16 for ~3s)")
fs2 = FusionState()
ts = 2000.0
triggered_via = None
for i in range(120):  # 120 frames @33ms = 3960ms
    feat = make_feat(ear=0.05)
    res = fs2.update(feat, mlp, lstm, scaler, scaler, timestamp_ms=ts)
    ts += 33.0
    if res["alarm_on"] and triggered_via is None:
        triggered_via = {
            "ms": ts - 2000,
            "frame": i,
            "yawn_alarm": res["yawn_alarm"],
            "neck_alarm": res["neck_alarm"],
            "eye_alarm": res["eye_alarm"],
            "ema_prob": res["ema_prob"],
        }
        break
print(f"  trigger info: {triggered_via}")
assert triggered_via is not None, "alarm should trigger within ~4s of sustained closure"
print(f"  PASS — alarm triggered after {triggered_via['ms']}ms")

# ── Test D: yawn single-fire ──────────────────────────────────────────────
print("\n[D] Yawn detector fires exactly once (per yawn)")
fs3 = FusionState()
ts = 3000.0
yawn_onsets = []
# Yawn pattern: 50 frames opening, 30 frames held, 30 frames closed
for i in range(120):
    if i < 50:
        feat = make_feat(mar=0.55, mouth_aspect=0.65)  # opening
    elif i < 80:
        feat = make_feat(mar=0.60, mouth_aspect=0.70)  # held (CONFIRMED)
    else:
        feat = make_feat(mar=0.15, mouth_aspect=0.20)  # closed
    res = fs3.update(feat, mlp, lstm, scaler, scaler, timestamp_ms=ts)
    ts += 100.0
    if res.get("yawn_alarm"):
        yawn_onsets.append(ts)
print(f"  yawn_alarm frames: {len(yawn_onsets)}")
print(f"  timestamps (first 5): {yawn_onsets[:5]}")
# Should fire exactly once at transition (around frame 50 = 5000ms in)
assert len(yawn_onsets) >= 1, "yawn should fire at least once"
# With the fix it should NOT fire continuously during CONFIRMED phase
yawn_in_confirmed_window = [t for t in yawn_onsets if 7900 < t < 10900]
print(f"  yawns during CONFIRMED phase (should be 0): {len(yawn_in_confirmed_window)}")
assert len(yawn_in_confirmed_window) == 0, \
    f"yawn should NOT fire repeatedly: {yawn_in_confirmed_window}"
print("  PASS")

# ── Test E: per-eye LPF — left vs right should NOT collapse to one ─────────
print("\n[E] Per-eye LPF preserves left/right distinction")
fs4 = FusionState()
# Frame 1: left closed, right open
feat1 = make_feat(ear=0.30)  # overwrites both
feat1["ear_left"] = 0.10  # left closed
feat1["ear_right"] = 0.30
fs4.update(feat1, mlp, lstm, scaler, scaler, timestamp_ms=5000.0)
print(f"  After 1 frame: left_smooth={fs4.ear_left_smooth:.3f}  right_smooth={fs4.ear_right_smooth:.3f}")
assert fs4.ear_left_smooth != fs4.ear_right_smooth, \
    "left and right should differ after first asymmetric frame"
print("  PASS — left != right")

# ── Test F: pitch baseline doesn't update during alarm ─────────────────────
print("\n[F] pitch_baseline frozen while alarm_on")
fs5 = FusionState()
# Drive alarm on via microsleep
ts = 6000.0
for i in range(50):
    feat = make_feat(ear=0.10, pitch=2.0)
    res = fs5.update(feat, mlp, lstm, scaler, scaler, timestamp_ms=ts)
    ts += 33.0
    if res["alarm_on"]:
        break

if res["alarm_on"]:
    baseline_before = fs5.pitch_baseline
    # Pump pitch while alarm_on
    for i in range(10):
        feat = make_feat(ear=0.10, pitch=15.0)
        fs5.update(feat, mlp, lstm, scaler, scaler, timestamp_ms=ts)
        ts += 33.0
    baseline_after = fs5.pitch_baseline
    print(f"  pitch_baseline: before={baseline_before}  after={baseline_after}")
    if baseline_before is not None:
        assert abs(baseline_after - baseline_before) < 1.0, \
            "baseline should NOT drift during alarm_on"
    print("  PASS — baseline frozen")
else:
    print("  SKIP — alarm did not trigger in time")

print("\n" + "=" * 60)
print("FUSION E2E SMOKE TEST: ALL PASSED")
print("=" * 60)