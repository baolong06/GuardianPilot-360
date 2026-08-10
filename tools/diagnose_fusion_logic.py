# -*- coding: utf-8 -*-
"""
Diagnostic trực tiếp trên FusionState — bypass MediaPipe, feed feature giả.
Phát hiện bug logic: cảnh báo sớm/muộn, kẹt ON, flap ON↔OFF, sai state machine.

Vì model đã load sẵn ở app.py, ta import các class từ src/ và tự tạo mock model
trả output mong muốn để tách bạch logic fusion vs model.
"""
import os, sys
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.fusion import FusionState


# ── Mock model/scaler ───────────────────────────────────────────────────────
class MockScaler:
    def transform(self, x):
        return np.asarray(x, dtype=np.float32)


class MockMLP:
    """Trả P(non-drowsy) cho mỗi frame."""
    def __init__(self, mode="open"):
        self.mode = mode
        self.calls = 0
    def predict(self, x, verbose=0):
        self.calls += 1
        ear = float(x[0, 2])  # ear_avg là cột 2 của MLP_FEAT_COLS
        if self.mode == "open":
            p = 0.9 if ear > 0.25 else 0.4
        elif self.mode == "closed":
            p = 0.1
        elif self.mode == "responsive":
            # EAR thấp → drowsy, EAR cao → not drowsy
            p = max(0.0, min(1.0, 1.5 - ear * 4))
        else:
            p = 0.5
        return np.array([[p]], dtype=np.float32)


class MockLSTM:
    def __init__(self, mode="open"):
        self.mode = mode
        self.calls = 0
    def predict(self, x, verbose=0):
        self.calls += 1
        ear = float(np.mean(x[0, :, 0]))  # ear_avg trong window
        if self.mode == "open":
            p = 0.85 if ear > 0.25 else 0.5
        elif self.mode == "closed":
            p = 0.2
        elif self.mode == "responsive":
            p = max(0.0, min(1.0, 1.5 - ear * 4))
        else:
            p = 0.5
        return np.array([[p]], dtype=np.float32)


def make_feat(ear=0.3, mar=0.3, neck_tilt=0.0, pitch=0.0, yaw=0.0, roll=0.0,
              mouth_aspect=None):
    if mouth_aspect is None:
        mouth_aspect = mar
    return {
        "ear_left":  ear,
        "ear_right": ear,
        "ear_avg":   ear,
        "mar":       mar,
        "mouth_aspect": mouth_aspect,
        "pitch":     pitch,
        "yaw":       yaw,
        "roll":      roll,
        "neck_tilt": neck_tilt,
    }


def run_sequence(seq, label="", mlp_mode="responsive", lstm_mode="responsive",
                 verbose=True):
    fs = FusionState()
    mlp = MockMLP(mode=mlp_mode)
    lstm = MockLSTM(mode=lstm_mode)
    scaler = MockScaler()
    rows = []
    for i, (t_ms, feat) in enumerate(seq):
        out = fs.update(feat, mlp, lstm, scaler, scaler, timestamp_ms=t_ms)
        rows.append({
            "i":      i,
            "t_ms":   t_ms,
            "ear":    feat["ear_avg"],
            "neck":   feat["neck_tilt"],
            "mar":    feat["mar"],
            "ema":    out["ema_prob"],
            "alarm":  out["alarm_on"],
            "neck_alarm": out["neck_alarm"],
            "eye_alarm":  out["eye_alarm"],
            "yawn_alarm": out["yawn_alarm"],
            "state":  out["drowsiness_state"],
            "score":  out["drowsiness_score"],
            "lvl":    out["alert_level"],
            "p_mlp":  out["p_mlp_drowsy"],
            "p_lstm": out["p_lstm_drowsy"],
            "eyes_open_ms":  getattr(fs, "eyes_open_streak_ms", None),
            "eyes_close_ms": getattr(fs, "eye_closed_streak_ms", None),
            "neck_peak":     max(fs.neck_peak_buffer) if fs.neck_peak_buffer else 0,
        })
    if verbose:
        print(f"\n{'='*70}\n  TEST: {label}\n{'='*70}")
        print(f"  {'i':>2} {'t(ms)':>5} {'EAR':>5} {'Neck':>6} {'MAR':>5} "
              f"{'p_mlp':>6} {'p_lstm':>6} {'EMA':>6} {'alarm':>5} {'state':>10} "
              f"{'eyeOn':>5} {'eyeOff':>6} {'neckPk':>6}")
        for r in rows:
            print(f"  {r['i']:>2} {r['t_ms']:>5} {r['ear']:>5.2f} "
                  f"{r['neck']:>6.1f} {r['mar']:>5.2f} "
                  f"{r['p_mlp']:>6.3f} {r['p_lstm'] or 0:>6.3f} "
                  f"{r['ema']:>6.3f} {str(r['alarm']):>5} {r['state']:>10} "
                  f"{r['eyes_open_ms']/1000:>4.1f}s "
                  f"{r['eyes_close_ms']/1000:>5.1f}s "
                  f"{r['neck_peak']:>6.1f}")
    return rows


# ── Test scenarios ─────────────────────────────────────────────────────────
def scenario_baseline_open():
    """30 frames @ 100ms: mắt mở bình thường → phải NORMAL."""
    seq = [(i * 100, make_feat(ear=0.32)) for i in range(30)]
    return run_sequence(seq, "BASELINE — mắt mở bình thường (EAR=0.32)")


def scenario_normal_to_closed_microsleep():
    """
    10 frames mắt mở → 15 frames mắt nhắm (EAR=0.10) → 10 frames mở lại.
    Mong đợi: alarm bật sau ~1.2s nhắm liên tục (microsleep).
    """
    seq = []
    for i in range(10):
        seq.append((i * 100, make_feat(ear=0.32)))         # open
    for i in range(15):
        seq.append(((10 + i) * 100, make_feat(ear=0.10)))  # closed
    for i in range(10):
        seq.append(((25 + i) * 100, make_feat(ear=0.32)))  # open
    return run_sequence(seq, "MICROSLEEP — nhắm 1.5s rồi mở lại")


def scenario_normal_to_drowsy_progressive():
    """
    EAR giảm dần 0.32 → 0.18 trong 5s, giữ ở 0.18 thêm 5s.
    Mong đợi: alarm bật khi EAR<0.18 kéo dài.
    """
    seq = []
    ear = 0.32
    for i in range(50):
        if i < 25:
            ear = max(0.18, 0.32 - i * 0.006)
        seq.append((i * 100, make_feat(ear=ear)))
    return run_sequence(seq, "PROGRESSIVE DROWSY — EAR giảm dần")


def scenario_false_alarm_blink():
    """
    Mắt bình thường, chớp mắt 1 lần (EAR về 0.05 trong 200ms).
    Mong đợi: KHÔNG alarm (chớp mắt không phải microsleep).
    """
    seq = []
    for i in range(30):
        ear = 0.30
        if i == 10:
            ear = 0.05  # blink
        elif i == 11:
            ear = 0.20  # mid-blink
        seq.append((i * 100, make_feat(ear=ear)))
    return run_sequence(seq, "FALSE ALARM TEST — chớp mắt 200ms")


def scenario_false_alarm_ear_oscillation():
    """
    EAR dao động quanh ngưỡng 0.18 (0.20 ↔ 0.16).
    Test xem có alarm stuck-on do EAR bounce.
    """
    seq = []
    for i in range(60):
        ear = 0.20 if i % 2 == 0 else 0.16
        seq.append((i * 100, make_feat(ear=ear)))
    return run_sequence(seq, "FALSE ALARM — EAR dao động quanh ngưỡng")


def scenario_head_nod():
    """
    Neck tilt đột biến: 0° → 25° trong 200ms → về 0°.
    Mong đợi: neck_alarm=True trong khi cú gật.
    """
    seq = []
    for i in range(10):
        seq.append((i * 100, make_feat(neck_tilt=0.0)))
    for i in range(3):
        # cú gật 300ms, mỗi frame ~100ms
        seq.append(((10 + i) * 100, make_feat(neck_tilt=8.0 * (i + 1))))
    for i in range(2):
        seq.append(((13 + i) * 100, make_feat(neck_tilt=24.0 - 8.0 * i)))
    for i in range(15):
        seq.append(((15 + i) * 100, make_feat(neck_tilt=0.0)))
    return run_sequence(seq, "HEAD NOD — cú gật 300ms")


def scenario_yawn():
    """MAR=0.6, aspect=0.7 trong 2s → ngáp."""
    seq = []
    for i in range(10):
        seq.append((i * 100, make_feat(mar=0.3, mouth_aspect=0.3)))
    for i in range(20):
        seq.append(((10 + i) * 100, make_feat(mar=0.6, mouth_aspect=0.7)))
    for i in range(10):
        seq.append(((30 + i) * 100, make_feat(mar=0.3, mouth_aspect=0.3)))
    return run_sequence(seq, "YAWN — miệng mở rộng 2s")


def scenario_alarm_stuck_on():
    """
    Sau khi alarm bật, mắt mở lại bình thường.
    Mong đ�i: alarm TẮT trong vòng 0.5-1s sau khi mắt mở.
    """
    seq = []
    # 5s m�
    for i in range(50):
        seq.append((i * 100, make_feat(ear=0.32)))
    # 2s nhắm → trigger alarm
    for i in range(20):
        seq.append(((50 + i) * 100, make_feat(ear=0.10)))
    # 5s mở lại → alarm phải tắt
    for i in range(50):
        seq.append(((70 + i) * 100, make_feat(ear=0.32)))
    return run_sequence(seq, "ALARM STUCK-ON — sau microsleep, mở lại bình thường")


def scenario_quick_recovery():
    """Microsleep 1.2s → m� lại → kiểm tra alarm có bật/tắt đúng lúc không."""
    seq = []
    for i in range(50):
        seq.append((i * 100, make_feat(ear=0.32)))
    # nhắm 1.2s
    for i in range(12):
        seq.append(((50 + i) * 100, make_feat(ear=0.10)))
    # mở lại 2s
    for i in range(20):
        seq.append(((62 + i) * 100, make_feat(ear=0.32)))
    return run_sequence(seq, "QUICK RECOVERY — 1.2s nhắm rồi mở")


def scenario_ear_at_threshold():
    """
    EAR = 0.18 (= threshold) liên tục 3s.
    Test edge case tại threshold.
    """
    seq = []
    for i in range(30):
        seq.append((i * 100, make_feat(ear=0.18)))
    return run_sequence(seq, "EAR AT THRESHOLD — 0.18 liên tục 3s")


def scenario_low_light_face_lost():
    """
    20 frames mắt mở → 5 frames không có mặt → 20 frames mở.
    Mong đợi: alarm không bật (face mất tạm thời).
    """
    seq = []
    for i in range(20):
        seq.append((i * 100, make_feat(ear=0.32)))
    # Khi face mất, fusion.update() không được gọi → test khác
    for i in range(20):
        seq.append(((25 + i) * 100, make_feat(ear=0.32)))
    return run_sequence(seq, "RECOVERY FROM FACE LOST — giả lập")


# ── Summary ────────────────────────────────────────────────────────────────
def summary(all_results):
    print("\n\n" + "=" * 70)
    print("  TÓM TẮT VẤN ĐỀ PHÁT HIỆN")
    print("=" * 70)

    issues = []

    for label, rows in all_results.items():
        # Tìm alarm stuck
        last_alarm = None
        last_alarm_idx = None
        for r in rows:
            if r["alarm"] and last_alarm_idx is None:
                last_alarm_idx = r["i"]
            if not r["alarm"] and last_alarm_idx is not None:
                gap = r["i"] - last_alarm_idx
                if gap > 30:
                    issues.append(
                        f"  ⚠ [{label}] alarm stuck ON trong {gap} frame "
                        f"(~{gap*100}ms) trước khi tắt"
                    )
                last_alarm_idx = None

        # Tìm flap ON↔OFF (nhiều lần đổi trạng thái trong thời gian ngắn)
        flips = 0
        for i in range(1, len(rows)):
            if rows[i]["alarm"] != rows[i - 1]["alarm"]:
                flips += 1
        if flips >= 6:
            issues.append(
                f"  ⚠ [{label}] alarm FLAP {flips} lần — không ổn định"
            )

        # Kiểm tra microsleep case
        if "MICROSLEEP" in label or "STUCK" in label or "QUICK" in label:
            microsleep_frames = [
                r for r in rows
                if r["ear"] < 0.18 and r["eyes_close_ms"] >= 1200
            ]
            if microsleep_frames:
                first_alarm = next(
                    (r for r in rows if r["alarm"]), None
                )
                if first_alarm is None:
                    issues.append(
                        f"  ❌ [{label}] microsleep không trigger alarm!"
                    )
                else:
                    close_start = next(
                        (r["i"] for r in rows if r["ear"] < 0.18), None
                    )
                    if close_start is not None:
                        latency = first_alarm["i"] - close_start
                        if latency > 30:
                            issues.append(
                                f"  ⚠ [{label}] alarm TRỄ: {latency*100}ms "
                                f"sau khi nhắm mắt"
                            )

    if not issues:
        print("  ✅ Không phát hiện vấn đề rõ ràng.")
    for i in issues:
        print(i)


if __name__ == "__main__":
    results = {}
    results["BASELINE"]        = scenario_baseline_open()
    results["MICROSLEEP"]      = scenario_normal_to_closed_microsleep()
    results["PROGRESSIVE"]     = scenario_normal_to_drowsy_progressive()
    results["FALSE_BLINK"]     = scenario_false_alarm_blink()
    results["EAR_OSCILLATION"] = scenario_false_alarm_ear_oscillation()
    results["HEAD_NOD"]        = scenario_head_nod()
    results["YAWN"]            = scenario_yawn()
    results["STUCK_ON"]        = scenario_alarm_stuck_on()
    results["QUICK_RECOVER"]   = scenario_quick_recovery()
    results["EAR_THRESHOLD"]   = scenario_ear_at_threshold()
    summary(results)
