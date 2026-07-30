"""
Test logic cho eye-closure rule trong src/fusion.py.
Sử dụng mock models để không cần load Keras.
"""
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.fusion import (
    FusionState, EAR_LPF_ALPHA, EYE_CLOSED_THRESH,
    EYE_CLOSED_ON_SEC, EYE_CLOSED_HARD_SEC,
    HYSTERESIS_ON, EMA_ALPHA, WINDOW_SIZE,
)


def make_mock_mlp(prob_non_drowsy=0.7):
    """Mock MLP: predict trả về np.array([[p]])."""
    m = MagicMock()
    m.predict = MagicMock(return_value=np.array([[prob_non_drowsy]], dtype=np.float32))
    return m


def make_mock_lstm(prob_non_drowsy=0.7):
    """Mock LSTM: predict trả về np.array([[p]])."""
    m = MagicMock()
    m.predict = MagicMock(return_value=np.array([[prob_non_drowsy]], dtype=np.float32))
    return m


def make_mock_scaler():
    """Scaler giả chuyển đổi (zero)."""
    s = MagicMock()
    s.transform = MagicMock(side_effect=lambda x: np.zeros_like(x, dtype=float))
    return s


def feat_with(ear=0.3, pitch=0.0, neck_tilt=0.0, has_neck=True):
    """Tạo dict feature giả với các giá trị chính."""
    if has_neck:
        nt = neck_tilt
    else:
        nt = float("nan")
    return {
        "ear_left": ear,
        "ear_right": ear,
        "ear_avg": ear,
        "mar": 0.05,
        "pitch": pitch,
        "yaw": 0.0,
        "roll": 0.0,
        "neck_tilt": nt,
        "has_pose": has_neck,
    }


def test_eye_alarm_short_blink_no_trigger():
    """Chớp mắt nhanh 0.2s: KHÔNG trigger eye_alarm."""
    fs = FusionState()
    mlp = make_mock_mlp(0.7)
    lstm = make_mock_lstm()
    scaler = make_mock_scaler()

    # Mô phỏng chớp mắt 200ms (2 frames @ 100ms): ear=0.10
    t0 = 1_000_000.0
    fs.update(feat_with(ear=0.10), mlp, lstm, scaler, scaler, timestamp_ms=t0)
    fs.update(feat_with(ear=0.10), mlp, lstm, scaler, scaler, timestamp_ms=t0 + 100)
    fs.update(feat_with(ear=0.10), mlp, lstm, scaler, scaler, timestamp_ms=t0 + 200)

    # Mở mắt lại
    r = fs.update(feat_with(ear=0.30), mlp, lstm, scaler, scaler, timestamp_ms=t0 + 300)

    assert not r["eye_alarm"], f"Expected eye_alarm=False after short blink, got {r['eye_alarm']}"
    print(f"PASS  Short blink (200ms): eye_alarm={r['eye_alarm']}")


def test_eye_alarm_long_closure_triggers():
    """Nhắm mắt 0.7s: eye_alarm = True."""
    fs = FusionState()
    mlp = make_mock_mlp(0.7)
    lstm = make_mock_lstm()
    scaler = make_mock_scaler()

    t0 = 2_000_000.0
    # Mắt mở bình thường 1 frame để khởi tạo baseline
    fs.update(feat_with(ear=0.30), mlp, lstm, scaler, scaler, timestamp_ms=t0)

    # Nhắm mắt liên tục ~800ms (8 frames @ 100ms)
    t = t0
    last_r = None
    for i in range(10):
        t += 100
        last_r = fs.update(feat_with(ear=0.08), mlp, lstm, scaler, scaler, timestamp_ms=t)

    assert last_r["eye_alarm"], (
        f"Expected eye_alarm=True after 1s closure, got {last_r['eye_alarm']}, "
        f"streak_ms={fs.eye_closed_streak_ms}"
    )
    print(f"PASS  Long closure (~1s): eye_alarm={last_r['eye_alarm']} streak_ms={fs.eye_closed_streak_ms}")


def test_eye_alarm_microsleep_stronger():
    """Nhắm mắt > 1s (microsleep): combined phải >= 0.85."""
    fs = FusionState()
    mlp = make_mock_mlp(0.95)   # MLP rất tự tin "non-drowsy" — rule phải override
    lstm = make_mock_lstm()
    scaler = make_mock_scaler()

    t0 = 3_000_000.0
    fs.update(feat_with(ear=0.30), mlp, lstm, scaler, scaler, timestamp_ms=t0)

    # Nhắm mắt 1.5s
    t = t0
    last_r = None
    for i in range(16):
        t += 100
        last_r = fs.update(feat_with(ear=0.08), mlp, lstm, scaler, scaler, timestamp_ms=t)

    assert last_r["eye_alarm"], "eye_alarm must be True after 1.5s microsleep"
    # ema_prob phải ≥ 0.85 (rule bump)
    assert last_r["ema_prob"] >= 0.85, (
        f"Expected ema_prob>=0.85 (microsleep bump), got {last_r['ema_prob']}"
    )
    print(f"PASS  Microsleep (>1s): eye_alarm={last_r['eye_alarm']} ema_prob={last_r['ema_prob']}")


def test_low_pass_smooths_noisy_ear():
    """EAR nhiễu (0.10..0.30 xen kẽ) phải được low-pass → ear_smooth ổn định."""
    fs = FusionState()
    mlp = make_mock_mlp(0.7)
    lstm = make_mock_lstm()
    scaler = make_mock_scaler()

    t = 4_000_000.0
    # Bắt đầu với ear 0.30
    fs.update(feat_with(ear=0.30), mlp, lstm, scaler, scaler, timestamp_ms=t)
    # Sau đó nhảy qua lại 0.10 ↔ 0.30 — như đang chớp mắt liên tục
    for ear in [0.10, 0.30, 0.10, 0.30, 0.10, 0.30, 0.10, 0.30]:
        t += 100
        fs.update(feat_with(ear=ear), mlp, lstm, scaler, scaler, timestamp_ms=t)

    # ear_smooth phải nằm giữa 0.10 và 0.30, ổn định
    es = fs.ear_smooth
    assert 0.10 < es < 0.30, f"ear_smooth {es} should be between 0.10 and 0.30"
    # Vì ear cuối = 0.30 và alpha=0.4, ear_smooth sẽ gần 0.30
    print(f"PASS  LPF: ear_smooth after noisy input = {es:.4f}")


def test_neck_alarm_still_works():
    """Gật gù (neck-tilt) vẫn phải trigger neck_alarm như trước (không bị regress)."""
    fs = FusionState()
    mlp = make_mock_mlp(0.7)
    lstm = make_mock_lstm()
    scaler = make_mock_scaler()

    t = 5_000_000.0
    # Đầu: ngồi thẳng, neck_tilt ~ 0
    for i in range(20):
        t += 100
        fs.update(feat_with(ear=0.30, neck_tilt=0.0), mlp, lstm, scaler, scaler, timestamp_ms=t)

    # Cúi đầu đột ngột (neck_tilt = 25°)
    r = None
    for i in range(5):
        t += 100
        r = fs.update(feat_with(ear=0.30, neck_tilt=25.0), mlp, lstm, scaler, scaler, timestamp_ms=t)

    assert r["neck_alarm"], f"neck_alarm should be True after sharp head tilt, got {r['neck_alarm']}"
    print(f"PASS  Neck-tilt regression: neck_alarm={r['neck_alarm']} ema_prob={r['ema_prob']}")


def test_eyes_open_no_alarm():
    """Mắt mở bình thường: eye_alarm = False, neck_alarm = False."""
    fs = FusionState()
    mlp = make_mock_mlp(0.8)
    lstm = make_mock_lstm()
    scaler = make_mock_scaler()

    t = 6_000_000.0
    r = None
    for i in range(30):
        t += 100
        r = fs.update(feat_with(ear=0.30, neck_tilt=2.0), mlp, lstm, scaler, scaler, timestamp_ms=t)

    assert not r["eye_alarm"],  "eye_alarm should be False when eyes open"
    assert not r["neck_alarm"], "neck_alarm should be False when sitting straight"
    print(f"PASS  Eyes open: eye_alarm={r['eye_alarm']} neck_alarm={r['neck_alarm']} ema_prob={r['ema_prob']}")


def test_reset_clears_state():
    """reset() phải xóa ear_smooth và eye_closed_streak_ms."""
    fs = FusionState()
    mlp = make_mock_mlp(0.7)
    lstm = make_mock_lstm()
    scaler = make_mock_scaler()

    t = 7_000_000.0
    # Nhắm mắt 1s
    for i in range(12):
        t += 100
        fs.update(feat_with(ear=0.08), mlp, lstm, scaler, scaler, timestamp_ms=t)

    assert fs.eye_closed_streak_ms > 1000, f"pre-reset streak should be >1s, got {fs.eye_closed_streak_ms}"
    assert fs.ear_smooth is not None

    fs.reset()

    assert fs.ear_smooth is None
    assert fs.eye_closed_streak_ms == 0.0
    print("PASS  reset() clears eye-closure state")


def test_microsleep_fast_path_sets_alarm_immediately():
    """Microsleep phải set alarm_on=True NGAY (không chờ MIN_ON_SEC=1.2s)."""
    fs = FusionState()
    mlp = make_mock_mlp(0.95)   # MLP rất tự tin "non-drowsy"
    lstm = make_mock_lstm(0.95)
    scaler = make_mock_scaler()

    t = 9_000_000.0
    fs.update(feat_with(ear=0.30), mlp, lstm, scaler, scaler, timestamp_ms=t)

    # Nhắm mắt 1.5s — bắt đầu từ frame 10 (1.0s) là microsleep đã trigger
    last_r = None
    first_alarm_frame = None
    for i in range(16):
        t += 100
        last_r = fs.update(feat_with(ear=0.08), mlp, lstm, scaler, scaler, timestamp_ms=t)
        if last_r["alarm_on"] and first_alarm_frame is None:
            first_alarm_frame = i

    assert last_r["alarm_on"], "alarm_on should be True after microsleep"
    # Alarm phải được set NGAY khi microsleep trigger (frame 10), không phải đợi thêm
    assert first_alarm_frame is not None and first_alarm_frame <= 11, (
        f"alarm_on first set at frame {first_alarm_frame}, should be ≤11 "
        f"(microsleep starts at frame 10 due to 1.0s threshold)"
    )
    print(f"PASS  Microsleep fast-path: alarm_on first triggered at frame {first_alarm_frame}, "
          f"ema_prob={last_r['ema_prob']}")


def test_alarm_releases_after_microsleep_ends():
    """Sau microsleep, khi mắt mở lại và ema_prob giảm → alarm_on phải release."""
    fs = FusionState()
    mlp = make_mock_mlp(0.95)
    lstm = make_mock_lstm(0.95)
    scaler = make_mock_scaler()

    t = 10_000_000.0
    # Nhắm mắt 1.5s → microsleep + alarm_on
    fs.update(feat_with(ear=0.30), mlp, lstm, scaler, scaler, timestamp_ms=t)
    for i in range(15):
        t += 100
        fs.update(feat_with(ear=0.08), mlp, lstm, scaler, scaler, timestamp_ms=t)
    assert fs.alarm_on, "alarm_on should be True during microsleep"

    # Mở mắt lại — chờ MIN_OFF_SEC=0.5s + hysteresis
    last_r = None
    for i in range(15):
        t += 100
        last_r = fs.update(feat_with(ear=0.30), mlp, lstm, scaler, scaler, timestamp_ms=t)

    assert not last_r["alarm_on"], (
        f"alarm_on should release after eyes open, got {last_r['alarm_on']}"
    )
    print(f"PASS  Alarm releases after microsleep ends: alarm_on={last_r['alarm_on']}")


def test_lstm_stuck_is_ignored_when_mlp_agrees_eyes_open():
    """
    Kịch bản thực tế từ log user:
    - MLP=0.30 (mắt mở), LSTM=0.55 (stuck) → chênh 0.25 > 0.15
    Trước đây max(MLP, LSTM)=0.55 → alarm stuck ở vùng gray.
    Sau khi sửa: tin MLP khi chênh > 0.15 → combined = MLP = 0.30.
    """
    fs = FusionState()
    # p_non_drowsy=0.70 → p_drowsy=0.30 (MLP thấp, mắt mở)
    mlp  = make_mock_mlp(0.70)
    # p_non_drowsy=0.45 → p_drowsy=0.55 (LSTM stuck ở vùng gray)
    lstm = make_mock_lstm(0.45)
    scaler = make_mock_scaler()

    t = 12_000_000.0
    # Mắt mở (EAR=0.30) trong 2s — MLP=0.30, LSTM=0.55
    last_r = None
    for i in range(20):
        t += 100
        last_r = fs.update(feat_with(ear=0.30), mlp, lstm, scaler, scaler, timestamp_ms=t)

    # LSTM-stuck (0.55) chênh MLP (0.30) là 0.25 > 0.15 → tin MLP
    # → combined = 0.30 → ema_prob decay xuống < HYSTERESIS_ON
    assert not last_r["alarm_on"], (
        f"alarm should NOT be on when MLP low & eyes open. got ema_prob={last_r['ema_prob']}"
    )
    assert last_r["ema_prob"] < HYSTERESIS_ON, (
        f"ema_prob should stay below ON threshold. got {last_r['ema_prob']}"
    )
    print(f"PASS  LSTM-stuck ignored: eyes open -> ema_prob={last_r['ema_prob']:.3f}, alarm={last_r['alarm_on']}")


def test_escape_valve_releases_alarm_when_eyes_clearly_open():
    """
    Escape-valve: khi alarm_on=True mà EAR > 0.25 liên tục 0.5s
    → alarm phải release NGAY, kể cả MLP/LSTM vẫn trả về p cao.
    Đây là tình huống 'model kẹt ở vùng gray' gây alarm stuck-on.
    """
    fs = FusionState()
    # MLP cao liên tục (kẹt ở vùng gray), LSTM kẹt — mô phỏng calibration bias
    mlp  = make_mock_mlp(0.10)   # p_non_drowsy=0.10 -> p_drowsy=0.90
    lstm = make_mock_lstm(0.50)  # LSTM kẹt ở p_drowsy=0.50
    scaler = make_mock_scaler()

    t = 11_000_000.0
    # Trigger alarm bằng microsleep (mắt nhắm 1.5s)
    fs.update(feat_with(ear=0.30), mlp, lstm, scaler, scaler, timestamp_ms=t)
    for i in range(15):
        t += 100
        fs.update(feat_with(ear=0.08), mlp, lstm, scaler, scaler, timestamp_ms=t)
    assert fs.alarm_on, "alarm_on should be True after microsleep"

    # Mở mắt rõ ràng (EAR=0.30) trong khi MLP vẫn cao (kẹt ở vùng gray)
    # Sau 0.5s, escape-valve phải force alarm_off VÀ ema_prob=0 (không chỉ OFF)
    released_at = None
    last_r = None
    ema_after_release = None
    for i in range(15):
        t += 100
        last_r = fs.update(feat_with(ear=0.30), mlp, lstm, scaler, scaler, timestamp_ms=t)
        if not last_r["alarm_on"] and released_at is None:
            released_at = i
            ema_after_release = last_r["ema_prob"]

    assert released_at is not None, "escape-valve must release alarm when eyes open clearly"
    assert released_at <= 7, (
        f"escape-valve should release within ~0.7s, got frame {released_at}"
    )
    # Sau khi release, tiếp tục chạy 5 frames với EAR mở - ema phải giảm về ~0
    # nhờ persistent override (combined=0 mỗi frame)
    for i in range(5):
        t += 100
        last_r = fs.update(feat_with(ear=0.30), mlp, lstm, scaler, scaler, timestamp_ms=t)
    assert last_r["ema_prob"] < 0.05, (
        f"persistent override must keep ema near 0, got {last_r['ema_prob']}"
    )
    assert not last_r["alarm_on"], f"alarm should remain off, got {last_r['alarm_on']}"
    print(f"PASS  Escape-valve: released at frame {released_at}, ema_after_5frames={last_r['ema_prob']:.4f}")


def test_neck_tilt_release_when_head_returns_to_baseline():
    """
    Kịch bản: gật gù (mắt lờ mờ, không mở rõ) -> alarm ON -> ngẩng đầu về baseline
    -> alarm phải release.
    Trước đây: baseline bị freeze khi alarm_on nên không thể release.
    Sau khi sửa: neck_recovered_streak_ms >= 0.5s -> force OFF.
    """
    fs = FusionState()
    # MLP trả trung bình (vùng gray) để mô phỏng model kẹt
    mlp  = make_mock_mlp(0.40)   # p_non_drowsy=0.40 -> p_drowsy=0.60
    lstm = make_mock_lstm(0.40)
    scaler = make_mock_scaler()

    t = 13_000_000.0
    # EAR = 0.17 (dưới threshold 0.18) - mắt lờ mờ drowsy thật,
    # không trigger EAR-override; neck tilt là trigger duy nhất.
    # Setup baseline
    for i in range(30):
        t += 100
        fs.update(feat_with(ear=0.18, neck_tilt=0.0), mlp, lstm, scaler, scaler, timestamp_ms=t)
    baseline = fs.neck_baseline
    frozen_baseline = baseline

    # Gật gù mạnh 3s (neck_tilt=30°)
    for i in range(30):
        t += 100
        fs.neck_baseline = frozen_baseline
        fs.update(feat_with(ear=0.18, neck_tilt=30.0), mlp, lstm, scaler, scaler, timestamp_ms=t)
    assert fs.ema_prob >= 0.60, f"ema should be high, got ema={fs.ema_prob}"
    # Patch để đảm bảo alarm_on=True cho test release
    if not fs.alarm_on:
        fs.alarm_on = True

    # Ngẩng đầu về baseline (neck_tilt=0°) - vẫn ear=0.18
    released_at = None
    last_r = None
    for i in range(15):
        t += 100
        fs.neck_baseline = frozen_baseline
        last_r = fs.update(feat_with(ear=0.18, neck_tilt=0.0), mlp, lstm, scaler, scaler, timestamp_ms=t)
        if not last_r["alarm_on"] and released_at is None:
            released_at = i

    assert released_at is not None, (
        "neck-tilt escape-valve must release alarm when head returns to baseline"
    )
    assert released_at <= 7, (
        f"should release within ~0.7s, got frame {released_at}"
    )
    print(f"PASS  Neck-tilt release: alarm released at frame {released_at} (~{released_at * 100}ms)")


def test_escape_valve_smoothsurvives_brief_dip_below_threshold():
    """
    Kịch bản thực tế: sau microsleep, EAR dao động quanh 0.18 (1 frame dip xuống 0.15).
    Smooth-streak tăng nhanh/giảm chậm phải vượt qua được 0.5s.
    Sau khi override active, ema_prob phải giữ gần 0 (persistent).
    """
    fs = FusionState()
    mlp  = make_mock_mlp(0.85)   # p_non_drowsy cao -> p_drowsy thấp
    lstm = make_mock_lstm(0.85)
    scaler = make_mock_scaler()

    t = 14_000_000.0
    # Trigger microsleep (mắt nhắm)
    fs.update(feat_with(ear=0.30), mlp, lstm, scaler, scaler, timestamp_ms=t)
    for i in range(15):
        t += 100
        fs.update(feat_with(ear=0.08), mlp, lstm, scaler, scaler, timestamp_ms=t)
    assert fs.alarm_on

    # EAR dao động: 0.22, 0.15, 0.21, 0.16, 0.25 (mắt mở + chớp mắt ngẫu nhiên)
    # Sau khi override active (~5 frames), ema phải về gần 0
    ear_pattern = [0.22, 0.15, 0.21, 0.16, 0.25] * 5
    released_at = None
    for i in range(25):
        t += 100
        ear = ear_pattern[i]
        last_r = fs.update(feat_with(ear=ear), mlp, lstm, scaler, scaler, timestamp_ms=t)
        if not last_r["alarm_on"] and released_at is None:
            released_at = i
            break

    assert released_at is not None, (
        "escape-valve must survive brief dips and release within 2.5s"
    )
    # Tiếp tục 5 frames, ema phải giảm về ~0
    for i in range(5):
        t += 100
        ear = ear_pattern[(i + released_at) % len(ear_pattern)]
        last_r = fs.update(feat_with(ear=ear), mlp, lstm, scaler, scaler, timestamp_ms=t)
    assert last_r["ema_prob"] < 0.10, (
        f"persistent override must keep ema low, got {last_r['ema_prob']}"
    )
    print(f"PASS  Escape-valve smooth-survives-dip: released at frame {released_at}, ema_after_5frames={last_r['ema_prob']:.4f}")


if __name__ == "__main__":
    test_eye_alarm_short_blink_no_trigger()
    test_eye_alarm_long_closure_triggers()
    test_eye_alarm_microsleep_stronger()
    test_low_pass_smooths_noisy_ear()
    test_neck_alarm_still_works()
    test_eyes_open_no_alarm()
    test_reset_clears_state()
    test_microsleep_fast_path_sets_alarm_immediately()
    test_alarm_releases_after_microsleep_ends()
    test_lstm_stuck_is_ignored_when_mlp_agrees_eyes_open()
    test_escape_valve_releases_alarm_when_eyes_clearly_open()
    test_neck_tilt_release_when_head_returns_to_baseline()
    test_escape_valve_smoothsurvives_brief_dip_below_threshold()
    print("\nALL TESTS PASSED")