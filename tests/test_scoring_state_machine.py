"""
Unit tests cho Drowsiness Scoring Engine + State Machine 5 cấp.

Kiểm tra transition:
  NORMAL → FATIGUE → DROWSY → MICROSLEEP → CRITICAL
và recovery ngược lại.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.scoring import DrowsinessScorer, DriverState


def _step(scorer: DrowsinessScorer, t_ms: float, **kwargs):
    defaults = dict(
        p_mlp_drowsy=0.0,
        p_lstm_drowsy=None,
        perclos=0.0,
        eye_closed_streak_ms=0.0,
        neck_alarm=False,
        eye_alarm=False,
        yawn_alarm=False,
    )
    defaults.update(kwargs)
    return scorer.update(timestamp_ms=t_ms, **defaults)


def test_starts_normal():
    scorer = DrowsinessScorer()
    state, score = _step(scorer, 0, p_mlp_drowsy=0.1)
    assert state == DriverState.NORMAL
    assert score < 0.40
    print(f"PASS test_starts_normal: state={state.name}, score={score:.3f}")


def test_normal_to_fatigue():
    scorer = DrowsinessScorer()
    state, _ = _step(scorer, 0, p_mlp_drowsy=0.45)
    assert state == DriverState.FATIGUE
    print(f"PASS test_normal_to_fatigue: state={state.name}")


def test_fatigue_to_drowsy_after_debounce():
    scorer = DrowsinessScorer()
    # Vào FATIGUE
    _step(scorer, 0, p_mlp_drowsy=0.45)
    # Chưa đủ 500ms → vẫn FATIGUE dù score cao
    state, _ = _step(scorer, 200, p_mlp_drowsy=0.60)
    assert state == DriverState.FATIGUE
    # Đủ debounce → DROWSY
    state, _ = _step(scorer, 600, p_mlp_drowsy=0.60)
    assert state == DriverState.DROWSY
    print(f"PASS test_fatigue_to_drowsy_after_debounce: state={state.name}")


def test_drowsy_to_microsleep():
    scorer = DrowsinessScorer()
    _step(scorer, 0, p_mlp_drowsy=0.45)          # → FATIGUE
    _step(scorer, 600, p_mlp_drowsy=0.60)        # → DROWSY (sau debounce)
    # DROWSY cần >= 800ms trước khi lên MICROSLEEP
    state, _ = _step(scorer, 1000, p_mlp_drowsy=0.80)
    assert state == DriverState.DROWSY
    state, _ = _step(scorer, 1500, p_mlp_drowsy=0.80)
    assert state == DriverState.MICROSLEEP
    print(f"PASS test_drowsy_to_microsleep: state={state.name}")


def test_eye_closure_boosts_to_microsleep_score():
    scorer = DrowsinessScorer()
    _, score = _step(scorer, 0, p_mlp_drowsy=0.2, eye_closed_streak_ms=1200)
    assert score >= 0.85
    print(f"PASS test_eye_closure_boosts_to_microsleep_score: score={score:.3f}")


def test_perclos_boost():
    scorer = DrowsinessScorer()
    _, score_hi = _step(scorer, 0, p_mlp_drowsy=0.2, perclos=0.75)
    assert score_hi >= 0.75
    scorer.reset()
    _, score_mid = _step(scorer, 0, p_mlp_drowsy=0.2, perclos=0.55)
    assert score_mid >= 0.60
    print(f"PASS test_perclos_boost: hi={score_hi:.3f}, mid={score_mid:.3f}")


def test_critical_after_no_recovery():
    """Ở DROWSY/MICROSLEEP >= 5s không phục hồi → CRITICAL."""
    scorer = DrowsinessScorer()
    _step(scorer, 0, p_mlp_drowsy=0.45)           # FATIGUE
    _step(scorer, 600, p_mlp_drowsy=0.60)         # DROWSY
    # Giữ score cao trong >5s
    state = DriverState.DROWSY
    t = 600
    for _ in range(60):
        t += 100
        state, _ = _step(scorer, t, p_mlp_drowsy=0.65)
    assert state == DriverState.CRITICAL
    print(f"PASS test_critical_after_no_recovery: state={state.name}")


def test_recovery_drowsy_to_fatigue_to_normal():
    scorer = DrowsinessScorer()
    _step(scorer, 0, p_mlp_drowsy=0.45)
    _step(scorer, 600, p_mlp_drowsy=0.60)         # DROWSY
    # Giữ đủ debounce rồi hạ score
    state, _ = _step(scorer, 1500, p_mlp_drowsy=0.40)  # < DROWSY off(0.50) → FATIGUE
    assert state == DriverState.FATIGUE
    state, _ = _step(scorer, 2200, p_mlp_drowsy=0.20)  # < FATIGUE off(0.35) → NORMAL
    assert state == DriverState.NORMAL
    print(f"PASS test_recovery_drowsy_to_fatigue_to_normal: state={state.name}")


def test_hysteresis_no_flap():
    """Score nằm giữa ON/OFF threshold → không dao động."""
    scorer = DrowsinessScorer()
    _step(scorer, 0, p_mlp_drowsy=0.45)  # → FATIGUE (ON=0.40)
    # Score 0.37: > OFF(0.35) nhưng < ON DROWSY(0.55) → giữ FATIGUE
    state, _ = _step(scorer, 600, p_mlp_drowsy=0.37)
    assert state == DriverState.FATIGUE
    print(f"PASS test_hysteresis_no_flap: state={state.name}")


def test_reset():
    scorer = DrowsinessScorer()
    _step(scorer, 0, p_mlp_drowsy=0.45)
    scorer.reset()
    assert scorer.current_state == DriverState.NORMAL
    print("PASS test_reset")


def test_full_escalation_chain():
    """Chuỗi đầy đủ NORMAL→…→CRITICAL rồi recovery."""
    scorer = DrowsinessScorer()
    states = []

    def record(t, **kw):
        s, _ = _step(scorer, t, **kw)
        states.append(s)
        return s

    record(0, p_mlp_drowsy=0.10)                 # NORMAL
    record(100, p_mlp_drowsy=0.45)               # FATIGUE
    record(700, p_mlp_drowsy=0.60)               # DROWSY
    record(1600, p_mlp_drowsy=0.80)              # MICROSLEEP
    # CRITICAL qua no-recovery
    t = 1600
    for _ in range(55):
        t += 100
        record(t, p_mlp_drowsy=0.80)
    assert DriverState.CRITICAL in states
    # Recovery: CRITICAL sticky đến khi score < DROWSY OFF (0.50)
    s = record(t + 2000, p_mlp_drowsy=0.40)
    assert s == DriverState.DROWSY
    print(f"PASS test_full_escalation_chain: final={s.name}, saw_critical=True")


if __name__ == "__main__":
    test_starts_normal()
    test_normal_to_fatigue()
    test_fatigue_to_drowsy_after_debounce()
    test_drowsy_to_microsleep()
    test_eye_closure_boosts_to_microsleep_score()
    test_perclos_boost()
    test_critical_after_no_recovery()
    test_recovery_drowsy_to_fatigue_to_normal()
    test_hysteresis_no_flap()
    test_reset()
    test_full_escalation_chain()
    print("\nAll scoring tests passed.")
