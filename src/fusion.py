"""
FusionState: MLP (per-frame) + LSTM (30-frame window) + neck-tilt rule
→ EMA smoothing → hysteresis debounce → alarm_on flag.

Quy ước model: output = P(Non-Drowsy), nên p_drowsy = 1 - output.
"""
from __future__ import annotations

import math
import time
from collections import deque

import numpy as np

from .perclos import PERCLOSTracker
from .runtime_profile import get_runtime_profile
from .scoring import DrowsinessScorer
from .frequency import EventFrequencyCounter
from .looking_away import LookingAwayDetector

# ── Hyperparameters (từ notebook) ────────────────────────────────────────────
WINDOW_SIZE         = 30     # số frame trong cửa sổ LSTM
EMA_ALPHA           = 0.3    # giảm từ 0.5 → 0.3 (ổn định hơn, tránh flap)
NECK_BASELINE_ALPHA = 0.005  # freeze chậm — baseline KHÔNG drift theo cú gật (5x chậm hơn trước)
NECK_TILT_ALARM_DEG = 15.0   # threshold cho alarm (tăng từ 8.0)
HYSTERESIS_ON       = 0.65   # tăng từ 0.55 → 0.65 (model thật trả ~0.59 cho eyes-open)
HYSTERESIS_OFF      = 0.35   # tăng từ 0.30 → 0.35
MIN_ON_SEC          = 0.5    # cú gật phải kéo dài >= 0.5s mới trigger (tăng từ 0.15)
MIN_OFF_SEC         = 0.5

# Pitch-based head nod (face solvePnP — không cần pose vai, phù hợp edge/IR cam)
PITCH_BASELINE_ALPHA = 0.005
PITCH_NOD_PEAK_DEG   = 12.0   # thấp hơn neck pose vì pitch nhạy hơn khi cúi đầu
PITCH_NOD_CURRENT_DEG = 8.0

# Eye-closure rule (mô phỏng neck-tilt rule, phản ứng nhanh với sleepy eye)
EAR_LPF_ALPHA       = 0.5    # low-pass filter cho ear_avg trước khi feed model + rule
EYE_CLOSED_THRESH   = 0.16   # giảm từ 0.18 → 0.16 (chỉ trigger khi thực sự nhắm, tránh EAR=0.18 bị coi là microsleep)
EYE_CLOSED_ON_SEC   = 0.8    # mắt nhắm liên tục >= 0.8s  → ép combined >= HYSTERESIS_ON
EYE_CLOSED_HARD_SEC = 1.2    # mắt nhắm liên tục >= 1.2s (microsleep thật) → ép combined >= 0.85

# Yawn-detection rule (tách ngáp vs nói chuyện)
# Phân biệt ngáp vs nói chuyện dựa trên:
#   1. MAR thô (mouth aspect ratio): ngáp ≈ 0.5-0.9, nói ≈ 0.3-0.6
#   2. Thời gian duy trì: ngáp 2-6s, nói chỉ 100-500ms
#   3. Cooldown: không trigger 2 lần ngáp trong 10s
YAWN_MAR_THRESH         = 0.40   # MAR phải vượt mới coi là "mở miệng rộng" (giảm từ 0.45)
YAWN_ASPECT_MIN         = 0.50   # aspect (dọc/ngang) ≥ 0.50 → miệng tròn/dọc → ngáp (giảm từ 0.65)
                                 # aspect < 0.50 → miệng dẹt ngang → nói
YAWN_DURATION_MIN_SEC   = 1.2    # duy trì MAR+aspect ≥ 1.2s → ngáp thật (tăng từ 0.8)
                                 # (BuzzFeed yawn video: yawn 1.1-2.4s, speech ≤ 0.6s)
YAWN_COOLDOWN_SEC       = 10.0   # không trigger 2 lần trong 10s

# 6 features đưa vào LSTM buffer (theo thứ tự model được train)
LSTM_FEAT_COLS = ("ear_avg", "mar", "pitch", "yaw", "roll", "neck_tilt")

# 9 features đưa vào MLP (theo thứ tự model được train)
MLP_FEAT_COLS = ("ear_left", "ear_right", "ear_avg", "mar",
                 "pitch", "yaw", "roll", "neck_tilt", "has_neck_tilt")


class FusionState:
    """Stateful fusion — giữ buffer LSTM, EMA, và debounce timer."""

    def __init__(self):
        self.ema_prob:         float | None = None
        self.neck_baseline:    float | None = None
        self.feature_buffer:   deque        = deque(maxlen=WINDOW_SIZE)
        self.alarm_on:         bool         = False
        self.last_ts_ms:       float | None = None
        self.time_above_on_ms: float        = 0.0
        self.time_below_off_ms:float        = 0.0
        # Eye-closure rule state
        self.ear_smooth:            float | None = None   # low-pass ear_avg
        self.eye_closed_streak_ms:  float        = 0.0    # thời gian mắt liên tục nhắm
        # Escape-valve: mắt mở rõ ràng liên tục (tránh alarm stuck-on)
        self.eyes_open_streak_ms:   float        = 0.0
        # Neck-tilt escape-valve: neck_tilt đã về sát baseline liên tục
        self.neck_recovered_streak_ms: float     = 0.0
        # Peak detector: gật gù ngắn ~200ms, delta nhỏ nhưng nhọn
        # Track max |neck_tilt - baseline| trong 600ms gần nhất
        self.neck_peak_buffer: deque = deque(maxlen=30)  # ~600ms @30fps
        # Pitch nod (face-only fallback khi pose mất ở độ phân giải thấp)
        self.pitch_baseline:    float | None = None
        self.pitch_peak_buffer: deque        = deque(maxlen=30)
        # YawnDetector state
        self.yawn_state: str = "IDLE"          # IDLE | OPENING | CONFIRMED | COOLDOWN
        self.yawn_open_started_ms: float | None = None
        self.yawn_last_trigger_ms:  float = -1e9  # đã trigger lần cuối ở timestamp nào
        
        # PERCLOS tracker (L1)
        self.perclos_tracker: PERCLOSTracker = PERCLOSTracker(
            window_sec=30.0,
            eye_closed_threshold=EYE_CLOSED_THRESH
        )
        # Drowsiness scoring state machine (L2)
        self.scorer: DrowsinessScorer = DrowsinessScorer()
        # Frequency counters (L10 / L11)
        self.head_nod_counter = EventFrequencyCounter(window_sec=60.0)
        self.yawn_counter = EventFrequencyCounter(window_sec=60.0)
        # Looking-away (L18)
        self.looking_away = LookingAwayDetector()
        # Optional context risk (set from app before/after update)
        self.risk_multiplier: float = 1.0
        self.phone_suspected: bool = False

    def reset(self):
        self.__init__()

    def update(self, feat: dict, mlp_model, lstm_model,
               mlp_scaler, lstm_scaler,
               timestamp_ms: float | None = None) -> dict:
        """
        Nhận một dict features từ landmarks.extract_features(),
        trả về dict kết quả:
          p_mlp_drowsy, p_lstm_drowsy, neck_alarm, ema_prob, alarm_on
        """
        if timestamp_ms is None:
            timestamp_ms = time.time() * 1000.0

        # ── dt_ms (tính sớm vì cả eye-closure rule và debounce đều cần) ──
        dt_ms = 0.0
        if self.last_ts_ms is not None:
            dt_ms = max(0.0, timestamp_ms - self.last_ts_ms)

        # ── EAR low-pass filter (giảm nhiễu trước khi feed model + rule) ──
        raw_ear = feat.get("ear_avg", float("nan"))
        if math.isnan(raw_ear):
            ear_smooth = self.ear_smooth  # giữ giá trị cũ nếu frame hiện tại thiếu
        else:
            if self.ear_smooth is None:
                ear_smooth = raw_ear
            else:
                ear_smooth = EAR_LPF_ALPHA * raw_ear + (1 - EAR_LPF_ALPHA) * self.ear_smooth
        self.ear_smooth = ear_smooth

        # Ghi đè giá trị EAR đã được low-pass vào feat để MLP/LSTM nhận tín hiệu ổn định
        feat = dict(feat)
        if ear_smooth is not None:
            feat["ear_left"]  = ear_smooth
            feat["ear_right"] = ear_smooth
            feat["ear_avg"]   = ear_smooth

        # ── PERCLOS Tracker (L1 - new) ────────────────────────────────────
        perclos_ratio = 0.0
        if ear_smooth is not None:
            perclos_ratio = self.perclos_tracker.update(timestamp_ms, ear_smooth)
        
        # ── MLP per-frame ──────────────────────────────────────────────────
        has_neck = 0 if math.isnan(feat.get("neck_tilt", float("nan"))) else 1
        mlp_row = np.array(
            [feat.get(c, 0.0) if c != "has_neck_tilt" else has_neck
             for c in MLP_FEAT_COLS],
            dtype=np.float32,
        )
        mlp_row = np.nan_to_num(mlp_row, nan=0.0)
        x_mlp = mlp_scaler.transform(mlp_row.reshape(1, -1))
        p_non_drowsy_mlp = float(mlp_model.predict(x_mlp, verbose=0)[0, 0])
        p_mlp_drowsy = 1.0 - p_non_drowsy_mlp

        # ── Neck-tilt alarm ────────────────────────────────────────────────
        neck_tilt = feat.get("neck_tilt", float("nan"))
        neck_alarm = False
        if not math.isnan(neck_tilt):
            if self.neck_baseline is None:
                self.neck_baseline = neck_tilt
            elif not self.alarm_on:
                self.neck_baseline = (
                    (1 - NECK_BASELINE_ALPHA) * self.neck_baseline
                    + NECK_BASELINE_ALPHA * neck_tilt
                )

            # Peak detection: track delta lớn nhất trong window ~600ms
            # để bắt cú gật ngắn 200-500ms dù EMA/baseline chưa kịp phản ứng
            current_delta = abs(neck_tilt - self.neck_baseline)
            self.neck_peak_buffer.append(current_delta)
            peak_delta = max(self.neck_peak_buffer) if self.neck_peak_buffer else 0.0

            # ── Trigger: cú gật dứt khoát ──
            # Cần peak > 18° VÀ current > 12° (cú gật sâu & rõ ràng)
            # Tăng threshold để tránh false-positive từ head movement nhẹ
            if peak_delta > 18.0 and current_delta > 12.0:
                neck_alarm = True

            # ── CLEAR peak buffer khi neck_tilt đã về gần baseline ──
            # Nếu delta hiện tại < 6° (về gần baseline) → coi như cú gật đã kết thúc
            # → xóa buffer để tránh alarm kéo dài sau khi ngẩng đầu
            if current_delta < 6.0 and not self.alarm_on:
                self.neck_peak_buffer.clear()

        # ── Pitch-based head nod (face-only, edge-safe) ───────────────────
        # Pose vai thường mất ở 256×192 / webcam góc hẹp → neck_tilt = NaN.
        # Pitch từ solvePnP (face mesh) vẫn ổn định — chuẩn DMS trên ô tô.
        use_pitch_nod = get_runtime_profile().get("use_pitch_nod", True)
        pitch = feat.get("pitch", float("nan"))
        if use_pitch_nod and not math.isnan(pitch):
            if self.pitch_baseline is None:
                self.pitch_baseline = pitch
            elif not self.alarm_on:
                self.pitch_baseline = (
                    (1 - PITCH_BASELINE_ALPHA) * self.pitch_baseline
                    + PITCH_BASELINE_ALPHA * pitch
                )
            pitch_delta = abs(pitch - self.pitch_baseline)
            self.pitch_peak_buffer.append(pitch_delta)
            peak_pitch = max(self.pitch_peak_buffer) if self.pitch_peak_buffer else 0.0
            if peak_pitch > PITCH_NOD_PEAK_DEG and pitch_delta > PITCH_NOD_CURRENT_DEG:
                neck_alarm = True
            if pitch_delta < 4.0 and not self.alarm_on:
                self.pitch_peak_buffer.clear()

        # Neck-tilt release: khi alarm đang bật do neck-tilt, nếu neck_tilt
        # đã về sát baseline (< 8°) liên tục 0.5s → force tắt alarm.
        if self.alarm_on and not math.isnan(neck_tilt) and self.neck_baseline is not None:
            if abs(neck_tilt - self.neck_baseline) < 8.0:
                self.neck_recovered_streak_ms += dt_ms
            else:
                self.neck_recovered_streak_ms = 0.0
        else:
            self.neck_recovered_streak_ms = 0.0

        if self.alarm_on and self.neck_recovered_streak_ms >= 0.5 * 1000:
            self.alarm_on = False
            self.ema_prob = 0.0
            self.time_above_on_ms   = 0.0
            self.time_below_off_ms  = 0.0
            self.neck_peak_buffer.clear()

        # Eye-closure rule (song song neck-tilt, phản ứng nhanh) ───────
        eye_alarm = False
        eye_microsleep = False
        if ear_smooth is not None:
            if ear_smooth < EYE_CLOSED_THRESH:
                self.eye_closed_streak_ms += dt_ms
            else:
                self.eye_closed_streak_ms = 0.0
            if self.eye_closed_streak_ms >= EYE_CLOSED_HARD_SEC * 1000:
                eye_alarm = True
                eye_microsleep = True
            elif self.eye_closed_streak_ms >= EYE_CLOSED_ON_SEC * 1000:
                eye_alarm = True

        # YawnDetector state machine ───────────────────────────────────────
        yawn_alarm = False
        mar = feat.get("mar", float("nan"))
        mouth_aspect = feat.get("mouth_aspect", float("nan"))
        is_yawn_posture = (
            not math.isnan(mar)
            and not math.isnan(mouth_aspect)
            and mar > YAWN_MAR_THRESH
            and mouth_aspect > YAWN_ASPECT_MIN
        )

        if self.yawn_state == "IDLE":
            if is_yawn_posture:
                self.yawn_state = "OPENING"
                self.yawn_open_started_ms = timestamp_ms

        elif self.yawn_state == "OPENING":
            if is_yawn_posture:
                # MAR+aspect vẫn cao — kiểm tra đủ duration chưa
                if self.yawn_open_started_ms is not None:
                    elapsed = timestamp_ms - self.yawn_open_started_ms
                    if elapsed >= YAWN_DURATION_MIN_SEC * 1000:
                        self.yawn_state = "CONFIRMED"
                        yawn_alarm = True
            else:
                # MAR xuống trước duration → chỉ là nói/cười/nháy mắt
                self.yawn_state = "IDLE"
                self.yawn_open_started_ms = None

        elif self.yawn_state == "CONFIRMED":
            # Chờ MAR xuống < 0.30 (miệng đóng) rồi mới vào cooldown
            if not math.isnan(mar) and mar < 0.30:
                self.yawn_state = "COOLDOWN"
                self.yawn_last_trigger_ms = timestamp_ms

        elif self.yawn_state == "COOLDOWN":
            if (timestamp_ms - self.yawn_last_trigger_ms) >= YAWN_COOLDOWN_SEC * 1000:
                self.yawn_state = "IDLE"

        # ── LSTM trên cửa sổ gần nhất (có thể tắt ở edge profile) ─────────
        enable_lstm = get_runtime_profile().get("enable_lstm", True)
        self.feature_buffer.append(
            [feat.get(c, float("nan")) for c in LSTM_FEAT_COLS]
        )
        p_lstm_drowsy = None
        if enable_lstm and len(self.feature_buffer) == WINDOW_SIZE:
            import pandas as pd
            seq = pd.DataFrame(
                list(self.feature_buffer), columns=list(LSTM_FEAT_COLS)
            ).ffill().bfill().values.astype(np.float32)
            n_feat = seq.shape[1]
            seq_scaled = lstm_scaler.transform(seq).reshape(1, WINDOW_SIZE, n_feat)
            p_non_drowsy_lstm = float(
                lstm_model.predict(seq_scaled, verbose=0)[0, 0]
            )
            p_lstm_drowsy = 1.0 - p_non_drowsy_lstm

        # ── Fusion ────────────────────────────────────────────────────────
        # LSTM hay bị "kẹt" ở ~0.55 (model kém reactive với EAR).
        # Chỉ dùng LSTM khi nó đồng thuận với MLP (chênh lệch < 0.15)
        # — nếu MLP thấp nhưng LSTM cao bất thường → tin MLP hơn.
        if p_lstm_drowsy is None or abs(p_lstm_drowsy - p_mlp_drowsy) > 0.15:
            combined = p_mlp_drowsy
        else:
            combined = max(p_mlp_drowsy, p_lstm_drowsy)
        if neck_alarm:
            combined = max(combined, HYSTERESIS_ON)
        if yawn_alarm:
            # Ngáp kéo dài ≥ 1.5s → ép combined, nhưng không bypass debounce
            combined = max(combined, HYSTERESIS_ON)
        if eye_microsleep:
            # Microsleep that (>=1.0s) → set trực tiếp, không chờ EMA
            combined = max(combined, 0.85)
        elif eye_alarm:
            # Mat nham keo dai (>=0.6s) → bump qua hysteresis ON
            combined = max(combined, HYSTERESIS_ON)

        # ── Persistent EAR-override (escape-valve) ────────────────────────
        # Nếu mắt đã mở rõ ràng >= 0.5s liên tục → vô hiệu hóa hoàn toàn
        # alarm (combined=0 ngay từ đầu). Persistent miễn là mắt còn mở
        # → MLP-stuck không kéo alarm bật lại. Reset khi EAR < threshold.
        ear_open_overridden = (
            ear_smooth is not None and ear_smooth > EYE_CLOSED_THRESH
        )
        if ear_open_overridden:
            self.eyes_open_streak_ms = min(
                self.eyes_open_streak_ms + dt_ms,
                2.0 * 1000,
            )
        else:
            self.eyes_open_streak_ms = max(0.0, self.eyes_open_streak_ms - dt_ms * 0.5)

        # ── Gật gù: neck_alarm là tín hiệu pose trực tiếp, KHÔNG bị EAR-open đè ──
        # Trước đây: khi gật, mắt vẫn mở → eyes_open_streak tăng → combined*=0.5
        # → KHÔNG đủ vượt hysteresis → không trigger alarm lúc gật (sai!)
        # Fix: nếu neck_alarm=True, KHÔNG áp dụng EAR-open suppression.
        # Vẫn cho alarm chạy bình thường qua hysteresis + debounce.
        if self.eyes_open_streak_ms >= 0.5 * 1000:
            if not neck_alarm:
                # Chỉ suppress khi KHÔNG có gật gù
                combined = 0.0
                if self.alarm_on:
                    self.alarm_on = False
                    self.time_above_on_ms   = 0.0
                    self.time_below_off_ms  = 0.0
                self.ema_prob = 0.0
        else:
            if self.eyes_open_streak_ms >= 0.3 * 1000 and not neck_alarm:
                combined *= 0.1

        # ── EMA ──────────────────────────────────────────────────────────
        if self.ema_prob is None:
            self.ema_prob = combined
        else:
            self.ema_prob = (1 - EMA_ALPHA) * self.ema_prob + EMA_ALPHA * combined

        # Eye microsleep: ép ema_prob lên cao ngay để vượt hysteresis + debounce
        if eye_microsleep:
            self.ema_prob = max(self.ema_prob, 0.85)
            # Fast-path: microsleep thật → cảnh báo tức thì (bypass MIN_ON_SEC)
            self.alarm_on = True
            self.eyes_open_streak_ms = 0.0

        # ── Debounce theo thời gian thực ─────────────────────────────────
        self.last_ts_ms = timestamp_ms

        if self.ema_prob >= HYSTERESIS_ON:
            self.time_above_on_ms  += dt_ms
            self.time_below_off_ms  = 0.0
        elif self.ema_prob <= HYSTERESIS_OFF:
            self.time_below_off_ms += dt_ms
            self.time_above_on_ms   = 0.0
        else:
            self.time_above_on_ms   = 0.0
            self.time_below_off_ms  = 0.0

        if not self.alarm_on and self.time_above_on_ms >= MIN_ON_SEC * 1000:
            self.alarm_on = True
        elif self.alarm_on and self.time_below_off_ms >= MIN_OFF_SEC * 1000:
            self.alarm_on = False

        # ── Looking away (L18) ────────────────────────────────────────────
        yaw_v = feat.get("yaw", float("nan"))
        yaw_for_la = None if (isinstance(yaw_v, float) and math.isnan(yaw_v)) else yaw_v
        la = self.looking_away.update(timestamp_ms, yaw_for_la)

        # ── Drowsiness Scoring Engine (L2) ────────────────────────────────
        # State machine 5 cấp chạy song song với alarm_on (binary).
        # Không ép alarm_on từ scorer — tránh phá escape-valve / neck-release.
        driver_state, drowsiness_score = self.scorer.update(
            timestamp_ms=timestamp_ms,
            p_mlp_drowsy=p_mlp_drowsy,
            p_lstm_drowsy=p_lstm_drowsy,
            perclos=perclos_ratio,
            eye_closed_streak_ms=self.eye_closed_streak_ms,
            neck_alarm=neck_alarm,
            eye_alarm=eye_alarm,
            yawn_alarm=yawn_alarm,
            looking_away=la["looking_away"],
            phone_suspected=self.phone_suspected,
            risk_multiplier=self.risk_multiplier,
        )

        # ── Frequency counters (L10 / L11) ────────────────────────────────
        head_nod_count = self.head_nod_counter.update(timestamp_ms, neck_alarm)
        yawn_count = self.yawn_counter.update(timestamp_ms, yawn_alarm)

        return {
            "p_mlp_drowsy":      round(p_mlp_drowsy, 4),
            "p_lstm_drowsy":     round(p_lstm_drowsy, 4) if p_lstm_drowsy is not None else None,
            "neck_alarm":        neck_alarm,
            "eye_alarm":         eye_alarm,
            "yawn_alarm":        yawn_alarm,
            "ema_prob":          round(self.ema_prob, 4),
            "alarm_on":          self.alarm_on,
            "perclos":           round(perclos_ratio, 4),
            "perclos_ratio":     round(perclos_ratio, 4),
            "drowsiness_state":  driver_state.name,
            "drowsiness_score":  round(drowsiness_score, 4),
            "alert_level":       int(driver_state),
            "head_nod_count_window": head_nod_count,
            "yawn_count_window": yawn_count,
            **la,
            "phone_suspected":   self.phone_suspected,
            "risk_multiplier":   self.risk_multiplier,
        }
