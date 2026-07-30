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

# ── Hyperparameters (từ notebook) ────────────────────────────────────────────
WINDOW_SIZE         = 30     # số frame trong cửa sổ LSTM
EMA_ALPHA           = 0.5    # nhanh — bắt cú gật thoáng qua 200-300ms
NECK_BASELINE_ALPHA = 0.005  # freeze chậm — baseline KHÔNG drift theo cú gật (5x chậm hơn trước)
NECK_TILT_ALARM_DEG = 8.0    # nhạy cao — gật gù thật chỉ lệch 8-15°
HYSTERESIS_ON       = 0.55   # bắt tín hiệu yếu
HYSTERESIS_OFF      = 0.30
MIN_ON_SEC          = 0.15   # cú gật 150ms phải trigger (gật gù ngắn)
MIN_OFF_SEC         = 0.5

# Eye-closure rule (mô phỏng neck-tilt rule, phản ứng nhanh với sleepy eye)
EAR_LPF_ALPHA       = 0.5    # low-pass filter cho ear_avg trước khi feed model + rule
EYE_CLOSED_THRESH   = 0.20   # ngưỡng mắt nhắm (sau khi low-pass)
EYE_CLOSED_ON_SEC   = 0.6    # mắt nhắm liên tục >= 0.6s  → ép combined >= HYSTERESIS_ON
EYE_CLOSED_HARD_SEC = 1.0    # mắt nhắm liên tục >= 1.0s (microsleep thật) → ép combined >= 0.85

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

            # Instant trigger: peak hiện tại > 8° → gật gù dứt khoát
            if peak_delta > NECK_TILT_ALARM_DEG:
                neck_alarm = True

        # Neck-tilt release: khi alarm đang bật do neck-tilt, nếu neck_tilt
        # đã về sát baseline (< 5°) liên tục 0.5s → force tắt alarm.
        # (Tránh stuck-on khi người dùng đã ngẩng đầu nhưng decoder network
        # kẹt ở vùng gray; baseline bị freeze nên cần cờ riêng.)
        if self.alarm_on and not math.isnan(neck_tilt) and self.neck_baseline is not None:
            if abs(neck_tilt - self.neck_baseline) < 5.0:
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

        # ── LSTM trên cửa sổ gần nhất ─────────────────────────────────────
        self.feature_buffer.append(
            [feat.get(c, float("nan")) for c in LSTM_FEAT_COLS]
        )
        p_lstm_drowsy = None
        if len(self.feature_buffer) == WINDOW_SIZE:
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

        if self.eyes_open_streak_ms >= 0.5 * 1000:
            # Persistent override: mắt mở >= 0.5s → combined=0 và alarm=OFF
            # ─── NGOẠI TRỪ neck_alarm (gật gù là tín hiệu pose trực tiếp, không được đè)
            if neck_alarm:
                # Gật gù mà mắt mở → vẫn giữ tín hiệu, nhưng giảm nhẹ (giảm 50%)
                # để EAR-open có trọng số phản đối hợp lý
                combined *= 0.5
            else:
                combined = 0.0
                if self.alarm_on:
                    self.alarm_on = False
                    self.time_above_on_ms   = 0.0
                    self.time_below_off_ms  = 0.0
                # Reset EMA mỗi frame miễn là mắt còn mở → MLP/LSTM-stuck
                # không kéo alarm bật lại frame kế tiếp
                self.ema_prob = 0.0
        else:
            # Trong lúc eyes_open_streak đang buildup (chưa đủ 0.5s),
            # vẫn giảm combined nặng để EMA tụt nhanh — nhưng KHÔNG đè neck_alarm
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

        return {
            "p_mlp_drowsy":  round(p_mlp_drowsy, 4),
            "p_lstm_drowsy": round(p_lstm_drowsy, 4) if p_lstm_drowsy is not None else None,
            "neck_alarm":    neck_alarm,
            "eye_alarm":     eye_alarm,
            "ema_prob":      round(self.ema_prob, 4),
            "alarm_on":      self.alarm_on,
        }
