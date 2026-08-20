"""
Drowsiness Scoring Engine - 5-level State Machine

Chuyển từ binary alarm (on/off) sang 5 trạng thái:
- NORMAL (0): Tỉnh táo
- FATIGUE (1): Mệt mỏi (dấu hiệu đầu tiên)
- DROWSY (2): Buồn ngủ (nguy cơ trung bình)
- MICROSLEEP (3): Ngủ gật (nguy cơ cao)
- CRITICAL (4): Nguy hiểm (không phục hồi sau cảnh báo)

Reference: PRD DMS-12, DMS-13, ALT-01~05, US1.2
"""
from __future__ import annotations

from enum import IntEnum


class DriverState(IntEnum):
    """5 trạng thái tài xế theo PRD."""
    NORMAL = 0
    FATIGUE = 1
    DROWSY = 2
    MICROSLEEP = 3
    CRITICAL = 4


class DrowsinessScorer:
    """
    Tính điểm drowsiness tổng hợp từ nhiều nguồn:
    - MLP/LSTM prob
    - PERCLOS
    - Eye closure streak
    - Yawn frequency
    - Neck-tilt/head-nod
    - Face visibility
    
    Output: DriverState (0-4) + drowsiness_score (0.0-1.0)
    """
    
    # Ngưỡng chuyển state mặc định (hysteresis giữa ON/OFF).
    #
    # H3: đây là CLASS attribute, chỉ dùng làm giá trị khởi tạo. Mỗi instance
    # copy sang `self.thresholds` trong __init__ và chỉ đọc/ghi bản copy đó.
    # Trước đây app.py ghi thẳng vào dict này → mọi instance (kể cả instance
    # tạo sau reset, và các test chạy chung process) đều bị đổi ngưỡng theo.
    THRESHOLDS = {
        # (threshold_on, threshold_off)
        DriverState.FATIGUE: (0.40, 0.35),      # Mệt mỏi nhẹ
        DriverState.DROWSY: (0.55, 0.50),       # Buồn ngủ rõ ràng
        DriverState.MICROSLEEP: (0.75, 0.70),   # Ngủ gật
        DriverState.CRITICAL: (0.85, 0.80),     # Nguy hiểm
    }
    
    # Thời gian tối thiểu ở state trước khi chuyển (debounce)
    MIN_STATE_DURATION_MS = {
        DriverState.NORMAL: 0,        # Có thể chuyển ngay
        DriverState.FATIGUE: 500,     # Ít nhất 0.5s
        DriverState.DROWSY: 800,      # Ít nhất 0.8s
        DriverState.MICROSLEEP: 0,    # Microsleep trigger ngay (nguy hiểm)
        DriverState.CRITICAL: 1000,   # Ít nhất 1s không phục hồi
    }
    
    # CRITICAL: nếu ở DROWSY/MICROSLEEP mà không phục hồi sau X giây
    CRITICAL_NO_RECOVERY_MS = 5000  # 5 giây không recovery → CRITICAL
    
    def __init__(self):
        self.current_state = DriverState.NORMAL
        self.state_entered_at_ms: float = 0.0
        self.last_update_ms: float = 0.0

        # Tracking recovery: thời gian ở state cao mà không giảm xuống
        self.time_in_high_state_ms: float = 0.0

        # H3: bản copy per-instance của ngưỡng — sửa runtime không rò rỉ ra
        # class attribute (và do đó không rò rỉ sang instance/test khác).
        self.thresholds: dict[DriverState, tuple[float, float]] = {
            state: tuple(values) for state, values in self.THRESHOLDS.items()
        }

    def set_threshold(self, state: DriverState, on: float, off: float) -> None:
        """HITL: đổi ngưỡng ON/OFF của một state cho RIÊNG instance này."""
        on = max(0.0, min(1.0, float(on)))
        off = max(0.0, min(on, float(off)))
        self.thresholds[state] = (on, off)

    def get_thresholds(self) -> dict[str, tuple[float, float]]:
        return {state.name: values for state, values in self.thresholds.items()}

    def update(
        self,
        timestamp_ms: float,
        p_mlp_drowsy: float,
        p_lstm_drowsy: float | None,
        perclos: float,
        eye_closed_streak_ms: float,
        neck_alarm: bool,
        eye_alarm: bool,
        yawn_alarm: bool,
        looking_away: bool = False,
        phone_suspected: bool = False,
        risk_multiplier: float = 1.0,
    ) -> tuple[DriverState, float]:
        """
        Cập nhật trạng thái drowsiness.
        
        Returns:
            (driver_state, drowsiness_score)
        """
        # ── Tính drowsiness score tổng hợp ──────────────────────────────
        score = self._compute_score(
            p_mlp_drowsy, p_lstm_drowsy, perclos,
            eye_closed_streak_ms, neck_alarm, eye_alarm, yawn_alarm,
            looking_away=looking_away,
            phone_suspected=phone_suspected,
        )
        # Context risk (speed / long drive) — clamp after boost
        if risk_multiplier and risk_multiplier != 1.0:
            score = max(0.0, min(1.0, score * float(risk_multiplier)))
        
        # ── State transition với hysteresis ─────────────────────────────
        dt_ms = timestamp_ms - self.last_update_ms if self.last_update_ms > 0 else 0.0
        time_in_current_state_ms = timestamp_ms - self.state_entered_at_ms
        
        new_state = self._transition_state(
            score, time_in_current_state_ms, dt_ms
        )
        
        # ── CRITICAL condition: không recovery sau 5s ở DROWSY/MICROSLEEP ──
        if self.current_state in (
            DriverState.DROWSY, DriverState.MICROSLEEP, DriverState.CRITICAL
        ):
            if score >= self.thresholds[DriverState.DROWSY][0]:
                self.time_in_high_state_ms += dt_ms
            else:
                self.time_in_high_state_ms = 0.0

            if (
                self.current_state != DriverState.CRITICAL
                and self.time_in_high_state_ms >= self.CRITICAL_NO_RECOVERY_MS
            ):
                new_state = DriverState.CRITICAL
        else:
            self.time_in_high_state_ms = 0.0

        # ── Update state ────────────────────────────────────────────────
        if new_state != self.current_state:
            self.current_state = new_state
            self.state_entered_at_ms = timestamp_ms

        self.last_update_ms = timestamp_ms

        return self.current_state, score

    def _compute_score(
        self,
        p_mlp: float,
        p_lstm: float | None,
        perclos: float,
        eye_closed_ms: float,
        neck_alarm: bool,
        eye_alarm: bool,
        yawn_alarm: bool,
        looking_away: bool = False,
        phone_suspected: bool = False,
    ) -> float:
        """
        Tính drowsiness score (0.0-1.0) từ tất cả signals.
        """
        # ── Base: MLP + LSTM (như fusion.py hiện tại) ──────────────────
        if p_lstm is None or abs(p_lstm - p_mlp) > 0.15:
            base = p_mlp
        else:
            base = max(p_mlp, p_lstm)
        
        # ── PERCLOS boost ──────────────────────────────────────────────
        if perclos >= 0.70:
            base = max(base, 0.75)
        elif perclos >= 0.50:
            base = max(base, 0.60)
        
        # ── Eye closure duration boost ─────────────────────────────────
        if eye_closed_ms >= 1200:
            base = max(base, 0.85)
        elif eye_closed_ms >= 800:
            base = max(base, 0.60)
        
        # ── Rule-based alarms boost ────────────────────────────────────
        if neck_alarm:
            base = max(base, 0.55)
        
        if eye_alarm:
            base = max(base, 0.60)
        
        if yawn_alarm:
            base = max(base, 0.50)

        # Distraction signals (outline) — elevate at least to FATIGUE/DROWSY band
        if looking_away:
            base = max(base, 0.45)
        if phone_suspected:
            base = max(base, 0.55)
        
        return max(0.0, min(1.0, base))
    
    def _transition_state(
        self,
        score: float,
        time_in_current_ms: float,
        dt_ms: float,
    ) -> DriverState:
        """
        Xác định state mới dựa trên score + hysteresis + debounce.
        
        Hysteresis: threshold khác nhau cho chuyển lên (ON) vs xuống (OFF)
        Debounce: phải ở state đủ lâu mới được chuyển
        """
        current = self.current_state
        
        # ── Check debounce: đã đủ lâu ở state hiện tại chưa? ───────────
        min_duration = self.MIN_STATE_DURATION_MS.get(current, 0)
        if time_in_current_ms < min_duration:
            return current  # chưa đủ lâu, giữ nguyên
        
        # ── Transition logic với hysteresis ────────────────────────────
        # Check từ cao xuống thấp (recovery)
        # CRITICAL chỉ hạ khi phục hồi rõ (dưới ngưỡng DROWSY OFF),
        # không dùng CRITICAL OFF — tránh flap ngay sau khi escalate vì score
        # vẫn nằm giữa DROWSY ON và CRITICAL OFF.
        # Fix #3: nếu score < 0.30 (recovery rất rõ), cho về NORMAL ngay để tránh
        # kẹt CRITICAL khi LSTM window vẫn trả cao dù đã tỉnh.
        if current == DriverState.CRITICAL:
            if score < 0.30:
                return DriverState.NORMAL
            _, drowsy_off = self.thresholds[DriverState.DROWSY]
            if score < drowsy_off:
                return DriverState.DROWSY
            return current  # sticky CRITICAL cho đến khi phục hồi thật

        if current == DriverState.MICROSLEEP:
            _, th_off = self.thresholds[DriverState.MICROSLEEP]
            if score < th_off:
                return DriverState.DROWSY

        if current == DriverState.DROWSY:
            _, th_off = self.thresholds[DriverState.DROWSY]
            if score < th_off:
                return DriverState.FATIGUE

        if current == DriverState.FATIGUE:
            _, th_off = self.thresholds[DriverState.FATIGUE]
            if score < th_off:
                return DriverState.NORMAL
        
        # Check từ thấp lên cao (degradation)
        if current == DriverState.NORMAL:
            th_on, _ = self.thresholds[DriverState.FATIGUE]
            if score >= th_on:
                return DriverState.FATIGUE
        
        if current == DriverState.FATIGUE:
            th_on, _ = self.thresholds[DriverState.DROWSY]
            if score >= th_on:
                return DriverState.DROWSY
        
        if current == DriverState.DROWSY:
            th_on, _ = self.thresholds[DriverState.MICROSLEEP]
            if score >= th_on:
                return DriverState.MICROSLEEP
        
        if current == DriverState.MICROSLEEP:
            th_on, _ = self.thresholds[DriverState.CRITICAL]
            if score >= th_on:
                return DriverState.CRITICAL
        
        # Không đủ điều kiện chuyển → giữ nguyên
        return current
    
    def reset(self):
        """Reset về NORMAL."""
        self.current_state = DriverState.NORMAL
        self.state_entered_at_ms = 0.0
        self.last_update_ms = 0.0
        self.time_in_high_state_ms = 0.0
    
    def get_state_name(self) -> str:
        """Trả về tên state dạng string."""
        return self.current_state.name
    
    def __repr__(self) -> str:
        return f"DrowsinessScorer(state={self.get_state_name()})"
