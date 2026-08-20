"""
PERCLOS (Percentage of Eye Closure) Tracker

Tính tỷ lệ % thời gian mắt nhắm trong rolling window 30 giây.
PERCLOS là chỉ số quan trọng để đánh giá mức độ buồn ngủ theo chuẩn nghiên cứu.

Reference: PRD DMS-06, EYE-04
"""
from __future__ import annotations

from collections import deque
from typing import Tuple


class PERCLOSTracker:
    """
    Theo dõi PERCLOS (Percentage of Eye Closure) trong rolling window.
    
    PERCLOS = (tổng thời gian mắt nhắm) / (tổng thời gian window)
    
    Window: 30 giây (theo chuẩn PERCLOS P80 trong nghiên cứu drowsiness)
    """
    
    def __init__(self, window_sec: float = 30.0, eye_closed_threshold: float = 0.16):
        """
        Args:
            window_sec: Độ dài rolling window (giây), mặc định 30s
            eye_closed_threshold: Ngưỡng EAR để coi là mắt nhắm.

        H5: mặc định là 0.16 cho khớp `fusion.EYE_CLOSED_THRESH` và
        `thresholds._DEFAULTS["eye_closed_thresh"]`. Trước đây default ở đây là
        0.18 trong khi FusionState luôn khởi tạo tracker với 0.16 — hai con số
        khác nhau cho cùng một khái niệm "mắt nhắm" là nguồn nhầm lẫn.
        Ngưỡng hiệu lực khi chạy thật KHÔNG đổi (vẫn 0.16).
        """
        self.window_ms = window_sec * 1000.0
        self.eye_closed_threshold = eye_closed_threshold
        
        # Lưu các sample (timestamp_ms, is_closed)
        # Chỉ giữ samples trong window gần nhất
        self.samples: deque[Tuple[float, bool]] = deque()
        
        # Cache kết quả để tránh tính lại khi không cần
        self._last_perclos: float | None = None
        self._last_update_ts: float | None = None
    
    def update(self, timestamp_ms: float, ear_smooth: float) -> float:
        """
        Cập nhật sample mới và tính PERCLOS.
        
        Args:
            timestamp_ms: Timestamp hiện tại (milliseconds)
            ear_smooth: EAR đã qua low-pass filter
        
        Returns:
            perclos_ratio: Tỷ lệ PERCLOS (0.0-1.0)
        """
        # Xác định mắt nhắm hay mở
        is_closed = (ear_smooth < self.eye_closed_threshold)
        
        # Thêm sample mới
        self.samples.append((timestamp_ms, is_closed))
        
        # Xóa các sample cũ ngoài window
        cutoff_ts = timestamp_ms - self.window_ms
        while self.samples and self.samples[0][0] < cutoff_ts:
            self.samples.popleft()
        
        # Tính PERCLOS
        perclos = self._calculate_perclos(timestamp_ms)
        
        # Cache
        self._last_perclos = perclos
        self._last_update_ts = timestamp_ms
        
        return perclos
    
    def _calculate_perclos(self, current_ts: float) -> float:
        """
        Tính PERCLOS từ các samples trong window.
        
        Phương pháp: tính tổng thời gian giữa các sample liên tiếp khi mắt nhắm,
        chia cho tổng thời gian window thực tế.
        """
        if len(self.samples) < 2:
            # Chưa đủ data
            return 0.0
        
        total_closed_ms = 0.0
        
        # Duyệt qua các cặp sample liên tiếp
        for i in range(len(self.samples) - 1):
            ts1, closed1 = self.samples[i]
            ts2, closed2 = self.samples[i + 1]
            
            dt = ts2 - ts1
            
            # Nếu mắt nhắm ở sample đầu, cộng thời gian này vào total_closed
            # (giả định trạng thái giữ nguyên cho đến sample tiếp theo)
            if closed1:
                total_closed_ms += dt
        
        # Thời gian thực tế của window (có thể < 30s nếu mới bắt đầu)
        window_start_ts = self.samples[0][0]
        actual_window_ms = current_ts - window_start_ts
        
        if actual_window_ms <= 0:
            return 0.0
        
        # PERCLOS ratio
        perclos = total_closed_ms / actual_window_ms
        
        # Clamp về [0, 1]
        return max(0.0, min(1.0, perclos))
    
    def get_perclos(self) -> float:
        """
        Lấy giá trị PERCLOS gần nhất (không tính lại).
        
        Returns:
            perclos_ratio: 0.0 nếu chưa có data, otherwise giá trị cached
        """
        return self._last_perclos if self._last_perclos is not None else 0.0
    
    def reset(self):
        """Reset toàn bộ state."""
        self.samples.clear()
        self._last_perclos = None
        self._last_update_ts = None
    
    def __repr__(self) -> str:
        n_samples = len(self.samples)
        perclos = self.get_perclos()
        return f"PERCLOSTracker(samples={n_samples}, perclos={perclos:.3f})"
