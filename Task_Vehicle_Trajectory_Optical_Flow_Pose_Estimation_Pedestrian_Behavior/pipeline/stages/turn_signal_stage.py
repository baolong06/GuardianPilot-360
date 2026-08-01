import cv2
import numpy as np
from collections import deque
from pipeline.core.base import Stage
import logging

logger = logging.getLogger(__name__)


class TurnSignalStage(Stage):
    """
    Phát hiện đèn xi nhan — v1.1 FINAL

    Fix ảnh 5: thêm `camera_view='rear'` và logic đảo left/right.
    """

    def __init__(
        self,
        history_len=12,
        flicker_min_cycles=2,
        roi_w_frac=0.22,
        roi_h_frac=0.28,
        roi_y_frac=0.50,        # ↓ từ 0.55 — đèn hậu nằm giữa-dưới
        min_orange_px=18,
        camera_view='front',    # 'front' (front dashcam) hoặc 'rear'
    ):
        self.history_len        = history_len
        self.flicker_min_cycles = flicker_min_cycles
        self.roi_w_frac         = roi_w_frac
        self.roi_h_frac         = roi_h_frac
        self.roi_y_frac         = roi_y_frac
        self.min_orange_px      = min_orange_px
        self.conf_threshold     = conf_threshold
        self.camera_view        = camera_view   # 'rear' → đảo trái/phải

        self.history   = {}
        self.last_seen = {}
        self.frame_count = 0

    def reset(self):
        self.history.clear()
        self.last_seen.clear()
        self.frame_count = 0

    def _get_headlight_rois(self, frame, bbox):
        fh, fw = frame.shape[:2]
        x1, y1, x2, y2 = map(int, bbox)
        w = x2 - x1
        h = y2 - y1
        if w < 40 or h < 30:
            return None, None, None, None

        roi_w      = max(12, int(w * self.roi_w_frac))
        roi_h      = max(10, int(h * self.roi_h_frac))
        roi_y_start = y1 + int(h * self.roi_y_frac)

        lx1 = max(0,  x1)
        lx2 = min(fw, x1 + roi_w)
        ly1 = max(0,  roi_y_start)
        ly2 = min(fh, roi_y_start + roi_h)

        rx1 = max(0,  x2 - roi_w)
        rx2 = min(fw, x2)
        ry1 = max(0,  roi_y_start)
        ry2 = min(fh, roi_y_start + roi_h)

        left_box  = (lx1, ly1, lx2, ly2)
        right_box = (rx1, ry1, rx2, ry2)
        left_roi  = frame[ly1:ly2, lx1:lx2]
        right_roi = frame[ry1:ry2, rx1:rx2]

        return left_roi, right_roi, left_box, right_box

    def _count_signal_pixels(self, roi):
        if roi is None or roi.size == 0:
            return 0, 0.0
        try:
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        except Exception:
            return 0, 0.0
        lower = np.array([8,  100, 120], dtype=np.uint8)
        upper = np.array([35, 255, 255], dtype=np.uint8)
        mask  = cv2.inRange(hsv, lower, upper)
        n_px  = int(np.sum(mask > 0))
        mean_v = float(np.mean(hsv[:, :, 2])) if n_px > 0 else 0.0
        return n_px, mean_v

    def _is_signal_on(self, roi):
        n_px, _ = self._count_signal_pixels(roi)
        return n_px >= self.min_orange_px

    def _analyze_flicker(self, signal_history):
        if len(signal_history) < 4:
            return False, 0.0
        arr         = np.array(signal_history, dtype=int)
        transitions = int(np.sum(np.abs(np.diff(arr))))
        on_ratio    = float(np.mean(arr))
        min_trans   = self.flicker_min_cycles * 2
        is_flicker  = (transitions >= min_trans and 0.20 <= on_ratio <= 0.80)
        if is_flicker:
            flicker_score = min(1.0, transitions / (min_trans * 2))
            balance_score = 1.0 - abs(on_ratio - 0.5) * 2
            confidence    = flicker_score * 0.6 + balance_score * 0.4
        else:
            confidence = 0.0
        return is_flicker, confidence

    def _cleanup_stale_tracks(self, timeout=30):
        stale = [t for t, last in self.last_seen.items()
                 if self.frame_count - last > timeout]
        for t in stale:
            self.history.pop(t, None)
            self.last_seen.pop(t, None)

    def process(self, data):
        frame = data.get("frame")
        if frame is None:
            return data

        self.frame_count += 1
        self._cleanup_stale_tracks()

        detections  = data.get("detections",  [])
        track_ids   = data.get("track_ids",   [])
        class_names = data.get("class_names", [])
        turn_signals = {}

        for i, tid in enumerate(track_ids):
            tid        = int(tid)
            class_name = class_names[i] if i < len(class_names) else ""
            if class_name not in ('car', 'truck', 'bus', 'motorcycle'):
                continue
            if i >= len(detections):
                continue

            try:
                bbox = detections[i]
                left_roi, right_roi, left_box, right_box = \
                    self._get_headlight_rois(frame, bbox)

                if left_roi is None:
                    continue

                # left_on/right_on theo góc nhìn CAMERA (bbox coordinates)
                left_on_cam  = self._is_signal_on(left_roi)
                right_on_cam = self._is_signal_on(right_roi)

                if tid not in self.history:
                    self.history[tid] = deque(maxlen=self.history_len)
                self.history[tid].append((left_on_cam, right_on_cam))
                self.last_seen[tid] = self.frame_count

                history_list = list(self.history[tid])
                left_hist    = [h[0] for h in history_list]
                right_hist   = [h[1] for h in history_list]

                left_flicker,  left_conf  = self._analyze_flicker(left_hist)
                right_flicker, right_conf = self._analyze_flicker(right_hist)

                # Xác định hướng theo tọa độ camera
                if left_flicker and right_flicker:
                    cam_direction = 'none'
                    confidence    = 0.0
                elif left_flicker and left_conf >= self.conf_threshold:
                    cam_direction = 'left'
                    confidence    = left_conf
                elif right_flicker and right_conf >= self.conf_threshold:
                    cam_direction = 'right'
                    confidence    = right_conf
                else:
                    cam_direction = 'none'
                    confidence    = 0.0

                # FIX ảnh 5: Đảo trái/phải nếu camera nhìn từ phía sau xe
                if self.camera_view == 'rear' and cam_direction != 'none':
                    actual_direction = 'left' if cam_direction == 'right' else 'right'
                else:
                    actual_direction = cam_direction

                turn_signals[tid] = {
                    'signal':     actual_direction,   # key là 'signal' để pipeline dùng
                    'confidence': confidence,
                    'left_box':   left_box,
                    'right_box':  right_box,
                    'left_on':    left_on_cam,
                    'right_on':   right_on_cam,
                    'left_conf':  left_conf,
                    'right_conf': right_conf,
                }

                if actual_direction != 'none':
                    logger.debug(
                        f"TID {tid}: turn={actual_direction} "
                        f"(cam={cam_direction}, conf={confidence:.2f})"
                    )

            except Exception as e:
                logger.error(f"TurnSignal error tid={tid}: {e}", exc_info=True)

        data["turn_signals"] = turn_signals
        return data