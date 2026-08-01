import torch
import numpy as np
from collections import deque
from pipeline.core.base import Stage
from models.behavior_clf.model import BehaviorGRU
import logging

logger = logging.getLogger(__name__)


class BehaviorStage(Stage):
    """
    Phân loại hành vi xe/người — v3.2 FINAL
    """

    PERSON_CLASSES = {'person', 'bicycle'}
    VEHICLE_CLASSES = {'car', 'truck', 'bus', 'motorcycle'}

    def __init__(
        self,
        model_path="models/behavior_clf/weights/best.pth",
        window_len=10,
        turn_signal_threshold=0.35,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # num_layers=2 khớp với training/configs/config.yaml (behavior_clf → num_layers: 2)
        self.model = BehaviorGRU(input_dim=4, hidden_dim=128, num_layers=2, num_classes=4)

        try:
            state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
            # Auto-detect num_layers từ checkpoint để tránh mismatch
            detected_layers = 1
            for key in state_dict.keys():
                if key.startswith("gru.weight_hh_l"):
                    idx = int(key.split("gru.weight_hh_l")[1].split("_")[0].split(".")[0])
                    detected_layers = max(detected_layers, idx + 1)
            self.model = BehaviorGRU(input_dim=4, hidden_dim=128, num_layers=detected_layers, num_classes=4)
            self.model.load_state_dict(state_dict)
            print(f"[OK] BehaviorGRU loaded (num_layers={detected_layers}) from {model_path}")
            logger.info(f"BehaviorGRU loaded OK (num_layers={detected_layers})")
        except Exception as e:
            logger.error(f"Failed to load BehaviorGRU from {model_path}: {e}")
            raise

        self.model.to(self.device)
        self.model.eval()
        self.window_len = window_len

        self.turn_signal_threshold = turn_signal_threshold

        # ── Threshold cho XE ──────────────────────────────────────────
        self.base_speed_thresh        = 0.55
        self.std_speed_thresh         = 0.20
        self.std_area_thresh          = 900
        self.std_w_thresh             = 12.0
        self.std_h_thresh             = 12.0
        self.direction_changes_thresh = 5
        self.boost_stop               = 0.08
        self.min_frames_for_stop      = 6

        # ── Threshold riêng cho NGƯỜI ─────────────────────────────────
        self.person_speed_thresh      = 0.008
        self.person_min_frames        = 8
        self.person_std_thresh        = 0.005

        self.history        = {}
        self.smooth_history = {}
        self.bbox_history   = {}
        self.last_seen      = {}
        self.frame_count    = 0

    def reset(self):
        self.history.clear()
        self.smooth_history.clear()
        self.bbox_history.clear()
        self.last_seen.clear()

    def _cleanup_stale_tracks(self):
        timeout = 30
        stale = [t for t, last in self.last_seen.items()
                 if self.frame_count - last > timeout]
        for t in stale:
            self.history.pop(t, None)
            self.smooth_history.pop(t, None)
            self.bbox_history.pop(t, None)
            self.last_seen.pop(t, None)

    # ── Heuristic stop cho XE ─────────────────────────────────────────
    def _is_stop_vehicle(self, avg_speed, std_speed, std_area, std_w, std_h,
                         direction_changes, adaptive_thresh, n_frames):
        if n_frames < self.min_frames_for_stop:
            return False
        if avg_speed > adaptive_thresh * 0.7:
            return False
        conds = (
            avg_speed         < adaptive_thresh,
            std_speed         < self.std_speed_thresh,
            std_area          < self.std_area_thresh,
            std_w             < self.std_w_thresh,
            std_h             < self.std_h_thresh,
            direction_changes < self.direction_changes_thresh,
        )
        return sum(conds) >= 5

    # ── Heuristic stop cho NGƯỜI ──────────────────────────────────────
    def _is_stop_person(self, avg_speed, std_speed, n_frames):
        if n_frames < self.person_min_frames:
            return False
        return (avg_speed < self.person_speed_thresh and
                std_speed < self.person_std_thresh)

    def process(self, data):
        frame = data.get("frame")
        if frame is None:
            return data

        self.frame_count += 1
        self._cleanup_stale_tracks()

        detections   = data.get("detections",   [])
        track_ids    = data.get("track_ids",    [])
        class_names  = data.get("class_names",  [])
        turn_signals = data.get("turn_signals", {})
        behaviors    = {}

        ego_dx = data.get("ego_dx", 0.0)
        ego_dy = data.get("ego_dy", 0.0)
        h_frame, w_frame = frame.shape[:2]
        label_map = {0: 'stop', 1: 'straight', 2: 'turn_left', 3: 'turn_right'}

        for i, tid in enumerate(track_ids):
            tid = int(tid)
            class_name = class_names[i] if i < len(class_names) else ""

            if class_name == 'traffic light':
                continue
            if i >= len(detections):
                continue

            try:
                bbox = detections[i]
                x1, y1, x2, y2 = bbox.astype(int)
                cx   = (x1 + x2) / 2.0
                cy   = (y1 + y2) / 2.0
                w    = x2 - x1
                h    = y2 - y1
                area = w * h

                if tid not in self.history:
                    self.history[tid]        = deque(maxlen=self.window_len)
                    self.bbox_history[tid]   = deque(maxlen=self.window_len)
                    self.smooth_history[tid] = deque(maxlen=7)

                self.history[tid].append((cx, cy))
                self.bbox_history[tid].append((w, h, area))
                self.last_seen[tid] = self.frame_count

                # ── FIX BUG: Kiểm tra trạng thái tĩnh TRƯỚC khi áp dụng turn_signal override ──
                # Tính sơ bộ speed để biết xe có đang đứng yên không
                _n = len(self.history[tid])
                _is_stationary = False
                if _n >= 3:
                    _window_pre = np.array(self.history[tid])
                    _dx = np.diff(_window_pre[:, 0]) / w_frame
                    _dy = np.diff(_window_pre[:, 1]) / h_frame
                    _speed_pre = np.sqrt(_dx**2 + _dy**2)
                    _avg_speed_pre = float(np.mean(_speed_pre))
                    # Nếu tốc độ trung bình rất nhỏ → đang dừng (bất kể xi nhan)
                    _is_stationary = _avg_speed_pre < 0.008

                # Turn signal override CHỈ áp dụng khi xe ĐANG CHUYỂN ĐỘNG
                if tid in turn_signals and not _is_stationary:
                    ts        = turn_signals[tid]
                    signal    = ts.get('signal', 'none')
                    conf      = ts.get('confidence', 0.0)
                    if signal in ('left', 'right') and conf >= self.turn_signal_threshold:
                        final_label = 'turn_left' if signal == 'left' else 'turn_right'
                        final_conf  = min(conf, 0.95)
                        behaviors[tid] = {'label': final_label, 'confidence': final_conf}
                        self.smooth_history[tid].clear()
                        print(f"Behavior TID {tid}: {final_label} via signal ({final_conf:.2f})")
                        continue

                n_frames = len(self.history[tid])
                if n_frames < self.window_len:
                    continue

                window      = np.array(self.history[tid])
                bbox_window = np.array(self.bbox_history[tid])

                dx_raw = np.diff(window[:, 0]) / w_frame
                dy_raw = np.diff(window[:, 1]) / h_frame
                dx     = dx_raw - ego_dx
                dy     = dy_raw - ego_dy
                speed  = np.sqrt(dx**2 + dy**2)

                heading = np.arctan2(dy, dx) / np.pi
                for j in range(1, len(heading)):
                    if speed[j-1] < 0.001:
                        heading[j] = heading[j-1] if j > 0 else 0.0

                avg_speed = float(np.mean(speed))
                std_speed = float(np.std(speed))

                is_person = class_name in self.PERSON_CLASSES

                if is_person:
                    is_stop = self._is_stop_person(avg_speed, std_speed, n_frames)
                    if is_stop:
                        behaviors[tid] = {'label': 'stop', 'confidence': 0.75}
                        print(f"Behavior TID {tid}: stop (person standing, spd={avg_speed:.5f})")
                        continue
                    behaviors[tid] = {'label': 'straight', 'confidence': 0.80}
                    print(f"Behavior TID {tid}: straight (person walking, spd={avg_speed:.5f})")
                    continue

                # ── GRU path cho XE ───────────────────────────────────
                std_area = float(np.std(bbox_window[:, 2]))
                std_w    = float(np.std(bbox_window[:, 0]))
                std_h    = float(np.std(bbox_window[:, 1]))

                scale           = np.clip(np.sqrt(area) / 200.0, 0.7, 2.0)
                adaptive_thresh = self.base_speed_thresh * scale

                direction_changes = (
                    int(np.sum(np.abs(np.diff(np.sign(dx))) > 0)) +
                    int(np.sum(np.abs(np.diff(np.sign(dy))) > 0))
                )

                is_stop = self._is_stop_vehicle(
                    avg_speed, std_speed, std_area, std_w, std_h,
                    direction_changes, adaptive_thresh, n_frames,
                )

                features = []
                for j in range(self.window_len):
                    sp = speed[j-1]   if j > 0 else 0.0
                    hd = heading[j-1] if j > 0 else 0.0
                    features.append([window[j, 0] / w_frame,
                                     window[j, 1] / h_frame,
                                     sp, hd])

                features = np.array(features, dtype=np.float32)
                with torch.no_grad():
                    inp    = torch.tensor(features).unsqueeze(0).to(self.device)
                    output = self.model(inp)
                    probs  = torch.softmax(output, dim=1).cpu().numpy()[0]

                if is_stop:
                    probs[0] += self.boost_stop
                    probs     = probs / np.sum(probs)

                pred_label = int(np.argmax(probs))
                confidence = float(probs[pred_label])

                if pred_label == 0 and avg_speed >= adaptive_thresh:
                    pred_label = 1
                    confidence = float(probs[1])

                self.smooth_history[tid].append((pred_label, confidence))
                decay      = 0.8
                vote_score = {}
                for idx, (lbl, conf) in enumerate(reversed(self.smooth_history[tid])):
                    weight = decay ** idx
                    vote_score[lbl] = vote_score.get(lbl, 0.0) + conf * weight

                final_label = max(vote_score, key=vote_score.get)
                total_score = sum(vote_score.values())
                final_conf  = min(
                    vote_score[final_label] / total_score if total_score > 0 else 0.0,
                    0.95,
                )

                behaviors[tid] = {
                    'label':      label_map[final_label],
                    'confidence': final_conf,
                }
                print(
                    f"Behavior TID {tid} [{class_name}]: {label_map[final_label]} "
                    f"({final_conf:.2f}) spd={avg_speed:.4f} thresh={adaptive_thresh:.4f} stop={is_stop}"
                )

            except Exception as e:
                logger.error(f"Error TID {tid}: {e}", exc_info=True)
                continue

        data["behaviors"] = behaviors
        return data