import torch
import numpy as np
from collections import deque
from pipeline.core.base import Stage
from models.behavior_clf.model import BehaviorGRU
import logging

logger = logging.getLogger(__name__)


class PIEVehicleBehaviorStage(Stage):
    """Phân loại hành vi xe từ PIE-trained model.

    Output key: 'behaviors' (khớp với _draw_output trong pipeline.py).
    Note: Stage này không được gọi trong video_pipeline mặc định —
    BehaviorStage đã đảm nhiệm việc phân loại xe + người trong pipeline.
    PIEVehicleBehaviorStage có thể dùng như alternative hoặc cho standalone eval.
    """

    def __init__(self, model_path="models/pie_vehicle_behavior/weights/best.pth", window_len=10):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = BehaviorGRU(input_dim=4, hidden_dim=128, num_layers=3, num_classes=4)
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"Loaded PIEVehicleBehavior model from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load PIEVehicleBehavior model: {e}")
            raise

        self.window_len = window_len
        self.history = {}
        self.last_seen = {}
        self.frame_count = 0
        self.label_map = {0: 'stop', 1: 'straight', 2: 'turn_left', 3: 'turn_right'}

    def reset(self):
        self.history.clear()
        self.last_seen.clear()
        self.frame_count = 0

    def _cleanup_stale_tracks(self):
        timeout = 30
        stale = [t for t, last in self.last_seen.items() if self.frame_count - last > timeout]
        for t in stale:
            self.history.pop(t, None)
            self.last_seen.pop(t, None)

    def process(self, data):
        frame = data.get("frame")
        if frame is None:
            return data

        self.frame_count += 1
        self._cleanup_stale_tracks()

        detections = data.get("detections", [])
        track_ids = data.get("track_ids", [])
        class_names = data.get("class_names", [])
        h_frame, w_frame = frame.shape[:2]

        # NOTE: output key là "behaviors" để khớp với _draw_output trong pipeline.py
        behaviors = {}

        for i, tid in enumerate(track_ids):
            tid = int(tid)
            if i >= len(detections):
                continue
            if class_names[i] in ('person', 'traffic light'):
                continue

            try:
                bbox = detections[i]
                cx = (bbox[0] + bbox[2]) / 2
                cy = (bbox[1] + bbox[3]) / 2

                if tid not in self.history:
                    self.history[tid] = deque(maxlen=self.window_len)

                self.history[tid].append((cx, cy))
                self.last_seen[tid] = self.frame_count

                if len(self.history[tid]) < self.window_len:
                    continue

                window = np.array(self.history[tid])

                dx = np.diff(window[:, 0]) / w_frame
                dy = np.diff(window[:, 1]) / h_frame
                speed = np.sqrt(dx**2 + dy**2)
                heading = np.arctan2(dy, dx)

                features = []
                for j in range(self.window_len):
                    sp = speed[j-1] if j > 0 else 0.0
                    hd = heading[j-1] if j > 0 else 0.0
                    features.append([window[j, 0] / w_frame, window[j, 1] / h_frame, sp, hd])

                features = np.array(features, dtype=np.float32)

                with torch.no_grad():
                    inp = torch.tensor(features).unsqueeze(0).to(self.device)
                    output = self.model(inp)
                    probs = torch.softmax(output, dim=1).cpu().numpy()[0]

                pred_label = int(np.argmax(probs))
                confidence = float(probs[pred_label])

                behaviors[tid] = {
                    'label': self.label_map[pred_label],
                    'confidence': confidence
                }
            except Exception as e:
                logger.error(f"Error PIEVehicleBehavior TID {tid}: {e}", exc_info=True)
                continue

        data["behaviors"] = behaviors
        return data