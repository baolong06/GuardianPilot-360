import torch
import numpy as np
from collections import deque
from pipeline.core.base import Stage
from models.trajectory.model import TrajectoryLSTM
import json
import os
import logging

logger = logging.getLogger(__name__)

class PIEPedestrianTrajectoryStage(Stage):
    def __init__(self, model_path="models/pie_trajectory/pedestrian/weights/best.pth", obs_len=10, pred_len=12):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TrajectoryLSTM(input_dim=4, hidden_dim=128, num_layers=3, pred_len=pred_len, obs_len=obs_len)
        
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
            self.model.to(self.device)
            self.model.eval()
            print("[OK] PIEPedestrianTrajectory model loaded.")
            logger.info(f"Loaded pedestrian trajectory model from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load pedestrian trajectory model: {e}")
            raise

        
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.history = {}

        norm_path = os.path.join(os.path.dirname(model_path), "norm_params.json")
        if os.path.exists(norm_path):
            with open(norm_path, 'r') as f:
                self.norm_params = json.load(f)
            logger.info(f"Loaded norm_params from {norm_path}")
        else:
            logger.warning(f"norm_params.json not found at {norm_path}")
            self.norm_params = None

    def reset(self):
        """Gọi khi chuyển sang video mới để xóa toàn bộ track history."""
        self.history.clear()

    def _cleanup_stale_tracks(self, current_tids: set):
        """Xóa track_id không còn active trong frame hiện tại.
        Tránh memory leak khi chạy video dài với nhiều pedestrian.
        """
        stale = [tid for tid in list(self.history.keys()) if tid not in current_tids]
        for tid in stale:
            self.history.pop(tid, None)

    def process(self, data):
        detections = data.get("detections", [])
        track_ids = data.get("track_ids", [])
        class_names = data.get("class_names", [])
        trajectories = {}

        current_tids = set(int(tid) for tid in track_ids) if len(track_ids) > 0 else set()
        self._cleanup_stale_tracks(current_tids)

        for i, tid in enumerate(track_ids):
            tid = int(tid)
            if i >= len(detections):
                continue
            if class_names[i] != 'person':
                continue
            
            bbox = detections[i]
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            
            if self.norm_params is not None:
                cx_norm = (cx - self.norm_params['x_mean']) / (self.norm_params['x_std'] + 1e-8)
                cy_norm = (cy - self.norm_params['y_mean']) / (self.norm_params['y_std'] + 1e-8)
            else:
                cx_norm = cx
                cy_norm = cy
            
            if tid not in self.history:
                self.history[tid] = deque(maxlen=self.obs_len)
            
            self.history[tid].append([cx_norm, cy_norm, 0.0, 0.0])
            
            if len(self.history[tid]) >= 2:
                hist_list = list(self.history[tid])
                dx = hist_list[-1][0] - hist_list[-2][0]
                dy = hist_list[-1][1] - hist_list[-2][1]
                speed = np.sqrt(dx**2 + dy**2)
                heading = np.arctan2(dy, dx)
                self.history[tid][-1][2] = speed
                self.history[tid][-1][3] = heading
            
            if len(self.history[tid]) == self.obs_len:
                obs = np.array(self.history[tid], dtype=np.float32)
                obs_tensor = torch.tensor(obs).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    pred_norm = self.model(obs_tensor)
                pred = pred_norm.cpu().numpy()[0]
                
                if self.norm_params is not None:
                    pred[:, 0] = pred[:, 0] * self.norm_params['x_std'] + self.norm_params['x_mean']
                    pred[:, 1] = pred[:, 1] * self.norm_params['y_std'] + self.norm_params['y_mean']
                
                trajectories[tid] = pred.tolist()

        data["pedestrian_trajectories"] = trajectories
        return data