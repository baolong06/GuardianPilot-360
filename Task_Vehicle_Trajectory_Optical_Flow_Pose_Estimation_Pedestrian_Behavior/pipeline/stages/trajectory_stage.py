import torch
import numpy as np
from collections import deque
from pipeline.core.base import Stage
from models.trajectory.model import TrajectoryLSTM

class TrajectoryStage(Stage):
    def __init__(self, model_path="models/trajectory/weights/best.pth", obs_len=5, pred_len=5, use_inference=False):
        self.use_inference = use_inference
        if self.use_inference:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = TrajectoryLSTM(input_dim=4, hidden_dim=128, num_layers=3, 
                                        pred_len=pred_len, obs_len=obs_len)
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.history = {}

    def reset(self):
        self.history.clear()

    def process(self, data):
        detections = data.get("detections", [])
        track_ids = data.get("track_ids", [])
        # Không dùng trajectories cho behavior
        for i, tid in enumerate(track_ids):
            tid = int(tid)
            if i >= len(detections):
                continue
            bbox = detections[i]
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            
            if tid not in self.history:
                self.history[tid] = deque(maxlen=self.obs_len)
            self.history[tid].append([cx, cy, 0.0, 0.0])
            # Không cần output gì thêm
        return data