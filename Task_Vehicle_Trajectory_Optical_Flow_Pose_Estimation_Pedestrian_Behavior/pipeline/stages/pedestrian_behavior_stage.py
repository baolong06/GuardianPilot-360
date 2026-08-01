import torch
import numpy as np
from collections import deque
from pipeline.core.base import Stage
from models.pedestrian_behavior.model import PedestrianBehaviorGRU
import logging

logger = logging.getLogger(__name__)

class PedestrianBehaviorStage(Stage):
    """Phân loại hành vi người đi bộ (walking/standing) từ PIE-trained model."""
    
    def __init__(self, model_path="models/pedestrian_behavior/weights/best.pth", window_len=10):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model
        self.model = PedestrianBehaviorGRU(input_dim=4, hidden_dim=128, num_layers=3, num_classes=2)
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
            self.model.to(self.device)
            self.model.eval()
            print("✅ PedestrianBehavior model loaded.")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

        
        self.window_len = window_len
        self.history = {}  # track_id -> deque of (cx, cy)
        self.label_map = {0: 'standing', 1: 'walking'}
        
    def reset(self):
        self.history.clear()
        
    def process(self, data):
        frame = data.get("frame")
        if frame is None:
            return data
        
        detections = data.get("detections", [])
        track_ids = data.get("track_ids", [])
        class_names = data.get("class_names", [])
        h_frame, w_frame = frame.shape[:2]
        
        pedestrian_behaviors = {}
        
        for i, tid in enumerate(track_ids):
            tid = int(tid)
            class_name = class_names[i] if i < len(class_names) else ""
            
            # Chỉ xử lý người đi bộ
            if class_name != 'person':
                continue
            
            if i >= len(detections):
                continue
            
            bbox = detections[i]
            cx = (bbox[0] + bbox[2]) / 2.0
            cy = (bbox[1] + bbox[3]) / 2.0
            
            if tid not in self.history:
                self.history[tid] = deque(maxlen=self.window_len)
            self.history[tid].append((cx, cy))
            
            if len(self.history[tid]) < self.window_len:
                continue
            
            window = np.array(self.history[tid])
            
            # Tính speed và heading
            dx = np.diff(window[:, 0]) / w_frame
            dy = np.diff(window[:, 1]) / h_frame
            speed = np.sqrt(dx**2 + dy**2)
            heading = np.arctan2(dy, dx)
            
            features = []
            for j in range(self.window_len):
                if j == 0:
                    sp, hd = 0.0, 0.0
                else:
                    sp, hd = speed[j-1], heading[j-1]
                features.append([
                    window[j, 0] / w_frame,
                    window[j, 1] / h_frame,
                    sp, hd
                ])
            
            features = np.array(features, dtype=np.float32)
            
            with torch.no_grad():
                inp = torch.tensor(features).unsqueeze(0).to(self.device)
                output = self.model(inp)
                probs = torch.softmax(output, dim=1).cpu().numpy()[0]
            
            pred_label = np.argmax(probs)
            confidence = probs[pred_label]
            
            pedestrian_behaviors[tid] = {
                'label': self.label_map[pred_label],
                'confidence': confidence
            }
        
        data["pedestrian_behaviors"] = pedestrian_behaviors
        return data