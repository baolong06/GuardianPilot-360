import cv2
import numpy as np
from collections import deque
from pipeline.core.base import Stage

class DepthEstimationStage(Stage):
    def __init__(self, method="bbox", smooth_window=5, focal_length_factor=0.8):
        self.method = method
        self.smooth_window = smooth_window
        self.distance_history = {}
        self.focal_length_factor = focal_length_factor
        
        self.real_heights = {
            "person": 1.7,
            "car": 1.5,
            "truck": 3.5,
            "bus": 3.2,
            "motorcycle": 1.2,
            "bicycle": 1.2,
        }

    def reset(self):
        self.distance_history.clear()

    def estimate_distance(self, bbox, class_name, frame_shape):
        x1, y1, x2, y2 = bbox
        height_pixels = y2 - y1
        frame_height, frame_width = frame_shape[:2]
        
        if height_pixels < 1:
            return 100.0
        
        real_height = self.real_heights.get(class_name, 1.5)
        focal_length = frame_width * self.focal_length_factor
        distance = (real_height * focal_length) / height_pixels
        distance = np.clip(distance, 1.0, 100.0)
        
        # Xử lý trường hợp đối tượng bị cắt xén (bị lọt ra ngoài mép dưới màn hình)
        # Nếu mép dưới của bbox gần chạm đáy (cách < 10px), có nghĩa xe đang ở rất gần
        # chiều cao đo được nhỏ hơn chiều cao thực -> distance đo được bị cao hơn thực tế
        if y2 >= frame_height - 10:
            distance = min(distance, 5.0) # Ép khoảng cách xuống mức gần để tránh đánh giá sai nguy hiểm
            
        return distance

    def process(self, data):
        frame = data["frame"]
        detections = data.get("detections", [])
        class_names = data.get("class_names", [])
        track_ids = data.get("track_ids", [])

        distances = []
        for i, bbox in enumerate(detections):
            class_name = class_names[i] if i < len(class_names) else "unknown"
            dist = self.estimate_distance(bbox, class_name, frame.shape)
            
            if i < len(track_ids):
                tid = int(track_ids[i])
                if tid not in self.distance_history:
                    self.distance_history[tid] = deque(maxlen=self.smooth_window)
                self.distance_history[tid].append(dist)
                dist = np.mean(self.distance_history[tid])
            
            distances.append(dist)

        data["distances"] = distances
        return data