import numpy as np
from pipeline.core.base import Stage

class RiskFusionStage(Stage):
    def __init__(self, thresholds=None):
        # FIX: Không dùng mutable default argument
        if thresholds is None:
            thresholds = {"watch": 15, "alert": 8, "brake": 4}
        self.thresholds = thresholds
        self.last_distances = {}
        self.last_warning = None

    def process(self, data):
        distances = data.get("distances", [])
        class_names = data.get("class_names", [])
        confidences = data.get("confidences", [])
        track_ids = data.get("track_ids", [])
        
        # Dọn dẹp cache (chỉ giữ lại những track_ids hiện tại)
        current_tids = set([int(t) for t in track_ids]) if track_ids is not None else set()
        self.last_distances = {k: v for k, v in self.last_distances.items() if k in current_tids}

        if not distances:
            data["warning_level"] = "NONE"
            return data

        min_dist = min(distances)
        idx_min = distances.index(min_dist)
        nearest_class = class_names[idx_min] if idx_min < len(class_names) else "unknown"
        nearest_conf = confidences[idx_min] if idx_min < len(confidences) else 0
        
        tid_min = int(track_ids[idx_min]) if idx_min < len(track_ids) else -1
        
        # Tính khoảng cách thay đổi (relative delta_d)
        is_approaching = True
        if tid_min != -1:
            if tid_min in self.last_distances:
                delta_d = min_dist - self.last_distances[tid_min]
                if delta_d >= -0.05: 
                    is_approaching = False
            self.last_distances[tid_min] = min_dist

        if min_dist < self.thresholds["brake"]:
            warning_level = "BRAKE" if is_approaching else "WATCH"
        elif min_dist < self.thresholds["alert"]:
            warning_level = "ALERT" if is_approaching else "WATCH"
        elif min_dist < self.thresholds["watch"]:
            warning_level = "WATCH"
        else:
            warning_level = "NONE"

        data["warning_level"] = warning_level
        data["nearest_object"] = {
            "class": nearest_class,
            "confidence": nearest_conf,
            "distance": min_dist
        }
        return data