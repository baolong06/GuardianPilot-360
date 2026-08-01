import cv2
import numpy as np
from ultralytics import YOLO
from pipeline.core.base import Stage
import os

class DetectTrackStage(Stage):
    def __init__(self, model_path="yolov8n.pt", vehicle_conf=0.25, tl_conf=0.15, iou=0.5):
        self.model = YOLO(model_path)
        self.vehicle_conf = vehicle_conf
        self.tl_conf = tl_conf
        self.iou = iou
        self.model.fuse()
        
        self.valid_classes = ['car', 'truck', 'bus', 'motorcycle', 'bicycle', 'traffic light', 'person']
        # Find indices of valid classes in the model
        self.valid_class_ids = [k for k, v in self.model.names.items() if v in self.valid_classes]
        self.tracker_config = "bytetrack.yaml"

    def reset(self):
        # Reset YOLO predictor to clear tracking history (ByteTrack state)
        if hasattr(self.model, 'predictor') and self.model.predictor is not None:
            self.model.predictor = None

    def process(self, data):
        frame = data["frame"]
        
        # Use min(vehicle_conf, tl_conf) so YOLO doesn't drop traffic lights before our custom filter.
        # Pass classes to speed up tracking.
        results = self.model.track(
            frame, persist=True, conf=min(self.vehicle_conf, self.tl_conf), iou=self.iou, 
            tracker=self.tracker_config, classes=self.valid_class_ids, verbose=False
        )
        
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            cls = results[0].boxes.cls.int().cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().numpy()
            class_names = [self.model.names[c] for c in cls]
            
            filtered_boxes, filtered_conf, filtered_track, filtered_names = [], [], [], []
            for i, name in enumerate(class_names):
                # Tách threshold riêng cho traffic light
                if name == 'traffic light':
                    min_conf = self.tl_conf
                else:
                    min_conf = self.vehicle_conf
                if name in self.valid_classes and confs[i] >= min_conf:
                    filtered_boxes.append(boxes[i])
                    filtered_conf.append(confs[i])
                    filtered_track.append(track_ids[i])
                    filtered_names.append(name)
            
            boxes = np.array(filtered_boxes) if filtered_boxes else np.empty((0, 4))
            confs = np.array(filtered_conf) if filtered_conf else np.empty(0)
            track_ids = np.array(filtered_track) if filtered_track else np.empty(0)
            class_names = filtered_names
        else:
            boxes = np.empty((0, 4))
            confs = np.empty(0)
            cls = np.empty(0)
            track_ids = np.empty(0)
            class_names = []

        data["detections"] = boxes
        data["track_ids"] = track_ids
        data["class_names"] = class_names
        data["confidences"] = confs
        return data