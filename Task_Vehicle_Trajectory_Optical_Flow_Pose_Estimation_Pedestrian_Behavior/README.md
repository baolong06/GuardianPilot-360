# Advanced Vehicle & Pedestrian ADAS Perception System

An end-to-end Autonomous Driving & Advanced Driver Assistance System (ADAS) perception pipeline integrating Object Detection & Tracking, Monocular Distance Estimation, Ego-Motion Optical Flow, Multi-Modal Vehicle & Pedestrian Trajectory Prediction, Turn Signal Detection, Behavior Classification, Traffic Light State Recognition, and Collision Hazard Warning Fusion.

---

## 🌟 Key System Architecture & Computer Vision Pipeline

```
Input Frame / Video Stream
 ├── 1. DetectTrackStage (YOLOv8n + ByteTrack)
 ├── 2. DepthEstimationStage (Geometric Pinhole Camera Model)
 ├── 3. EgoMotionStage (Shi-Tomasi Corners + Lucas-Kanade Optical Flow)
 ├── 4. PIEVehicleTrajectoryStage (TrajectoryLSTM - Vehicle 10->12 Frames)
 ├── 5. PIEPedestrianTrajectoryStage (TrajectoryLSTM - Pedestrian 10->12 Frames)
 ├── 6. TurnSignalStage (HSV Color Filter + Temporal Flicker Analysis)
 ├── 7. BehaviorStage (BehaviorGRU + Heuristics for Vehicles)
 ├── 8. PedestrianBehaviorStage (PedestrianBehaviorGRU for Pedestrians)
 ├── 9. TrafficLightStage (ResNet18 / MobileNetV3 + HSV Fallback)
 └── 10. RiskFusionStage (Distance & Speed Collision Hazard Fusion)
```

---

## 📁 Repository Structure

* `app.py`: Web UI application using Gradio (`http://127.0.0.1:7860`).
* `real_time.py`: OpenCV webcam / stream real-time pipeline interface.
* `pipeline/`: Multi-stage pipeline coordinator (`PedestrianCVPipeline`).
* `models/`: PyTorch Deep Learning model architectures and trained weights.
* `scripts/`: Data parsers for PIE (XML) and BDD100K (JSON) with stream generators.
* `training/`: Hyperparameter configurations (`config.yaml`), DataLoaders, and training scripts.
* `evaluation/`: Evaluator scripts (ADE, FDE, Accuracy, F1, Confusion Matrix).

---

## 🚀 Getting Started

### 1. Requirements & Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Web UI Demo
```bash
python app.py
```

### 3. Run Real-time Webcam Stream
```bash
python real_time.py
```

### 4. Rebuild Unified Dataset & Validate
```bash
python scripts/build_unified_dataset.py
python scripts/validate_dataset.py
```

---

## 📊 Dataset Support & Multi-Modal Engineering

* **BDD100K Dataset:** Ground-truth ETL for single-image multi-class object detection (Car, Truck, Bus, Traffic Light, Traffic Sign). Memory-efficient streaming iterator (`ijson`).
* **PIE Dataset:** Multi-modal temporal video stream annotations (_annt.xml, _attributes.xml, _obd.xml). Features Foot-Point trajectory mapping (`[foot_x, foot_y]`), independent 4-attribute pedestrian behavior tracking (`action`, `look`, `gesture`, `cross`), and Ego-vehicle OBD sensor fusion (`obd_speed`, `gps_speed`, `heading_angle`, acceleration, gyroscope).
