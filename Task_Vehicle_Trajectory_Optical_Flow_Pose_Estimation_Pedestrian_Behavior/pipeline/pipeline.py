
import cv2
import os
import tempfile
import numpy as np
from pipeline.core.base import Pipeline
from pipeline.stages.detect_track_stage import DetectTrackStage
from pipeline.stages.depth_stage import DepthEstimationStage
from pipeline.stages.trajectory_stage import TrajectoryStage
from pipeline.stages.behavior_stage import BehaviorStage
from pipeline.stages.pedestrian_behavior_stage import PedestrianBehaviorStage
from pipeline.stages.risk_fusion_stage import RiskFusionStage
from pipeline.stages.traffic_light_stage import TrafficLightStage
from pipeline.stages.ego_motion_stage import EgoMotionStage
from pipeline.stages.turn_signal_stage import TurnSignalStage
# Import các stage mới từ PIE
from pipeline.stages.pie_vehicle_trajectory_stage import PIEVehicleTrajectoryStage
from pipeline.stages.pie_pedestrian_trajectory_stage import PIEPedestrianTrajectoryStage

class PedestrianCVPipeline:
    def __init__(self):
        # ===== VIDEO PIPELINE =====
        self.video_pipeline = Pipeline([
            DetectTrackStage(),
            DepthEstimationStage(method="bbox"),
            EgoMotionStage(),
            PIEVehicleTrajectoryStage(),
            PIEPedestrianTrajectoryStage(),
            TurnSignalStage(),
            BehaviorStage(),
            PedestrianBehaviorStage(),
            TrafficLightStage(),
            RiskFusionStage()
        ])
        
        # ===== IMAGE PIPELINE =====
        self.image_pipeline = Pipeline([
            DetectTrackStage(),
            DepthEstimationStage(method="bbox"),
            EgoMotionStage(),
            TurnSignalStage(),
            BehaviorStage(),
            PedestrianBehaviorStage(),
            TrafficLightStage(),
            RiskFusionStage()
        ])

    def run_on_image(self, frame):
        data = {"frame": frame}
        output = self.image_pipeline.run(data)
        return self._draw_output(frame, output)

    def run_on_video(self, video_path, output_path=None):
        if not isinstance(video_path, int):
            if not str(video_path).startswith(('http://', 'https://', 'rtmp://', 'rtsp://')):
                if not os.path.exists(video_path):
                    print(f"❌ File không tồn tại: {video_path}")
                    return None

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Không thể mở video: {video_path}")
            return None

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if output_path is None:
            output_dir = os.path.join(os.getcwd(), "outputs")
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, "output_video.mp4")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            data = {"frame": frame}
            output = self.video_pipeline.run(data)
            frame = self._draw_output(frame, output)
            out.write(frame)
            # gc.collect() đã bị xóa khỏi vòng lặp để tăng hiệu năng
            frame_count += 1
            if frame_count % 100 == 0:
                print(f"⏳ Đã xử lý {frame_count} frames")

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"✅ Hoàn thành! Đã xử lý {frame_count} frames.")
        print(f"📁 Video output: {output_path}")
        return output_path

    def _draw_output(self, frame, output):
        # Code giữ nguyên như bạn đã có (không thay đổi)
        detections = output.get("detections", [])
        track_ids = output.get("track_ids", [])
        class_names = output.get("class_names", [])
        confidences = output.get("confidences", [])
        distances = output.get("distances", [])
        behaviors = output.get("behaviors", {})
        pedestrian_behaviors = output.get("pedestrian_behaviors", {})
        traffic_lights = output.get("traffic_lights", {})
        turn_signals = output.get("turn_signals", {})
        predicted_trajectories = output.get("predicted_trajectories", {})
        pedestrian_trajectories = output.get("pedestrian_trajectories", {})

        for i, bbox in enumerate(detections):
            bbox = bbox.astype(int)
            class_name = class_names[i] if i < len(class_names) else ""
            has_track = i < len(track_ids)
            tid_int = int(track_ids[i]) if has_track else -1

            # Màu theo class
            if class_name == 'traffic light':
                color = (0, 255, 255)      # Vàng cho đèn
            elif class_name == 'person':
                color = (255, 200, 0)      # Cam cho người
            else:
                color = (0, 255, 0)        # Xanh cho xe

            # Vẽ bbox chính
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

            # Label chính
            label = f"ID:{tid_int}" if has_track else "ID:N/A"
            if class_name:
                label += f" {class_name}"
            if i < len(confidences):
                label += f" {confidences[i]:.2f}"
            if i < len(distances):
                label += f" {distances[i]:.1f}m"
            cv2.putText(frame, label, (bbox[0], bbox[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # === Vehicle Behavior ===
            if has_track and tid_int in behaviors and class_name not in ['traffic light', 'person']:
                behav = behaviors[tid_int]
                label_behav = f"{behav['label']}: {behav['confidence']*100:.1f}%"
                cv2.putText(frame, label_behav, (bbox[0], bbox[1]-25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)

            # === Pedestrian Behavior ===
            if has_track and tid_int in pedestrian_behaviors and class_name == 'person':
                ped_behav = pedestrian_behaviors[tid_int]
                label_ped = f"🚶 {ped_behav['label']}: {ped_behav['confidence']*100:.1f}%"
                cv2.putText(frame, label_ped, (bbox[0], bbox[1]-25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)

            # === Vehicle Trajectory Prediction ===
            if has_track and tid_int in predicted_trajectories and class_name not in ['traffic light', 'person']:
                traj = predicted_trajectories[tid_int]
                if len(traj) > 1:
                    pts = np.array(traj, dtype=np.int32)
                    # Vẽ trajectory màu xanh dương
                    for j in range(len(pts) - 1):
                        cv2.line(frame, (int(pts[j][0]), int(pts[j][1])),
                                 (int(pts[j+1][0]), int(pts[j+1][1])), (255, 0, 0), 2)
                    # Đánh dấu điểm cuối
                    cv2.circle(frame, (int(pts[-1][0]), int(pts[-1][1])), 5, (255, 0, 0), -1)

            # === Pedestrian Trajectory Prediction ===
            if has_track and tid_int in pedestrian_trajectories and class_name == 'person':
                traj = pedestrian_trajectories[tid_int]
                if len(traj) > 1:
                    pts = np.array(traj, dtype=np.int32)
                    # Vẽ trajectory màu cam
                    for j in range(len(pts) - 1):
                        cv2.line(frame, (int(pts[j][0]), int(pts[j][1])),
                                 (int(pts[j+1][0]), int(pts[j+1][1])), (0, 200, 255), 2)
                    cv2.circle(frame, (int(pts[-1][0]), int(pts[-1][1])), 5, (0, 200, 255), -1)

            # === Turn Signal ===
            if has_track and tid_int in turn_signals:
                ts = turn_signals[tid_int]
                signal = ts.get('signal', 'none')
                left_conf = ts.get('left_conf', 0.0)
                right_conf = ts.get('right_conf', 0.0)
                left_box = ts.get('left_box', bbox)
                right_box = ts.get('right_box', bbox)

                if left_conf > 0.15:
                    cv2.rectangle(frame, (left_box[0], left_box[1]), (left_box[2], left_box[3]), (0, 165, 255), 2)
                    cv2.putText(frame, f"LEFT {left_conf*100:.0f}%", (left_box[0], left_box[1]-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)
                if right_conf > 0.15:
                    cv2.rectangle(frame, (right_box[0], right_box[1]), (right_box[2], right_box[3]), (0, 255, 255), 2)
                    cv2.putText(frame, f"RIGHT {right_conf*100:.0f}%", (right_box[0], right_box[1]-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

                if signal == 'left':
                    cv2.putText(frame, "⬅️ LEFT", (bbox[0], bbox[1]-55),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
                elif signal == 'right':
                    cv2.putText(frame, "➡️ RIGHT", (bbox[0], bbox[1]-55),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                elif signal == 'both':
                    cv2.putText(frame, "BOTH", (bbox[0], bbox[1]-55),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # === Traffic Light ===
            if class_name == 'traffic light':
                key = tid_int if has_track else i
                if key in traffic_lights:
                    tl = traffic_lights[key]
                    state = tl['state']
                    conf = tl['confidence']
                    label_tl = f"SIGNAL: {state.upper()} ({conf*100:.1f}%)"
                    if state == 'red':
                        tl_color = (0, 0, 255)
                    elif state == 'yellow':
                        tl_color = (0, 255, 255)
                    elif state == 'green':
                        tl_color = (0, 255, 0)
                    else:
                        tl_color = (255, 255, 255)
                    cv2.putText(frame, label_tl, (bbox[0], bbox[1]-25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, tl_color, 1)

        # Warning
        warning = output.get("warning_level", "NONE")
        nearest = output.get("nearest_object", {})
        if warning != "NONE":
            color = (0, 0, 255) if warning == "BRAKE" else (0, 165, 255) if warning == "ALERT" else (0, 255, 255)
            cv2.putText(frame, f"WARNING: {warning}", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            if nearest:
                cv2.putText(frame, f"Nearest: {nearest.get('class','')} {nearest.get('distance',0):.1f}m",
                            (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        else:
            cv2.putText(frame, "SAFE", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        return frame