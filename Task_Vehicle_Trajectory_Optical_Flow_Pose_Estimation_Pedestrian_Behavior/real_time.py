import sys
import os
import cv2
import time
from pipeline.pipeline import PedestrianCVPipeline
from pipeline.stages.detect_track_stage import DetectTrackStage
from pipeline.stages.pie_vehicle_trajectory_stage import PIEVehicleTrajectoryStage  # Thay thế TrajectoryStage
from pipeline.stages.risk_fusion_stage import RiskFusionStage
from pipeline.core.base import Pipeline

class RealTimePipeline(Pipeline):
    def __init__(self, use_trajectory=False):
        stages = [DetectTrackStage()]
        if use_trajectory:
            stages.append(PIEVehicleTrajectoryStage())  # Đã thay bằng stage mới
        stages.append(RiskFusionStage())
        super().__init__(stages)
    
    def process_frame(self, frame):
        data = {"frame": frame}
        output = self.run(data)
        return self._draw_output(frame, output)
    
    def _draw_output(self, frame, output):
        detections = output.get("detections", [])
        track_ids = output.get("track_ids", [])
        for i, tid in enumerate(track_ids):
            bbox = detections[i].astype(int)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            cv2.putText(frame, f"ID:{tid}", (bbox[0], bbox[1]-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        warning = output.get("warning_level", "NONE")
        cv2.putText(frame, f"Warning: {warning}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        return frame

def run_realtime(use_trajectory=False, resize_factor=0.5, camera_id=0):
    print(f"🔄 Khởi tạo pipeline real-time (trajectory={'ON' if use_trajectory else 'OFF'})...")
    pipeline = RealTimePipeline(use_trajectory=use_trajectory)
    
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print("❌ Không thể mở webcam.")
        return
    
    print("📹 Webcam đã mở. Nhấn 'q' để thoát.")
    
    fps_display = 0
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if resize_factor != 1.0:
            h, w = frame.shape[:2]
            new_w = int(w * resize_factor)
            new_h = int(h * resize_factor)
            frame = cv2.resize(frame, (new_w, new_h))
        
        start_process = time.time()
        frame_out = pipeline.process_frame(frame)
        process_time = time.time() - start_process
        
        frame_count += 1
        if frame_count % 30 == 0:
            elapsed = time.time() - start_time
            fps_display = 30 / elapsed
            start_time = time.time()
        
        cv2.putText(frame_out, f"FPS: {fps_display:.1f}", (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        cv2.putText(frame_out, f"Process time: {process_time*1000:.0f}ms", (50, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
        
        cv2.imshow("Real-time Pedestrian Warning", frame_out)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_realtime(use_trajectory=False, resize_factor=0.5)