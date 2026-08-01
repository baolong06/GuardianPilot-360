import sys
import os
import cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline.pipeline import PedestrianCVPipeline

# Khởi tạo pipeline
pipeline = PedestrianCVPipeline()

# Chạy trên webcam (0 là camera mặc định)
# Có thể đổi thành tên file video nếu muốn test: "video_test.mp4"
pipeline.run_on_video(0)