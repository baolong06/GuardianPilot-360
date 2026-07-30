"""Lưu 1 frame giữa video ra PNG để kiểm tra bằng mắt."""
import cv2
VIDEO = r"E:/KhoiNghiep/GuardianPilot/Drowsiness Detection - Google Chrome 2026-07-30 05-48-18.mp4"
cap = cv2.VideoCapture(VIDEO)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
# Frame giữa video
cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
ok, frame = cap.read()
if ok:
    out = r"E:/KhoiNghiep/GuardianPilot/results/video_mid_frame.png"
    cv2.imwrite(out, frame)
    print(f"Saved mid frame to {out}")
    print(f"Frame shape: {frame.shape}, mean brightness: {frame.mean():.1f}")
else:
    print("Cannot read frame")
cap.release()
