"""
run.py — Entry point chạy GuardianPilot 360

Usage:
  python run.py                          # webcam real-time (default)
  python run.py --video path/to/clip.mp4 # chạy trên video có sẵn
  python run.py --no-display             # không hiển thị cửa sổ OpenCV
  python run.py --eeg                    # bật M2 (cần sensor EEG/EOG)
"""

import argparse
import os
import sys

# Đường dẫn tuyệt đối tới thư mục gốc project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from guardian_pilot import GuardianPilot360System


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GuardianPilot 360 — Driver Monitoring System"
    )
    parser.add_argument(
        "--video", type=str, default=None,
        help="Đường dẫn tới file video để test offline. Nếu không đặt → dùng webcam."
    )
    parser.add_argument(
        "--camera", type=int, default=0,
        help="Camera index (default: 0)"
    )
    parser.add_argument(
        "--fps", type=float, default=15.0,
        help="Tốc độ xử lý frame/giây (default: 15)"
    )
    parser.add_argument(
        "--eeg", action="store_true",
        help="Bật Agent M2 (cần sensor EEG/EOG thật được kết nối)"
    )
    parser.add_argument(
        "--no-display", action="store_true",
        help="Tắt cửa sổ OpenCV (chạy headless)"
    )
    parser.add_argument(
        "--audit-log", type=str, default="guardian_pilot_audit.log",
        help="Đường dẫn file audit log (JSONL)"
    )

    args = parser.parse_args()

    print("""
╔══════════════════════════════════════════════════════╗
║       GuardianPilot 360 — DMS Multi-Agent System     ║
║  M1: Drowsiness  M2: Microsleep  M3: Distracted      ║
║  M4: Landmark+Gaze  |  Orchestrator  |  Knowledge Graph ║
╚══════════════════════════════════════════════════════╝
""")

    # Khởi tạo toàn bộ hệ thống
    system = GuardianPilot360System.from_model_dir(
        base_dir             = BASE_DIR,
        sensor_eeg_available = args.eeg,
        audit_log            = args.audit_log,
    )

    display = not args.no_display

    if args.video:
        # Chạy trên video (offline test)
        system.run_on_video(
            video_path = args.video,
            target_fps = args.fps,
            display    = display,
        )
    else:
        # Chạy real-time trên webcam
        system.run_on_camera(
            camera_index = args.camera,
            target_fps   = args.fps,
            display      = display,
        )


if __name__ == "__main__":
    main()
