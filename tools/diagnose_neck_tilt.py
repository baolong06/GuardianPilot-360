"""
Deep diagnostic: Monitor ALL values related to nodding detection
Including pose landmarks detection rate, neck_tilt values, fusion state
"""
import argparse
import base64
import json
import sys
import time
from pathlib import Path

import cv2
import requests

API_BASE = "http://127.0.0.1:5000"

def main():
    parser = argparse.ArgumentParser(description="Deep neck-tilt diagnostic")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    # Check server
    res = requests.get(f"{API_BASE}/api/status")
    data = res.json()
    print(f"[Status] initialized={data['initialized']}")
    if not data['initialized']:
        print("[ERROR] Server not initialized!")
        sys.exit(1)

    if args.reset:
        requests.post(f"{API_BASE}/api/reset")
        print("[Reset] State reset")

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    print("\n" + "=" * 80)
    print("DEEP NECK TILT DIAGNOSTIC")
    print("=" * 80)
    print(f"{'T':>6} | {'has_pose':>9} | {'neck_tilt':>10} | {'has_neck':>9} | {'p_mlp':>7} | {'ema':>7} | {'N-alarm':>8} | {'alarm':>6}")
    print("-" * 80)

    start = time.time()
    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Downscale for inference
            small = cv2.resize(frame, (320, 240))
            _, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 75])
            data_url = "data:image/jpeg;base64," + base64.b64encode(buf).decode()

            res = requests.post(
                f"{API_BASE}/api/analyze_lite",
                json={"image": data_url},
                timeout=5
            )

            if res.status_code != 200:
                continue

            data = res.json()
            if not data.get("ok"):
                continue

            frame_count += 1
            elapsed = time.time() - start

            feat = data.get("features", {}) or {}

            neck_tilt = feat.get("neck_tilt")
            has_pose = feat.get("has_pose", False)
            p_mlp = data.get("p_mlp_drowsy", 0)
            ema = data.get("ema_prob", 0)
            neck_alarm = data.get("neck_alarm", False)
            alarm_on = data.get("alarm_on", False)

            # Format neck_tilt
            if neck_tilt is not None:
                try:
                    if float(neck_tilt) == float(neck_tilt):  # not NaN
                        nt_str = f"{neck_tilt:>10.2f}°"
                    else:
                        nt_str = "        NaN"
                except:
                    nt_str = f"{neck_tilt:>10}"
            else:
                nt_str = "        None"

            has_neck = "YES" if not (isinstance(neck_tilt, float) and neck_tilt != neck_tilt) else "NaN"

            print(f"{elapsed:>6.1f} | {str(has_pose):>9} | {nt_str} | {has_neck:>9} | {p_mlp:>7.3f} | {ema:>7.3f} | {str(neck_alarm):>8} | {str(alarm_on):>6}")

            if frame_count >= 150:
                break

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n[Stopped]")
    finally:
        cap.release()
        print(f"\nTotal frames: {frame_count}")

if __name__ == "__main__":
    main()
