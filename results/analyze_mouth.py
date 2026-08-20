"""
Phân tích MAR + aspect ratio trên video để tune ngưỡng cho YawnDetector.

Output: in ra percentile distribution của MAR và aspect_ratio,
        + liệt kê các đoạn MAR > 0.55 kéo dài ≥ 1s (ngáp tiềm năng).
"""
import cv2
import math
import statistics
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.landmarks import (
    compute_mar, LEFT_EYE_EAR_IDX, RIGHT_EYE_EAR_IDX,
    MOUTH_TOP, MOUTH_BOTTOM, MOUTH_LEFT, MOUTH_RIGHT,
)


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _resolve_model() -> str:
    """M1: tim .task qua model_search_roots thay vi path cung results/."""
    from src.model_loader import model_search_roots, resolve_artifact
    found = resolve_artifact("holistic_landmarker.task", model_search_roots(ROOT))
    if found is None:
        raise SystemExit("Khong tim thay holistic_landmarker.task")
    return str(found)


def main(video_path):
    import mediapipe as mp
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"video: {video_path}  fps={fps:.1f}  {W}x{H}")

    landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(
        mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=_resolve_model()),
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1,
        )
    )

    mar_list, aspect_list = [], []
    high_mar_segments = []   # (start_frame, end_frame, peak_mar, mean_aspect)
    seg_start = None
    seg_peak_mar = 0.0
    seg_aspect_sum, seg_aspect_n = 0.0, 0
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        import mediapipe as mp_img
        mp_image = mp_img.Image(image_format=mp_img.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_image)
        if result and result.face_landmarks:
            face_lm = result.face_landmarks[0]
            pts_px = [(lm.x * W, lm.y * H) for lm in face_lm]
            mar = compute_mar(pts_px)
            vertical = _dist(pts_px[MOUTH_TOP], pts_px[MOUTH_BOTTOM])
            horizontal = _dist(pts_px[MOUTH_LEFT], pts_px[MOUTH_RIGHT])
            aspect = vertical / (horizontal + 1e-6)
            mar_list.append(mar)
            aspect_list.append(aspect)

            if mar > 0.55:
                if seg_start is None:
                    seg_start = frame_idx
                    seg_peak_mar = mar
                    seg_aspect_sum, seg_aspect_n = aspect, 1
                else:
                    seg_peak_mar = max(seg_peak_mar, mar)
                    seg_aspect_sum += aspect
                    seg_aspect_n += 1
            else:
                if seg_start is not None:
                    high_mar_segments.append((seg_start, frame_idx, seg_peak_mar,
                                              seg_aspect_sum / seg_aspect_n))
                    seg_start = None
                    seg_peak_mar = 0.0
                    seg_aspect_sum, seg_aspect_n = 0.0, 0

        frame_idx += 1

    cap.release()
    landmarker.close()

    if seg_start is not None:
        high_mar_segments.append((seg_start, frame_idx, seg_peak_mar,
                                  seg_aspect_sum / seg_aspect_n))

    print(f"\nframes processed: {frame_idx}")
    print(f"frames with face: {len(mar_list)}")
    if not mar_list:
        print("No faces detected.")
        return

    print(f"\nMAR distribution:")
    print(f"  min={min(mar_list):.3f}  max={max(mar_list):.3f}  mean={statistics.mean(mar_list):.3f}")
    sorted_mar = sorted(mar_list)
    print(f"  p50={sorted_mar[len(sorted_mar)//2]:.3f}")
    print(f"  p75={sorted_mar[int(len(sorted_mar)*0.75)]:.3f}")
    print(f"  p90={sorted_mar[int(len(sorted_mar)*0.90)]:.3f}")
    print(f"  p95={sorted_mar[int(len(sorted_mar)*0.95)]:.3f}")

    print(f"\naspect (vertical/horizontal) distribution:")
    print(f"  min={min(aspect_list):.3f}  max={max(aspect_list):.3f}  mean={statistics.mean(aspect_list):.3f}")
    sorted_a = sorted(aspect_list)
    print(f"  p50={sorted_a[len(sorted_a)//2]:.3f}")
    print(f"  p75={sorted_a[int(len(sorted_a)*0.75)]:.3f}")
    print(f"  p90={sorted_a[int(len(sorted_a)*0.90)]:.3f}")

    print(f"\nSegments with MAR > 0.55 (potential yawns): {len(high_mar_segments)}")
    for i, (s, e, peak_mar, mean_a) in enumerate(high_mar_segments):
        dur_sec = (e - s) / fps
        print(f"  [{i:02d}] frames {s}-{e}  dur={dur_sec:.2f}s  peak_mar={peak_mar:.3f}  mean_aspect={mean_a:.3f}")


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "results/video_annotated_landmark.mp4")