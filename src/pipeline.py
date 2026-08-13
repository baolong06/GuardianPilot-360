"""
MediaPipe Holistic Landmarker wrapper với Multi-Person Support.
- Dùng Holistic landmarks để estimate face bounding box
- Chọn primary person (lớn nhất + gần center + lower position)
- Driver monitoring: ưu tiên người ở lower-center frame
"""
from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision as mp_vision

# ── Force CPU-only to avoid GPU driver conflicts in Docker ───────────────────
os.environ.setdefault("MEDIAPIPE_DISABLE_GPU", "1")

# ── Global singleton ────────────────────────────────────────────────────────────
_holistic: mp_vision.HolisticLandmarker | None = None
_model_path: str | None = None
_legacy_holistic = None

from .runtime_profile import get_runtime_profile

# ── Frame resize config (theo EDGE_PROFILE) ──────────────────────────────────
_profile = get_runtime_profile()
MEDIAPIPE_INPUT_WIDTH  = int(_profile["mediapipe_width"])
MEDIAPIPE_INPUT_HEIGHT = int(_profile["mediapipe_height"])
MEDIAPIPE_INPUT_SIZE   = (MEDIAPIPE_INPUT_WIDTH, MEDIAPIPE_INPUT_HEIGHT)

# ── Primary person selection config ────────────────────────────────────────────
# Face size thresholds — giảm để nhận diện được khi người ngồi xa camera.
# Trade-off: threshold thấp hơn → nhạy hơn với small faces, nhưng có thể pick up
# face artifact (poster, ảnh nhỏ trên tường). Anti-noise: chỉ giảm threshold khi
# single-person (n_faces=1) — nếu multi-person vẫn giữ threshold cao để tránh
# nhầm person xa với face artifact.
MIN_FACE_SIZE_PX = 35       # giảm từ 60 → 35 (face xa ~80px ở 640x480, sau resize về 480x360 ~60px)
MIN_FACE_SIZE_RATIO = 0.01  # giảm từ 0.02 → 0.01 (1% của frame là hợp lý cho face xa)
MIN_FACE_SIZE_PX_STRICT = 60  # threshold cũ — dùng cho multi-person để tránh false positive
MIN_FACE_SIZE_RATIO_STRICT = 0.02


@dataclass
class DebugInfo:
    """Debug info cho multi-person selection."""
    n_faces: int = 0
    face_scores: List[dict] = field(default_factory=list)
    primary_idx: int = -1
    crop_region: Optional[Tuple[int,int,int,int]] = None
    distant_fallback: bool = False  # True nếu primary được chọn qua distance fallback (single-person)


class TransformedResult:
    """
    Wrapper giữ holistic result với coordinates đã transform về original frame.
    Interface tương thích ngược với MediaPipe result.
    """
    def __init__(self, face_lm, pose_lm, debug: DebugInfo):
        self.face_landmarks = face_lm  # List[List[ScaledLandmark]]
        self.pose_landmarks = pose_lm   # List[List[ScaledLandmark]]
        self.debug = debug


def get_landmarker(model_path: str) -> mp_vision.HolisticLandmarker:
    """Lazy-load HolisticLandmarker singleton (CPU-only)."""
    global _holistic, _model_path
    if _holistic is None or _model_path != model_path:
        # NOTE: HolisticLandmarkerOptions in mediapipe 0.10.14 does NOT
        # accept `num_threads` (it's only in FaceDetector / PoseDetector).
        # CPU thread count is controlled via env var `OMP_NUM_THREADS` instead.
        options = mp_vision.HolisticLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=mp_vision.RunningMode.IMAGE,
        )
        _holistic = mp_vision.HolisticLandmarker.create_from_options(options)
        _model_path = model_path
    return _holistic


def _ensure_list(landmarks):
    """
    Chuẩn hóa landmarks thành list, bất kể input là:
    - List[NormalizedLandmark] (mp.solutions)
    - List[List[NormalizedLandmark]] (TransformedResult wrapped - take first)
    - Single NormalizedLandmark object (mp.tasks có thể trả)
    """
    if landmarks is None:
        return []
    # TransformedResult: [[lm1, lm2, ...]] → lấy list đầu tiên
    if isinstance(landmarks, list):
        if not landmarks:
            return []
        if isinstance(landmarks[0], list):
            return landmarks[0]
        return landmarks
    # Single object có .x → wrap
    if hasattr(landmarks, 'x'):
        return [landmarks]
    return []


def _estimate_face_bbox(face_landmarks, img_w, img_h) -> Tuple[int,int,int,int]:
    """
    Estimate bounding box từ face landmarks.
    FaceMesh có ~468 điểm, lấy min/max x,y.
    Returns (x_min, y_min, x_max, y_max) in pixels.
    """
    face_landmarks = _ensure_list(face_landmarks)
    if not face_landmarks:
        return None

    xs = [lm.x * img_w for lm in face_landmarks]
    ys = [lm.y * img_h for lm in face_landmarks]

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    return int(x_min), int(y_min), int(x_max), int(y_max)


def _score_person(
    face_bbox: Tuple[int,int,int,int],
    pose_landmarks,
    img_w: int, img_h: int,
    n_faces_total: int = 1,
) -> Tuple[float, dict]:
    """
    Tính score cho mỗi person để chọn primary.

    Scoring factors:
    1. Face size: lớn hơn = gần hơn = ưu tiên
    2. Center proximity: gần center frame tốt hơn
    3. Bottom position: lower-center = vị trí lái xe
    4. Has pose: có pose landmarks = body visible = real person

    Distance handling (khi người ngồi xa camera):
    - Multi-person (n_faces_total > 1): strict threshold — giữ nguyên
      MIN_FACE_SIZE_PX_STRICT/RATIO để tránh nhầm person xa với poster/artifact.
    - Single-person (n_faces_total == 1): relaxed threshold — dùng
      MIN_FACE_SIZE_PX/RATIO thấp hơn để KHÔNG bỏ lỡ user khi chỉ có 1 mặt.
    """
    x_min, y_min, x_max, y_max = face_bbox
    fw, fh = x_max - x_min, y_max - y_min

    # 1. Size score
    face_area = fw * fh
    size_ratio = face_area / (img_w * img_h)

    # Chọn threshold theo context: single-person → cho phép face nhỏ; multi → strict
    if n_faces_total <= 1:
        min_px    = MIN_FACE_SIZE_PX
        min_ratio = MIN_FACE_SIZE_RATIO
    else:
        min_px    = MIN_FACE_SIZE_PX_STRICT
        min_ratio = MIN_FACE_SIZE_RATIO_STRICT

    # Skip tiny faces
    if fw < min_px or size_ratio < min_ratio:
        return -1.0, {"reason": "too_small", "fw": fw, "fh": fh,
                       "size_ratio": round(size_ratio, 4)}
    
    size_score = size_ratio * 2000  # scale up
    
    # 2. Face center
    fc_x = (x_min + x_max) / 2
    fc_y = (y_min + y_max) / 2
    
    # Distance to center (normalized 0-1)
    dist_center = math.hypot(
        (fc_x - img_w/2) / img_w,
        (fc_y - img_h/2) / img_h
    )
    # Gần center hơn = score cao hơn
    center_score = max(0, 1.0 - dist_center * 1.2) * 80
    
    # 3. Bottom position score (driver monitoring)
    # Người lái xe ngồi ở lower portion của frame
    norm_y = fc_y / img_h
    if 0.35 <= norm_y <= 0.85:
        bottom_score = 70  # sweet spot cho driver position
    elif norm_y > 0.85:
        bottom_score = 50  # quá low = có thể body, not face
    elif norm_y > 0.6:
        bottom_score = 60
    else:
        bottom_score = 30  # quá cao = có thể background
    
    # 4. Pose bonus (có pose = real person, không phải artifact)
    pose_bonus = 40 if pose_landmarks else 0
    
    total = size_score + center_score + bottom_score + pose_bonus
    
    return total, {
        "face_bbox": face_bbox,
        "face_size": (fw, fh),
        "size_ratio": round(size_ratio, 4),
        "face_center": (round(fc_x, 1), round(fc_y, 1)),
        "norm_y": round(norm_y, 3),
        "dist_center": round(dist_center, 3),
        "size_score": round(size_score, 1),
        "center_score": round(center_score, 1),
        "bottom_score": bottom_score,
        "pose_bonus": pose_bonus,
        "total": round(total, 1),
    }


def _expand_crop(
    x_min: int, y_min: int, x_max: int, y_max: int,
    img_w: int, img_h: int, margin: float = 0.20
) -> Tuple[int, int, int, int]:
    """Expand bbox với margin, giới hạn trong frame."""
    fw, fh = x_max - x_min, y_max - y_min
    dx, dy = int(fw * margin), int(fh * margin)
    return (
        max(0, x_min - dx),
        max(0, y_min - dy),
        min(img_w, x_max + dx),
        min(img_h, y_max + dy),
    )


def _transform_landmarks(
    landmarks,
    crop_x: int, crop_y: int, crop_w: int, crop_h: int,
    img_w: int, img_h: int
) -> List:
    """Transform normalized landmarks từ cropped → original coordinates."""
    if not landmarks:
        return []

    result = []
    for lm in landmarks:
        # Vì crop region được crop từ full frame,
        # normalized coords trong crop = (orig_x - crop_x) / crop_w
        # Vậy orig_x = lm.x * crop_w + crop_x
        orig_x = lm.x * crop_w + crop_x
        orig_y = lm.y * crop_h + crop_y

        # Normalize về 0-1
        new_lm = _make_landmark(
            x=orig_x / img_w,
            y=orig_y / img_h,
            z=getattr(lm, 'z', 0.0),
            visibility=getattr(lm, 'visibility', None),
            presence=getattr(lm, 'presence', None),
        )
        result.append(new_lm)

    return result


def _make_landmark(x: float, y: float, z: float = 0.0,
                   visibility=None, presence=None):
    """
    Tạo một NormalizedLandmark tương thích với cả MediaPipe solutions API
    và MediaPipe Tasks API. Hai API trả về 2 loại object khác nhau:
      - mp.solutions.*: NormalizedLandmark(landmark.x, .y, .z, .visibility, .presence)
      - mp.tasks.*:    NormalizedLandmark(x=, y=, z=, visibility=, presence=)
    Hàm này thử cả 2 cách khởi tạo.
    """
    # Cách 1: positional (mp.solutions)
    try:
        import mediapipe.framework.formats.landmark_pb2 as landmark_pb2
        lm = landmark_pb2.NormalizedLandmark(
            x=float(x), y=float(y), z=float(z),
            visibility=visibility if visibility is not None else 0.0,
            presence=presence if presence is not None else 0.0,
        )
        return lm
    except Exception:
        pass
    # Cách 2: keyword args (mp.tasks)
    try:
        import mediapipe.tasks.python.components.containers.landmark_module as lm_mod
        lm = lm_mod.NormalizedLandmark(
            x=float(x), y=float(y), z=float(z),
            visibility=visibility, presence=presence,
        )
        return lm
    except Exception:
        pass
    # Fallback: simple namespace
    from types import SimpleNamespace
    return SimpleNamespace(x=float(x), y=float(y), z=float(z),
                           visibility=visibility, presence=presence)


def run_holistic(
    frame_bgr: np.ndarray,
    model_path: str,
    return_debug: bool = False
) -> Tuple:
    """
    Run Holistic với multi-person primary selection.
    
    Algorithm:
    1. Run Holistic trên full frame (lấy tất cả people)
    2. Với mỗi person: estimate face bbox từ face landmarks
    3. Score mỗi person (size + center + bottom + pose)
    4. Chọn person có score cao nhất
    5. Nếu có nhiều person trong kết quả, crop + re-run trên primary
    
    Returns:
        - TransformedResult if return_debug=False
        - (TransformedResult, DebugInfo) if return_debug=True
        - None if no face detected
    """
    logger = logging.getLogger("mediapipe")
    img_h, img_w = frame_bgr.shape[:2]
    debug = DebugInfo()

    # ── Resize to fixed MediaPipe input size ──────────────────────────────────
    # CRITICAL: Holistic Landmarker graph caches segmentation matrices internally
    # and requires all frames to have the same dimensions. Mixing 640x480 webcam
    # frames with 320x240 lite-mode frames causes
    # "current_mat->rows == previous_mat->rows (600 vs 240)" crash.
    # Solution: resize every frame to MEDIAPIPE_INPUT_SIZE before inference,
    # then scale landmarks back to original image coords.
    scale_x = img_w / MEDIAPIPE_INPUT_WIDTH
    scale_y = img_h / MEDIAPIPE_INPUT_HEIGHT
    rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
    if (img_w, img_h) != MEDIAPIPE_INPUT_SIZE:
        rgb_resized = cv2.resize(rgb, MEDIAPIPE_INPUT_SIZE,
                                 interpolation=cv2.INTER_AREA)
    else:
        rgb_resized = rgb
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_resized)

    # ── Step 1: Run Holistic ────────────────────────────────────────────────
    # Preferred: Holistic task model (.task). Fallback: mp.solutions.holistic
    # when task file is not available in runtime (common in fresh Docker env).
    #
    # NOTE: We wrap detect() in try/except because MediaPipe C++ layer can crash
    # with "packet is empty" assertion on first frame or GPU conflicts.
    # Returning None gracefully lets the caller treat this as "no face detected".
    face_landmarks_list = []
    pose_landmarks_list = []

    try:
        if Path(model_path).is_file():
            holistic = get_landmarker(model_path)
            result = holistic.detect(mp_image)
            # MediaPipe Tasks API returns face_landmarks as a flat list
            # of 478 NormalizedLandmark for the primary face (or empty list).
            # Wrap as List[List[lm]] to keep multi-person interface consistent.
            flat_face = result.face_landmarks or []
            flat_pose = result.pose_landmarks or []
            face_landmarks_list = [flat_face] if flat_face else []
            pose_landmarks_list = [flat_pose] if flat_pose else []
        else:
            global _legacy_holistic
            if _legacy_holistic is None:
                _legacy_holistic = mp.solutions.holistic.Holistic(
                    static_image_mode=True,
                    model_complexity=1,
                    refine_face_landmarks=True,
                )
            legacy = _legacy_holistic.process(rgb_resized)
            # mp.solutions returns face_landmarks.landmark as flat list
            face_landmarks_list = [list(legacy.face_landmarks.landmark)] if legacy.face_landmarks else []
            pose_landmarks_list = [list(legacy.pose_landmarks.landmark)] if legacy.pose_landmarks else []
    except Exception as exc:
        logger.warning("MediaPipe holistic.detect() failed: %s. Returning None.", exc)
        if return_debug:
            return None, debug
        return None
    
    # Kiểm tra face landmarks
    n_faces = len(face_landmarks_list) if face_landmarks_list else 0
    debug.n_faces = n_faces
    
    if n_faces == 0:
        if return_debug:
            return None, debug
        return None
    
    # ── Step 2: Estimate bbox và score mỗi person ───────────────────────────
    # Note: landmarks are normalized (0-1), so we use original img_w, img_h
    # for bbox/score calculations regardless of MediaPipe input size.
    scored = []
    for i, face_lm in enumerate(face_landmarks_list):
        pose_lm = None
        if pose_landmarks_list and i < len(pose_landmarks_list):
            pose_lm = pose_landmarks_list[i]

        bbox = _estimate_face_bbox(face_lm, img_w, img_h)
        # Truyền n_faces để _score_person chọn threshold phù hợp
        # (single-person → relaxed, multi-person → strict).
        score, info = _score_person(bbox, pose_lm, img_w, img_h,
                                    n_faces_total=n_faces)
        info["idx"] = i
        scored.append((score, info))
    
    # Sort by score descending
    scored.sort(key=lambda x: x[0], reverse=True)
    debug.face_scores = [{"score": s, **c} for s, c in scored]

    # ── Step 3: Select primary ───────────────────────────────────────────────
    primary_score, primary_info = scored[0]

    # ── Distance fallback (single-person case) ─────────────────────────────
    # Khi user ngồi xa, face có thể < MIN_FACE_SIZE_PX → bị reject với score=-1.
    # Nếu chỉ có 1 person duy nhất trong frame → KHÔNG có candidate khác để
    # chọn. Thay vì trả None (khiến user phải "lại gần" mới hoạt động), ta
    # dùng luôn person duy nhất này làm primary với score thấp + flag
    # "distant" để các module downstream biết mà giảm confidence (vd: bỏ
    # qua LSTM, tăng hysteresis threshold một chút — chưa implement ở đây).
    if primary_score < 0 and n_faces == 1:
        only_score, only_info = scored[0]
        primary_score = 0.0  # cho qua với score=0
        primary_info = only_info
        primary_info["distant"] = True
        primary_info["reason"] = "distant_single_person_fallback"
    elif primary_score < 0:
        if return_debug:
            return None, debug
        return None

    primary_idx = primary_info["idx"]
    debug.primary_idx = primary_idx
    debug.distant_fallback = primary_info.get("distant", False)
    
    # ── Step 4: Nếu có nhiều hơn 1 person, crop & re-run ───────────────────
    if n_faces > 1:
        # Crop vùng primary person. primary_bbox is in original image coords,
        # but we need to crop from rgb_resized (MediaPipe input size).
        primary_bbox = primary_info["face_bbox"]
        # Scale bbox from original coords → MediaPipe input coords
        sx_b = primary_bbox[0] / scale_x
        sy_b = primary_bbox[1] / scale_y
        sx_b2 = primary_bbox[2] / scale_x
        sy_b2 = primary_bbox[3] / scale_y
        crop_x, crop_y, crop_x2, crop_y2 = _expand_crop(
            sx_b, sy_b, sx_b2, sy_b2,
            MEDIAPIPE_INPUT_WIDTH, MEDIAPIPE_INPUT_HEIGHT, margin=0.25
        )
        debug.crop_region = (crop_x, crop_y, crop_x2, crop_y2)

        crop_w_mp = crop_x2 - crop_x
        crop_h_mp = crop_y2 - crop_y

        # Crop from resized RGB
        cropped_rgb = rgb_resized[crop_y:crop_y2, crop_x:crop_x2]
        cropped_mp = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cropped_rgb
        )

        # Re-run holistic on cropped
        cropped_result = holistic.detect(cropped_mp)

        # Transform landmarks from MediaPipe input coords → original image coords
        # Note: pass original img_w, img_h so _transform_landmarks uses correct
        # scale factors (crop_w * scale_x = original pixel size).
        transformed_face = []
        if cropped_result.face_landmarks:
            transformed_face = _transform_landmarks(
                cropped_result.face_landmarks[0],
                crop_x * scale_x, crop_y * scale_y,
                crop_w_mp * scale_x, crop_h_mp * scale_y,
                img_w, img_h
            )

        transformed_pose = []
        if cropped_result.pose_landmarks:
            transformed_pose = _transform_landmarks(
                cropped_result.pose_landmarks[0],
                crop_x * scale_x, crop_y * scale_y,
                crop_w_mp * scale_x, crop_h_mp * scale_y,
                img_w, img_h
            )

        out_result = TransformedResult([transformed_face], [transformed_pose], debug)

    else:
        # Chỉ 1 person, dùng trực tiếp
        debug.crop_region = None
        out_result = TransformedResult(face_landmarks_list, pose_landmarks_list, debug)
    
    if return_debug:
        return out_result, debug
    return out_result


# Alias cho backward compatibility
def run_holistic_simple(frame_bgr: np.ndarray, model_path: str):
    """
    Wrapper đơn giản, chỉ trả về MediaPipe result thuần.
    Dùng cho cases không cần multi-person handling.
    """
    global _holistic, _model_path
    if _holistic is None or _model_path != model_path:
        options = mp_vision.HolisticLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=mp_vision.RunningMode.IMAGE,
        )
        _holistic = mp_vision.HolisticLandmarker.create_from_options(options)
        _model_path = model_path
    
    rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    return _holistic.detect(mp_image)
