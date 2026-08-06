"""
MediaPipe Holistic Landmarker wrapper với Multi-Person Support.
- Dùng Holistic landmarks để estimate face bounding box
- Chọn primary person (lớn nhất + gần center + lower position)
- Driver monitoring: ưu tiên người ở lower-center frame
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision as mp_vision

# ── Global singleton ────────────────────────────────────────────────────────────
_holistic: mp_vision.HolisticLandmarker | None = None
_model_path: str | None = None

# ── Primary person selection config ────────────────────────────────────────────
MIN_FACE_SIZE_PX = 60      # face tối thiểu 60px (small faces = false positive)
MIN_FACE_SIZE_RATIO = 0.02  # face >= 2% của frame


@dataclass
class DebugInfo:
    """Debug info cho multi-person selection."""
    n_faces: int = 0
    face_scores: List[dict] = field(default_factory=list)
    primary_idx: int = -1
    crop_region: Optional[Tuple[int,int,int,int]] = None


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
    """Lazy-load HolisticLandmarker singleton."""
    global _holistic, _model_path
    if _holistic is None or _model_path != model_path:
        options = mp_vision.HolisticLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=mp_vision.RunningMode.IMAGE,
        )
        _holistic = mp_vision.HolisticLandmarker.create_from_options(options)
        _model_path = model_path
    return _holistic


def _estimate_face_bbox(face_landmarks, img_w, img_h) -> Tuple[int,int,int,int]:
    """
    Estimate bounding box từ face landmarks.
    FaceMesh có ~468 điểm, lấy min/max x,y.
    Returns (x_min, y_min, x_max, y_max) in pixels.
    """
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
    img_w: int, img_h: int
) -> Tuple[float, dict]:
    """
    Tính score cho mỗi person để chọn primary.
    
    Scoring factors:
    1. Face size: lớn hơn = gần hơn = ưu tiên
    2. Center proximity: gần center frame tốt hơn  
    3. Bottom position: lower-center = vị trí lái xe
    4. Has pose: có pose landmarks = body visible = real person
    """
    x_min, y_min, x_max, y_max = face_bbox
    fw, fh = x_max - x_min, y_max - y_min
    
    # 1. Size score
    face_area = fw * fh
    size_ratio = face_area / (img_w * img_h)
    
    # Skip tiny faces
    if fw < MIN_FACE_SIZE_PX or size_ratio < MIN_FACE_SIZE_RATIO:
        return -1.0, {"reason": "too_small"}
    
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
        new_lm = type(lm)(
            x=orig_x / img_w,
            y=orig_y / img_h,
            z=getattr(lm, 'z', 0.0)
        )
        result.append(new_lm)
    
    return result


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
    img_h, img_w = frame_bgr.shape[:2]
    debug = DebugInfo()
    
    # ── Convert to RGB ────────────────────────────────────────────────────────
    rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    
    # ── Step 1: Run Holistic ────────────────────────────────────────────────
    holistic = get_landmarker(model_path)
    result = holistic.detect(mp_image)
    
    # Kiểm tra face landmarks
    face_landmarks_list = result.face_landmarks
    n_faces = len(face_landmarks_list) if face_landmarks_list else 0
    debug.n_faces = n_faces
    
    if n_faces == 0:
        if return_debug:
            return None, debug
        return None
    
    # ── Step 2: Estimate bbox và score mỗi person ───────────────────────────
    scored = []
    for i, face_lm in enumerate(face_landmarks_list):
        pose_lm = None
        if result.pose_landmarks and i < len(result.pose_landmarks):
            pose_lm = result.pose_landmarks[i]
        
        bbox = _estimate_face_bbox(face_lm, img_w, img_h)
        score, info = _score_person(bbox, pose_lm, img_w, img_h)
        info["idx"] = i
        scored.append((score, info))
    
    # Sort by score descending
    scored.sort(key=lambda x: x[0], reverse=True)
    debug.face_scores = [{"score": s, **c} for s, c in scored]
    
    # ── Step 3: Select primary ───────────────────────────────────────────────
    primary_score, primary_info = scored[0]
    
    if primary_score < 0:
        if return_debug:
            return None, debug
        return None
    
    primary_idx = primary_info["idx"]
    debug.primary_idx = primary_idx
    
    # ── Step 4: Nếu có nhiều hơn 1 person, crop & re-run ───────────────────
    if n_faces > 1:
        # Crop vùng primary person
        primary_bbox = primary_info["face_bbox"]
        crop_x, crop_y, crop_x2, crop_y2 = _expand_crop(
            *primary_bbox, img_w, img_h, margin=0.25
        )
        debug.crop_region = (crop_x, crop_y, crop_x2, crop_y2)
        
        crop_w, crop_h = crop_x2 - crop_x, crop_y2 - crop_y
        
        # Crop frame
        cropped_rgb = rgb[crop_y:crop_y2, crop_x:crop_x2]
        cropped_mp = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cropped_rgb
        )
        
        # Re-run holistic on cropped
        cropped_result = holistic.detect(cropped_mp)
        
        # Transform landmarks về original coords
        transformed_face = []
        if cropped_result.face_landmarks:
            transformed_face = _transform_landmarks(
                cropped_result.face_landmarks[0],
                crop_x, crop_y, crop_w, crop_h, img_w, img_h
            )
        
        transformed_pose = []
        if cropped_result.pose_landmarks:
            transformed_pose = _transform_landmarks(
                cropped_result.pose_landmarks[0],
                crop_x, crop_y, crop_w, crop_h, img_w, img_h
            )
        
        out_result = TransformedResult([transformed_face], [transformed_pose], debug)
        
    else:
        # Chỉ 1 person, dùng trực tiếp
        debug.crop_region = None
        out_result = TransformedResult(face_landmarks_list, result.pose_landmarks, debug)
    
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
