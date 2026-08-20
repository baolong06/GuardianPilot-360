"""
Camera intrinsics cho head-pose (M9).

`compute_head_pose()` trong landmarks.py trước đây hard-code:
    focal_length = img_w
    cx, cy       = img_w / 2, img_h / 2
    dist_coeffs  = zeros(4)

Đó là giả định "camera pinhole chưa calib". Hệ quả: pitch/yaw/roll trả về là góc
**tương đối**, chỉ dùng được qua delta-so-với-baseline, KHÔNG phải góc tuyệt đối.
Mọi ngưỡng theo độ (ví dụ `yaw_thresh_deg = 25`) vì thế gắn chặt với ống kính
đang dùng và phải chỉnh lại khi đổi camera.

Module này giữ NGUYÊN giá trị mặc định đó (không đổi hành vi), nhưng cho phép
nạp thông số calib thật qua biến môi trường:

    CAMERA_FOCAL_PX        focal length theo pixel (fx = fy)
    CAMERA_FOCAL_X / _Y    focal riêng từng trục (ưu tiên hơn CAMERA_FOCAL_PX)
    CAMERA_CX / CAMERA_CY  principal point theo pixel
    CAMERA_DIST_COEFFS     "k1,k2,p1,p2[,k3]" cho cv2.solvePnP

Xem docs/CAMERA_CALIBRATION.md.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

ENV_FOCAL = "CAMERA_FOCAL_PX"
ENV_FOCAL_X = "CAMERA_FOCAL_X"
ENV_FOCAL_Y = "CAMERA_FOCAL_Y"
ENV_CX = "CAMERA_CX"
ENV_CY = "CAMERA_CY"
ENV_DIST = "CAMERA_DIST_COEFFS"


@dataclass(frozen=True)
class CameraIntrinsics:
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    calibrated: bool
    source: str


_cache: dict[tuple, CameraIntrinsics] = {}


def _env_float(name: str) -> float | None:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return None
    try:
        return float(raw)
    except ValueError:
        logger.warning("%s=%r không phải số — bỏ qua.", name, raw)
        return None


def _env_dist() -> np.ndarray | None:
    raw = os.getenv(ENV_DIST)
    if raw is None or not raw.strip():
        return None
    try:
        values = [float(p) for p in raw.replace(";", ",").split(",") if p.strip()]
    except ValueError:
        logger.warning("%s=%r không parse được — bỏ qua.", ENV_DIST, raw)
        return None
    if len(values) not in (4, 5, 8):
        logger.warning("%s cần 4, 5 hoặc 8 hệ số, nhận %d — bỏ qua.", ENV_DIST, len(values))
        return None
    return np.array(values, dtype=np.float64).reshape(-1, 1)


def _env_signature() -> tuple:
    return tuple(
        os.getenv(name) for name in
        (ENV_FOCAL, ENV_FOCAL_X, ENV_FOCAL_Y, ENV_CX, ENV_CY, ENV_DIST)
    )


def get_camera_intrinsics(img_w: int, img_h: int) -> CameraIntrinsics:
    """
    Trả về intrinsics cho ảnh kích thước (img_w, img_h).

    Mặc định (không set env nào) tái tạo CHÍNH XÁC giả định cũ, nên kết quả
    head-pose không đổi so với trước khi có module này.
    """
    key = (int(img_w), int(img_h), _env_signature())
    cached = _cache.get(key)
    if cached is not None:
        return cached

    fx = _env_float(ENV_FOCAL_X) or _env_float(ENV_FOCAL)
    fy = _env_float(ENV_FOCAL_Y) or _env_float(ENV_FOCAL)
    cx = _env_float(ENV_CX)
    cy = _env_float(ENV_CY)
    dist = _env_dist()

    calibrated = any(v is not None for v in (fx, fy, cx, cy, dist))

    if fx is None:
        fx = float(img_w)          # giả định cũ: focal = chiều rộng ảnh
    if fy is None:
        fy = fx
    if cx is None:
        cx = img_w / 2.0
    if cy is None:
        cy = img_h / 2.0
    if dist is None:
        dist = np.zeros((4, 1), dtype=np.float64)

    matrix = np.array(
        [[fx, 0.0, cx],
         [0.0, fy, cy],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    intrinsics = CameraIntrinsics(
        camera_matrix=matrix,
        dist_coeffs=dist,
        calibrated=calibrated,
        source="env" if calibrated else "uncalibrated_default",
    )
    _cache[key] = intrinsics
    return intrinsics


def describe() -> dict:
    """Tóm tắt cấu hình cho /api/runtime-profile."""
    return {
        "calibrated": any(
            os.getenv(name) for name in
            (ENV_FOCAL, ENV_FOCAL_X, ENV_FOCAL_Y, ENV_CX, ENV_CY, ENV_DIST)
        ),
        "env_vars": [ENV_FOCAL, ENV_FOCAL_X, ENV_FOCAL_Y, ENV_CX, ENV_CY, ENV_DIST],
        "note": (
            "Chưa calib → pitch/yaw/roll là góc tương đối, chỉ dùng qua "
            "delta-với-baseline. Xem docs/CAMERA_CALIBRATION.md."
        ),
    }


def clear_cache() -> None:
    """Dùng trong test khi đổi biến môi trường."""
    _cache.clear()
