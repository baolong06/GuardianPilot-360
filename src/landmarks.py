"""
Feature extraction từ MediaPipe Holistic Landmarker.
Trả về EAR, MAR, head-pose (pitch/yaw/roll), neck-tilt.
"""
import math
import numpy as np
import cv2

from .camera import get_camera_intrinsics

# ── Landmark indices (MediaPipe FaceMesh 468/478 topology) ──────────────────
LEFT_EYE_EAR_IDX  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_EAR_IDX = [33, 160, 158, 133, 153, 144]
MOUTH_TOP, MOUTH_BOTTOM = 13, 14
MOUTH_LEFT, MOUTH_RIGHT = 61, 291

HEADPOSE_IDX = {
    "nose_tip":          1,
    "chin":              152,
    "left_eye_corner":   33,
    "right_eye_corner":  263,
    "left_mouth":        61,
    "right_mouth":       291,
}
MODEL_3D_POINTS = np.array([
    (0.0,    0.0,    0.0),
    (0.0,  -330.0,  -65.0),
    (-225.0, 170.0, -135.0),
    (225.0,  170.0, -135.0),
    (-150.0, -150.0, -125.0),
    (150.0,  -150.0, -125.0),
], dtype=np.float64)


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def compute_ear(pts, idxs):
    p1, p2, p3, p4, p5, p6 = (pts[i] for i in idxs)
    return (_dist(p2, p6) + _dist(p3, p5)) / (2.0 * _dist(p1, p4) + 1e-6)


def compute_mar(pts):
    top   = pts[MOUTH_TOP]
    bot   = pts[MOUTH_BOTTOM]
    left  = pts[MOUTH_LEFT]
    right = pts[MOUTH_RIGHT]
    return _dist(top, bot) / (_dist(left, right) + 1e-6)


def compute_mouth_aspect(pts):
    """
    Aspect ratio thô của miệng: vertical / horizontal.
    - Ngáp: thường 0.7-1.2 (miệng mở rộng theo chiều dọc → tròn/dọc)
    - Nói chuyện: 0.3-0.5 (miệng mở hẹp theo chiều ngang khi phát âm)
    Trả về vertical và horizontal (px) kèm aspect để có thể dùng cho calibration.
    """
    vertical   = _dist(pts[MOUTH_TOP],    pts[MOUTH_BOTTOM])
    horizontal = _dist(pts[MOUTH_LEFT],   pts[MOUTH_RIGHT])
    aspect     = vertical / (horizontal + 1e-6)
    return vertical, horizontal, aspect


def compute_head_pose(pts_px, img_w, img_h, intrinsics=None):
    """
    Trả về (pitch, yaw, roll) theo độ, hoặc None nếu solvePnP thất bại.

    M9: thông số camera lấy từ `src.camera.get_camera_intrinsics()`. Khi chưa
    calib (mặc định) hàm đó trả về đúng giả định cũ — focal = img_w, tâm ảnh là
    principal point, không méo — nên kết quả KHÔNG đổi so với trước.
    Hệ quả của việc chưa calib: pitch/yaw/roll là góc TƯƠNG ĐỐI, chỉ nên dùng
    qua delta-so-với-baseline. Xem docs/CAMERA_CALIBRATION.md.

    Args:
        intrinsics: CameraIntrinsics tuỳ chọn (dùng để test / ép giá trị).
    """
    image_points = np.array(
        [pts_px[HEADPOSE_IDX[k]] for k in
         ("nose_tip", "chin", "left_eye_corner", "right_eye_corner",
          "left_mouth", "right_mouth")],
        dtype=np.float64,
    )
    if intrinsics is None:
        intrinsics = get_camera_intrinsics(img_w, img_h)
    cam_mat = intrinsics.camera_matrix
    dist_coef = intrinsics.dist_coeffs
    ok, rvec, _ = cv2.solvePnP(
        MODEL_3D_POINTS, image_points, cam_mat, dist_coef,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return None
    rmat, _ = cv2.Rodrigues(rvec)
    sy = math.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
    if sy < 1e-6:
        return None
    pitch = math.degrees(math.atan2(-rmat[2, 0], sy))
    yaw   = math.degrees(math.atan2(rmat[1, 0], rmat[0, 0]))
    roll  = math.degrees(math.atan2(rmat[2, 1], rmat[2, 2]))
    return pitch, yaw, roll


def compute_neck_tilt(pose_pts_px):
    """
    Góc (độ) giữa vector (trung điểm vai → mũi) và phương thẳng đứng.
    0° = đầu thẳng; tăng = cúi/nghiêng. None nếu thiếu pose.
    """
    # pose index: 0=nose, 11=left_shoulder, 12=right_shoulder
    try:
        nose = pose_pts_px[0]
        l_sh = pose_pts_px[11]
        r_sh = pose_pts_px[12]
    except IndexError:
        return None
    mid_sh = ((l_sh[0] + r_sh[0]) / 2.0, (l_sh[1] + r_sh[1]) / 2.0)
    dx = nose[0] - mid_sh[0]
    dy = mid_sh[1] - nose[1]   # y tăng xuống → đảo để "lên" là dương
    return math.degrees(math.atan2(abs(dx), max(dy, 1e-6)))


def extract_features(result, img_w: int, img_h: int) -> dict | None:
    """
    Nhận kết quả từ MediaPipe Holistic (run_holistic),
    trả về dict features hoặc None nếu không có face.
    
    Hỗ trợ cả MediaPipe result và TransformedResult wrapper.
    """
    if result is None:
        return None
    
    # Handle TransformedResult wrapper
    if hasattr(result, 'face_landmarks'):
        face_landmarks_list = result.face_landmarks
    else:
        face_landmarks_list = result.face_landmarks if hasattr(result, 'face_landmarks') else None
    
    if not face_landmarks_list or len(face_landmarks_list) == 0:
        return None
    
    # Lấy face landmarks đầu tiên (primary person)
    face_lm = face_landmarks_list[0]
    if not face_lm:
        return None

    pts_px = [(lm.x * img_w, lm.y * img_h) for lm in face_lm]

    ear_left  = compute_ear(pts_px, LEFT_EYE_EAR_IDX)
    ear_right = compute_ear(pts_px, RIGHT_EYE_EAR_IDX)
    ear_avg   = (ear_left + ear_right) / 2.0
    mar       = compute_mar(pts_px)
    _, _, mouth_aspect = compute_mouth_aspect(pts_px)

    hp = compute_head_pose(pts_px, img_w, img_h)
    pitch, yaw, roll = hp if hp is not None else (float("nan"),) * 3

    neck_tilt = float("nan")
    pose_lm = None
    if hasattr(result, 'pose_landmarks'):
        raw_pose = result.pose_landmarks
        if raw_pose and len(raw_pose) > 0:
            # TransformedResult stores pose as list of transformed landmarks
            pose_lm = raw_pose[0]
    
    if pose_lm:
        pose_pts = [(lm.x * img_w, lm.y * img_h) for lm in pose_lm]
        nt = compute_neck_tilt(pose_pts)
        if nt is not None:
            neck_tilt = nt

    return {
        "ear_left":     ear_left,
        "ear_right":    ear_right,
        "ear_avg":      ear_avg,
        "mar":          mar,
        "mouth_aspect": mouth_aspect,
        "pitch":        pitch,
        "yaw":          yaw,
        "roll":         roll,
        "neck_tilt":    neck_tilt,
        "has_pose":     bool(pose_lm),
    }
