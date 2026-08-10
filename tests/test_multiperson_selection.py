"""
Unit tests cho multi-person primary selection (src/pipeline.py).

Test các hàm scoring/bbox thuần — không cần load MediaPipe model.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import (
    _score_person,
    _estimate_face_bbox,
    _expand_crop,
    MIN_FACE_SIZE_PX,
)


class _LM:
    def __init__(self, x, y, z=0.0):
        self.x, self.y, self.z = x, y, z


def test_tiny_face_rejected():
    """Face quá nhỏ → score = -1."""
    # bbox 40x40 trên frame 640x480 — dưới MIN_FACE_SIZE_PX=60
    bbox = (100, 100, 140, 140)
    score, info = _score_person(bbox, pose_landmarks=None, img_w=640, img_h=480)
    assert score < 0
    assert info.get("reason") == "too_small"
    print(f"PASS test_tiny_face_rejected: score={score}")


def test_driver_position_beats_background():
    """
    Face lớn hơn ở lower-center (vị trí lái) phải thắng face nhỏ ở góc trên.
    """
    img_w, img_h = 640, 480
    # Driver: lớn, gần center, lower
    driver_bbox = (220, 200, 420, 400)  # ~200x200, center~(320,300)
    # Background: nhỏ hơn, góc trên
    bg_bbox = (20, 20, 100, 100)  # 80x80, center~(60,60)

    s_driver, info_d = _score_person(driver_bbox, pose_landmarks=[1], img_w=img_w, img_h=img_h)
    s_bg, info_b = _score_person(bg_bbox, pose_landmarks=None, img_w=img_w, img_h=img_h)

    assert s_driver > 0 and s_bg > 0
    assert s_driver > s_bg, f"driver={s_driver} should beat bg={s_bg}"
    assert info_d["pose_bonus"] == 40
    assert info_b["pose_bonus"] == 0
    print(f"PASS test_driver_position_beats_background: driver={s_driver:.1f} bg={s_bg:.1f}")


def test_pose_bonus_increases_score():
    bbox = (200, 180, 400, 380)
    s_no, _ = _score_person(bbox, None, 640, 480)
    s_yes, info = _score_person(bbox, [1], 640, 480)
    assert s_yes - s_no == 40
    assert info["pose_bonus"] == 40
    print(f"PASS test_pose_bonus_increases_score: delta={s_yes - s_no}")


def test_estimate_face_bbox():
    lms = [_LM(0.2, 0.3), _LM(0.5, 0.3), _LM(0.5, 0.6), _LM(0.2, 0.6)]
    bbox = _estimate_face_bbox(lms, 100, 100)
    assert bbox == (20, 30, 50, 60)
    assert _estimate_face_bbox([], 100, 100) is None
    print(f"PASS test_estimate_face_bbox: {bbox}")


def test_expand_crop_clamped():
    x0, y0, x1, y1 = _expand_crop(0, 0, 50, 50, img_w=100, img_h=100, margin=0.5)
    assert x0 == 0 and y0 == 0
    assert x1 <= 100 and y1 <= 100
    print(f"PASS test_expand_crop_clamped: {(x0, y0, x1, y1)}")


def test_primary_selection_sort():
    """Mô phỏng chọn primary: sort score desc → idx cao nhất."""
    img_w, img_h = 640, 480
    faces = [
        ((20, 20, 90, 90), None),           # nhỏ, góc
        ((220, 200, 420, 400), [1]),        # driver
        ((500, 50, 580, 130), None),        # góc phải trên
    ]
    scored = []
    for i, (bbox, pose) in enumerate(faces):
        s, info = _score_person(bbox, pose, img_w, img_h)
        info["idx"] = i
        scored.append((s, info))
    scored.sort(key=lambda x: x[0], reverse=True)
    assert scored[0][1]["idx"] == 1
    print(f"PASS test_primary_selection_sort: primary_idx={scored[0][1]['idx']}")


if __name__ == "__main__":
    test_tiny_face_rejected()
    test_driver_position_beats_background()
    test_pose_bonus_increases_score()
    test_estimate_face_bbox()
    test_expand_crop_clamped()
    test_primary_selection_sort()
    print("\nAll multiperson tests passed.")
