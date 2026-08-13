"""
Runtime profile — tách cấu hình dev (máy tính) vs edge (ô tô / Jetson).

PRD mục tiêu edge:
  - 15–30 FPS pipeline, latency < 200ms
  - RAM < 4GB, CPU < 80%
  - Inference ONNX/TensorRT INT8, rule engine nhẹ

Biến môi trường:
  EDGE_PROFILE=dev|edge   (mặc định: dev)
"""
from __future__ import annotations

import os
from typing import Any

_PROFILES: dict[str, dict[str, Any]] = {
    # Máy dev / demo web — ưu tiên độ mượt UI
    "dev": {
        "label": "development",
        "inference_interval_ms": 100,   # ~10 inference/s
        "inference_width": 480,        # tăng từ 320 → 480 (detect face xa tốt hơn)
        "inference_height": 360,       # tăng từ 240 → 360
        "display_fps_cap": 30,
        "enable_lstm": True,
        "mediapipe_width": 480,        # tăng từ 320 → 480 (face xa ~40px → ~60px → vượt MIN_FACE_SIZE_PX)
        "mediapipe_height": 360,       # tăng từ 240 → 360
        "omp_num_threads": None,
        "use_pitch_nod": True,
        "description": "Web demo — balanced CPU use, wider detection range",
    },
    # Edge ô tô / Jetson Nano class — ưu tiên ổn định & tiết kiệm tài nguyên
    "edge": {
        "label": "automotive_edge",
        "inference_interval_ms": 200,   # 5 FPS đủ cho DMS (PRD: temporal rules)
        "inference_width": 256,
        "inference_height": 192,
        "display_fps_cap": 15,
        "enable_lstm": False,           # bỏ LSTM window 30 frame — tiết kiệm ~40% infer
        "mediapipe_width": 256,
        "mediapipe_height": 192,
        "omp_num_threads": 2,
        "use_pitch_nod": True,
        "description": "Automotive edge — low CPU, rule-first DMS",
    },
}


def get_profile_name() -> str:
    name = os.getenv("EDGE_PROFILE", "dev").strip().lower()
    return name if name in _PROFILES else "dev"


def get_runtime_profile() -> dict[str, Any]:
    """Trả về profile hiện tại (copy) kèm tên."""
    name = get_profile_name()
    cfg = dict(_PROFILES[name])
    cfg["profile"] = name
    return cfg


def apply_process_limits() -> None:
    """Áp thread limit lên TF/OpenMP — gọi một lần khi khởi động app."""
    cfg = get_runtime_profile()
    n = cfg.get("omp_num_threads")
    if n is not None:
        os.environ.setdefault("OMP_NUM_THREADS", str(n))
        os.environ.setdefault("TF_NUM_INTRAOP_THREADS", str(n))
        os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
