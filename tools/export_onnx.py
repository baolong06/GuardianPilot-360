"""
ONNX export scaffold (P1) — export Keras MLP to ONNX when deps available.

Usage:
  python tools/export_onnx.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT_PATH = ROOT / "models" / "mlp_drowsiness_landmark.onnx"


def _find_mlp() -> Path | None:
    """
    Tìm model MLP qua đúng thứ tự ưu tiên mà app.py dùng.

    Trước đây file này trỏ cứng `models/mlp_drowsiness_landmark.keras`, nhưng
    thư mục models/ gốc chỉ chứa models/compatible/ → script LUÔN in
    "Model not found" và không ai để ý là nó chưa từng chạy được.
    """
    from src.model_loader import MLP_KERAS_NAME, model_search_roots, resolve_artifact
    return resolve_artifact(MLP_KERAS_NAME, model_search_roots(ROOT))


def main():
    mlp_path = _find_mlp()
    if mlp_path is None:
        print("Không tìm thấy mlp_drowsiness_landmark.keras trong "
              "models/compatible, models/ hay results/.")
        print("Chạy trước: python tools/convert_models.py --in-place")
        return 1
    print(f"Model: {mlp_path}")
    try:
        import keras
        import tf2onnx
        import tensorflow as tf
    except ImportError as exc:
        print(f"Missing export deps ({exc}). pip install tf2onnx")
        print("Scaffold OK — export skipped.")
        return 0

    model = keras.saving.load_model(str(mlp_path))
    spec = (tf.TensorSpec((None, 9), tf.float32, name="input"),)
    model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)
    OUT_PATH.write_bytes(model_proto.SerializeToString())
    print(f"Wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
