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

MLP_PATH = ROOT / "models" / "mlp_drowsiness_landmark.keras"
OUT_PATH = ROOT / "models" / "mlp_drowsiness_landmark.onnx"


def main():
    if not MLP_PATH.is_file():
        print(f"Model not found: {MLP_PATH}")
        print("Scaffold only — place Keras model then re-run with tf2onnx installed.")
        return 1
    try:
        import keras
        import tf2onnx
        import tensorflow as tf
    except ImportError as exc:
        print(f"Missing export deps ({exc}). pip install tf2onnx")
        print("Scaffold OK — export skipped.")
        return 0

    model = keras.saving.load_model(str(MLP_PATH))
    spec = (tf.TensorSpec((None, 9), tf.float32, name="input"),)
    model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)
    OUT_PATH.write_bytes(model_proto.SerializeToString())
    print(f"Wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
