"""
Load MLP/LSTM drowsiness models with cross-version Keras compatibility.

Preferred layout (after `python tools/convert_models.py`):
  models/compatible/mlp_drowsiness_landmark.weights.h5
  models/compatible/lstm_drowsiness_landmark.weights.h5
  models/compatible/landmark_scaler.pkl
  models/compatible/lstm_seq_scaler.pkl

Fallback: legacy `.keras` artifacts in models/ or results/ (with config sanitization).
"""
from __future__ import annotations

import io
import json
import tempfile
import zipfile
from pathlib import Path

import joblib

MLP_KERAS_NAME = "mlp_drowsiness_landmark.keras"
LSTM_KERAS_NAME = "lstm_drowsiness_landmark.keras"
MLP_WEIGHTS_NAME = "mlp_drowsiness_landmark.weights.h5"
LSTM_WEIGHTS_NAME = "lstm_drowsiness_landmark.weights.h5"
MLP_SCALER_NAME = "landmark_scaler.pkl"
LSTM_SCALER_NAME = "lstm_seq_scaler.pkl"
HOLISTIC_TASK_NAME = "holistic_landmarker.task"

_CONFIG_STRIP_KEYS = {
    "quantization_config",
    "input_axes",
    "output_axes",
    "shared_object_id",
}


def build_mlp_model():
    import keras
    from keras import layers

    inp = keras.Input(shape=(9,), name="input_layer")
    x = layers.Dense(64, activation="relu", name="dense")(inp)
    x = layers.Dropout(0.3, name="dropout")(x)
    x = layers.Dense(32, activation="relu", name="dense_1")(x)
    x = layers.Dropout(0.2, name="dropout_1")(x)
    out = layers.Dense(1, activation="sigmoid", name="dense_2")(x)
    return keras.Model(inp, out, name="MLP_Drowsiness_Landmark")


def build_lstm_model():
    import keras
    from keras import layers

    inp = keras.Input(shape=(30, 6), name="input_layer_1")
    x = layers.Masking(mask_value=0.0, name="masking")(inp)
    x = layers.LSTM(32, return_sequences=True, name="lstm")(x)
    x = layers.Dropout(0.3, name="dropout_2")(x)
    x = layers.LSTM(16, name="lstm_1")(x)
    x = layers.Dropout(0.2, name="dropout_3")(x)
    out = layers.Dense(1, activation="sigmoid", name="dense_3")(x)
    return keras.Model(inp, out, name="LSTM_Drowsiness_Landmark")


def _sanitize_config_obj(obj):
    if isinstance(obj, dict):
        for key in list(obj.keys()):
            if key in _CONFIG_STRIP_KEYS:
                obj.pop(key, None)
        if obj.get("class_name") == "DTypePolicy" and "config" in obj:
            return obj["config"].get("name", "float32")
        for key, value in list(obj.items()):
            obj[key] = _sanitize_config_obj(value)
        return obj
    if isinstance(obj, list):
        return [_sanitize_config_obj(v) for v in obj]
    return obj


def sanitize_keras_archive_bytes(data: bytes) -> bytes:
    with zipfile.ZipFile(io.BytesIO(data), "r") as zin:
        cfg = json.loads(zin.read("config.json"))
        cfg = _sanitize_config_obj(cfg)
        out = io.BytesIO()
        with zipfile.ZipFile(out, "w") as zout:
            for item in zin.infolist():
                payload = zin.read(item.filename)
                if item.filename == "config.json":
                    payload = json.dumps(cfg).encode()
                zout.writestr(item, payload)
        return out.getvalue()


def load_keras_model(path: Path):
    """Load .keras with direct + sanitized fallback."""
    loaders = []
    try:
        import keras
        loaders.append(lambda p: keras.saving.load_model(str(p), compile=False))
    except Exception:
        pass
    try:
        from tensorflow import keras as tf_keras
        loaders.append(lambda p: tf_keras.models.load_model(str(p), compile=False))
    except Exception:
        pass
    if not loaders:
        raise RuntimeError("No Keras loader available")

    raw = path.read_bytes()
    candidates = [raw, sanitize_keras_archive_bytes(raw)]
    last_exc = None
    for blob in candidates:
        with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as tmp:
            tmp.write(blob)
            tmp_path = Path(tmp.name)
        try:
            for fn in loaders:
                try:
                    return fn(tmp_path)
                except Exception as exc:
                    last_exc = exc
        finally:
            tmp_path.unlink(missing_ok=True)
    if last_exc:
        raise last_exc
    raise RuntimeError(f"Failed to load model: {path}")


def resolve_artifact(filename: str, search_roots: list[Path]) -> Path | None:
    for root in search_roots:
        if not root.is_dir():
            continue
        cand = root / filename
        if cand.is_file():
            return cand
    return None


def model_search_roots(base_dir: Path) -> list[Path]:
    return [
        base_dir / "models" / "compatible",
        base_dir / "models",
        base_dir / "results",
    ]


def load_drowsiness_bundle(base_dir: Path) -> dict:
    """
    Return dict with keys: mlp_model, lstm_model, mlp_scaler, lstm_scaler,
    holistic_task, load_mode ('weights' | 'keras').
    """
    roots = model_search_roots(base_dir)

    mlp_weights = resolve_artifact(MLP_WEIGHTS_NAME, roots)
    lstm_weights = resolve_artifact(LSTM_WEIGHTS_NAME, roots)
    mlp_keras = resolve_artifact(MLP_KERAS_NAME, roots)
    lstm_keras = resolve_artifact(LSTM_KERAS_NAME, roots)
    mlp_scaler_path = resolve_artifact(MLP_SCALER_NAME, roots)
    lstm_scaler_path = resolve_artifact(LSTM_SCALER_NAME, roots)
    holistic_task = resolve_artifact(HOLISTIC_TASK_NAME, roots)

    if not mlp_scaler_path or not lstm_scaler_path:
        raise FileNotFoundError(
            f"Missing scaler files ({MLP_SCALER_NAME}, {LSTM_SCALER_NAME})."
        )
    if not holistic_task:
        raise FileNotFoundError(f"Missing MediaPipe task file: {HOLISTIC_TASK_NAME}")

    if mlp_weights and lstm_weights:
        mlp_model = build_mlp_model()
        lstm_model = build_lstm_model()
        mlp_model.load_weights(str(mlp_weights))
        lstm_model.load_weights(str(lstm_weights))
        load_mode = "weights"
    elif mlp_keras and lstm_keras:
        mlp_model = load_keras_model(mlp_keras)
        lstm_model = load_keras_model(lstm_keras)
        load_mode = "keras"
    else:
        raise FileNotFoundError(
            "Missing model artifacts. Run: python tools/convert_models.py "
            "or place .weights.h5 / .keras files under models/compatible/."
        )

    return {
        "mlp_model": mlp_model,
        "lstm_model": lstm_model,
        "mlp_scaler": joblib.load(str(mlp_scaler_path)),
        "lstm_scaler": joblib.load(str(lstm_scaler_path)),
        "holistic_task": holistic_task,
        "load_mode": load_mode,
    }
