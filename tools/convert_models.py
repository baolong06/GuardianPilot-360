"""
Convert legacy Keras artifacts to Docker-compatible weights bundle.

Why: models saved with newer Keras (3.15+) may fail to deserialize in Docker
(Keras 3.5 + TF 2.17) due to unsupported config fields / functional ops.

This script rebuilds the known MLP/LSTM architecture, transfers trained
weights, and writes portable `.weights.h5` files under models/compatible/.

Usage:
  python tools/convert_models.py
  python tools/convert_models.py --in-place
  python tools/convert_models.py --source models --output models/compatible
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.model_loader import (  # noqa: E402
    HOLISTIC_TASK_NAME,
    LSTM_KERAS_NAME,
    LSTM_SCALER_NAME,
    LSTM_WEIGHTS_NAME,
    MLP_KERAS_NAME,
    MLP_SCALER_NAME,
    MLP_WEIGHTS_NAME,
    build_lstm_model,
    build_mlp_model,
    load_keras_model,
    load_drowsiness_bundle,
)


def _load_legacy_keras(source: Path):
  import keras

  mlp_path = source / MLP_KERAS_NAME
  lstm_path = source / LSTM_KERAS_NAME
  if not mlp_path.is_file() or not lstm_path.is_file():
    raise FileNotFoundError(
      f"Expected {MLP_KERAS_NAME} and {LSTM_KERAS_NAME} in {source}"
    )

  try:
    mlp_old = keras.saving.load_model(str(mlp_path), compile=False)
    lstm_old = keras.saving.load_model(str(lstm_path), compile=False)
    return mlp_old, lstm_old
  except Exception:
    # Fallback loader handles newer-config artifacts.
    mlp_old = load_keras_model(mlp_path)
    lstm_old = load_keras_model(lstm_path)
    return mlp_old, lstm_old


def convert(source: Path, output: Path, in_place: bool = False) -> Path:
  output.mkdir(parents=True, exist_ok=True)

  mlp_old, lstm_old = _load_legacy_keras(source)

  mlp_new = build_mlp_model()
  lstm_new = build_lstm_model()
  mlp_new.set_weights(mlp_old.get_weights())
  lstm_new.set_weights(lstm_old.get_weights())

  mlp_weights = output / MLP_WEIGHTS_NAME
  lstm_weights = output / LSTM_WEIGHTS_NAME
  mlp_new.save_weights(str(mlp_weights))
  lstm_new.save_weights(str(lstm_weights))

  for name in (MLP_SCALER_NAME, LSTM_SCALER_NAME):
    src = source / name
    if not src.is_file():
      alt = ROOT / "results" / name
      if alt.is_file():
        src = alt
      else:
        raise FileNotFoundError(f"Missing scaler: {name}")
    shutil.copy2(src, output / name)

  holistic_src = source / HOLISTIC_TASK_NAME
  if not holistic_src.is_file():
    alt = ROOT / "results" / HOLISTIC_TASK_NAME
    if alt.is_file():
      holistic_src = alt
  if holistic_src.is_file():
    shutil.copy2(holistic_src, output / HOLISTIC_TASK_NAME)

  if in_place:
    target = ROOT / "models"
    target.mkdir(exist_ok=True)
    for fname in (
      MLP_WEIGHTS_NAME,
      LSTM_WEIGHTS_NAME,
      MLP_SCALER_NAME,
      LSTM_SCALER_NAME,
      HOLISTIC_TASK_NAME,
    ):
      src = output / fname
      if src.is_file():
        shutil.copy2(src, target / fname)

  return output


def verify(bundle_dir: Path) -> None:
  bundle = load_drowsiness_bundle(ROOT)
  mlp = bundle["mlp_model"]
  lstm = bundle["lstm_model"]
  mlp_scaler = bundle["mlp_scaler"]
  lstm_scaler = bundle["lstm_scaler"]

  x = np.zeros((1, 9), dtype=np.float32)
  seq = np.zeros((30, 6), dtype=np.float32)
  p_mlp = float(mlp.predict(mlp_scaler.transform(x), verbose=0)[0, 0])
  seq_scaled = lstm_scaler.transform(seq).reshape(1, 30, 6)
  p_lstm = float(lstm.predict(seq_scaled, verbose=0)[0, 0])

  print(f"verify OK ({bundle['load_mode']})")
  print(f"  holistic_task: {bundle['holistic_task']}")
  print(f"  mlp output:    {p_mlp:.6f}")
  print(f"  lstm output:   {p_lstm:.6f}")
  print(f"  bundle roots:  {bundle_dir}")


def main() -> int:
  parser = argparse.ArgumentParser(description="Convert drowsiness models for Docker")
  parser.add_argument("--source", type=Path, default=ROOT / "models")
  parser.add_argument("--output", type=Path, default=ROOT / "models" / "compatible")
  parser.add_argument(
    "--in-place",
    action="store_true",
    help="Also copy converted weights/scalers/task into models/",
  )
  parser.add_argument("--verify-only", action="store_true")
  args = parser.parse_args()

  if args.verify_only:
    verify(args.output)
    return 0

  out = convert(args.source, args.output, in_place=args.in_place)
  print(f"Converted bundle -> {out}")
  print("Files:")
  for p in sorted(out.glob("*")):
    print(f"  - {p.name} ({p.stat().st_size:,} bytes)")

  verify(out)
  print("\nNext:")
  print("  docker compose build && docker compose up -d")
  print("  curl -X POST http://localhost:5000/api/init")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
