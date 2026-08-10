"""Tests for cross-version model loading helpers."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.model_loader import (
    build_lstm_model,
    build_mlp_model,
    load_drowsiness_bundle,
    model_search_roots,
    resolve_artifact,
)


def test_model_search_roots_order():
    base = Path("/app")
    roots = model_search_roots(base)
    assert roots[0].name == "compatible"
    assert roots[1].name == "models"


def test_resolve_artifact_prefers_compatible():
    base = Path(__file__).resolve().parent.parent
    roots = model_search_roots(base)
    mlp_w = resolve_artifact("mlp_drowsiness_landmark.weights.h5", roots)
    if mlp_w is None:
        pytest.skip("converted weights not present")
    assert "compatible" in str(mlp_w).replace("\\", "/")


@pytest.mark.skipif(
    not (Path(__file__).resolve().parent.parent / "models" / "compatible"
         / "mlp_drowsiness_landmark.weights.h5").is_file(),
    reason="converted bundle missing",
)
def test_load_drowsiness_bundle_weights_mode():
    base = Path(__file__).resolve().parent.parent
    bundle = load_drowsiness_bundle(base)
    assert bundle["load_mode"] == "weights"

    x = np.zeros((1, 9), dtype=np.float32)
    seq = np.zeros((30, 6), dtype=np.float32)
    p_mlp = float(bundle["mlp_model"].predict(
        bundle["mlp_scaler"].transform(x), verbose=0
    )[0, 0])
    seq_scaled = bundle["lstm_scaler"].transform(seq).reshape(1, 30, 6)
    p_lstm = float(bundle["lstm_model"].predict(seq_scaled, verbose=0)[0, 0])

    assert 0.0 <= p_mlp <= 1.0
    assert 0.0 <= p_lstm <= 1.0


def test_build_architectures_output_shape():
    mlp = build_mlp_model()
    lstm = build_lstm_model()
    assert mlp.output_shape == (None, 1)
    assert lstm.output_shape == (None, 1)
