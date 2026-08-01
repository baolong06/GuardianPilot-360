"""
Evaluation metrics cho pedestrian crossing intention prediction.

Dùng cho:
    - training/intent/train.py (validation loop)
    - evaluation/evaluator.py (offline evaluation)

IMPORTANT phân biệt:
    - crossing_action  = nhãn hành vi HIỆN TẠI (label từ cross attribute)
    - crossing_intention = nhãn DỰ ĐOÁN tương lai (label này)
"""
import numpy as np
from typing import List, Optional


def compute_accuracy(
    preds: List[int],
    truths: List[int],
) -> float:
    """Accuracy đơn giản. Chú ý: không dùng riêng lẻ khi có class imbalance."""
    if len(preds) == 0:
        return 0.0
    correct = sum(p == t for p, t in zip(preds, truths))
    return correct / len(truths)


def compute_f1_binary(
    preds: List[int],
    truths: List[int],
    pos_label: int = 1,
) -> float:
    """Binary F1 cho class dương (crossing = 1)."""
    tp = sum((p == pos_label and t == pos_label) for p, t in zip(preds, truths))
    fp = sum((p == pos_label and t != pos_label) for p, t in zip(preds, truths))
    fn = sum((p != pos_label and t == pos_label) for p, t in zip(preds, truths))
    if tp + fp == 0 or tp + fn == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_auc(
    probs: List[float],
    truths: List[int],
) -> float:
    """ROC-AUC score cho binary classification."""
    try:
        from sklearn.metrics import roc_auc_score
        if len(set(truths)) < 2:
            return float('nan')
        return float(roc_auc_score(truths, probs))
    except Exception:
        return float('nan')


def compute_time_to_event_accuracy(
    preds: List[int],
    truths: List[int],
    time_to_events: List[int],
    bins: Optional[List[int]] = None,
) -> dict:
    """
    Đánh giá accuracy theo time_to_event (số frame đến crossing_point).

    Bins mặc định: [0-15], [16-30], [31-60], [61+] frames
    (ứng với khoảng ~0-0.5s, ~0.5-1s, ~1-2s, ~2s+ @ 30fps)

    Returns:
        dict: {bin_label: accuracy}
    """
    if bins is None:
        bins = [0, 16, 31, 61, 999]
    bin_labels = [
        f"[{bins[i]}-{bins[i+1]-1}]" for i in range(len(bins) - 1)
    ]

    results = {}
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        indices = [
            j for j, tte in enumerate(time_to_events)
            if lo <= tte < hi
        ]
        if not indices:
            results[bin_labels[i]] = float('nan')
            continue
        bin_preds = [preds[j] for j in indices]
        bin_truths = [truths[j] for j in indices]
        acc = compute_accuracy(bin_preds, bin_truths)
        results[bin_labels[i]] = acc

    return results


def compute_intention_metrics(
    preds: List[int],
    truths: List[int],
    probs: Optional[List[float]] = None,
    time_to_events: Optional[List[int]] = None,
) -> dict:
    """
    Tổng hợp tất cả metrics cho intention evaluation.

    Args:
        preds:         predicted labels (0 or 1)
        truths:        ground truth labels (0 or 1)
        probs:         predicted probability for class 1 (will cross)
        time_to_events: frame count to crossing_point for each sample

    Returns:
        dict với keys: accuracy, f1, auc, time_to_event_accuracy
    """
    metrics = {
        "accuracy": compute_accuracy(preds, truths),
        "f1_binary": compute_f1_binary(preds, truths, pos_label=1),
        "n_samples": len(truths),
        "n_positive": sum(truths),
        "n_negative": len(truths) - sum(truths),
    }

    if probs is not None:
        metrics["auc"] = compute_auc(probs, truths)

    if time_to_events is not None:
        metrics["time_to_event_accuracy"] = compute_time_to_event_accuracy(
            preds, truths, time_to_events
        )

    return metrics
