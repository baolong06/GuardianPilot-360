"""
Evaluation scaffold — chạy trên tập có nhãn (data/ hoặc results/).

MVP: tính các chỉ số cơ bản từ CSV nhãn + prediction JSON.
Chưa thay thế benchmark đầy đủ mục 9.2; dùng để theo dõi khi có dataset.

Usage:
  python tools/evaluate.py --labels data/labels.csv --preds data/preds.csv --out reports/eval.md
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def confusion(y_true: list[int], y_pred: list[int]) -> dict:
    tp = fp = tn = fn = 0
    for t, p in zip(y_true, y_pred):
        if t == 1 and p == 1:
            tp += 1
        elif t == 0 and p == 1:
            fp += 1
        elif t == 0 and p == 0:
            tn += 1
        else:
            fn += 1
    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": _safe_div(tp, tp + fp),
        "recall": _safe_div(tp, tp + fn),
        "accuracy": _safe_div(tp + tn, tp + tn + fp + fn),
        "fpr": _safe_div(fp, fp + tn),
    }


def load_binary_csv(path: Path, label_col: str = "label", pred_col: str = "pred") -> tuple[list[int], list[int]]:
    y_true, y_pred = [], []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            y_true.append(int(float(row[label_col])))
            y_pred.append(int(float(row[pred_col])))
    return y_true, y_pred


def render_report(metrics: dict, title: str) -> str:
    lines = [
        f"# {title}",
        "",
        "| Metric | Value | PRD target |",
        "|--------|------:|------------|",
        f"| Accuracy | {metrics['accuracy']:.3f} | — |",
        f"| Precision | {metrics['precision']:.3f} | Face >95% (related) |",
        f"| Recall | {metrics['recall']:.3f} | Drowsiness >90% |",
        f"| FPR | {metrics['fpr']:.3f} | <5% |",
        "",
        f"Confusion: TP={metrics['tp']} FP={metrics['fp']} TN={metrics['tn']} FN={metrics['fn']}",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="GuardianPilot eval scaffold")
    parser.add_argument("--csv", type=Path, help="CSV with label,pred columns")
    parser.add_argument("--out", type=Path, default=Path("reports/eval.md"))
    parser.add_argument("--title", default="GuardianPilot Evaluation")
    args = parser.parse_args()

    if args.csv is None or not args.csv.is_file():
        # Demo / dry-run khi chưa có dataset
        report = (
            f"# {args.title}\n\n"
            "Chưa có file CSV nhãn. Tạo `labels,pred` CSV rồi chạy lại:\n\n"
            "```bash\n"
            "python tools/evaluate.py --csv data/preds.csv --out reports/eval.md\n"
            "```\n\n"
            "Targets (PRD 9.2): Face Detection >95%, Eye State >90%, "
            "Drowsiness Recall >90%, FP <5%.\n"
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report, encoding="utf-8")
        print(f"Wrote scaffold report -> {args.out}")
        return

    y_true, y_pred = load_binary_csv(args.csv)
    metrics = confusion(y_true, y_pred)
    report = render_report(metrics, args.title)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")

    # CSV summary
    csv_out = args.out.with_suffix(".csv")
    with csv_out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        w.writeheader()
        w.writerow(metrics)
    print(f"Wrote {args.out} and {csv_out}")
    print(report)


if __name__ == "__main__":
    main()
