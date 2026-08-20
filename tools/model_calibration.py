"""
Đo hành vi của MLP/LSTM theo EAR — biến "model bias" thành số đo theo dõi được.

BỐI CẢNH (C4): reports/live_diagnostic.md ghi nhận MLP trả p_drowsy ≈ 0.585 khi
mắt đang MỞ bình thường (EAR = 0.30), tức vượt cả ngưỡng cảnh báo. Đó là quan
sát một lần, chép tay vào file markdown. Script này chạy lại phép đo đó một
cách có hệ thống để:
  - biết ngay khi thay model / thay scaler thì bias đổi ra sao;
  - có số liệu cụ thể khi quyết định bật FORCE_RULE_ONLY=true.

ĐÂY KHÔNG PHẢI EVALUATION THẬT. Nó quét feature tổng hợp, không có nhãn người
thật. Đánh giá đúng nghĩa cần test set có nhãn (vấn đề C3, ngoài phạm vi).

Usage:
  python tools/model_calibration.py
  python tools/model_calibration.py --out reports/model_calibration.md
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# EAR quét từ nhắm hẳn tới mở rõ. Ngưỡng rule hiện tại là 0.16.
EAR_SWEEP = [0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.25, 0.30, 0.35]

# Feature "trung tính" cho các trường còn lại (lấy từ mẫu thật trong
# results/video_fusion_log.csv, khung hình tài xế tỉnh táo nhìn thẳng).
NEUTRAL = {
    "mar": 0.02,
    "mouth_aspect": 0.05,
    "pitch": -24.5,
    "yaw": -7.7,
    "roll": -177.9,
    "neck_tilt": 1.9,
}


def build_rows(bundle, with_pose: bool):
    from src.fusion import LSTM_FEAT_COLS, MLP_FEAT_COLS, _ffill_bfill, WINDOW_SIZE

    neck = NEUTRAL["neck_tilt"] if with_pose else float("nan")
    rows = []
    for ear in EAR_SWEEP:
        feat = {
            "ear_left": ear, "ear_right": ear, "ear_avg": ear,
            "mar": NEUTRAL["mar"], "mouth_aspect": NEUTRAL["mouth_aspect"],
            "pitch": NEUTRAL["pitch"], "yaw": NEUTRAL["yaw"],
            "roll": NEUTRAL["roll"], "neck_tilt": neck,
        }
        has_neck = 0 if neck != neck else 1

        mlp_row = np.array(
            [feat.get(c, 0.0) if c != "has_neck_tilt" else has_neck
             for c in MLP_FEAT_COLS], dtype=np.float32)
        mlp_row = np.nan_to_num(mlp_row, nan=0.0)
        x = np.nan_to_num(
            np.asarray(bundle["mlp_scaler"].transform(mlp_row.reshape(1, -1)),
                       dtype=np.float32), nan=0.0)
        p_mlp = 1.0 - float(bundle["mlp_model"].predict(x, verbose=0)[0, 0])

        seq = _ffill_bfill(np.asarray(
            [[feat[c] for c in LSTM_FEAT_COLS]] * WINDOW_SIZE, dtype=np.float32))
        seq = np.nan_to_num(seq, nan=0.0)
        seq_scaled = np.nan_to_num(
            np.asarray(bundle["lstm_scaler"].transform(seq), dtype=np.float32),
            nan=0.0).reshape(1, WINDOW_SIZE, len(LSTM_FEAT_COLS))
        p_lstm = 1.0 - float(bundle["lstm_model"].predict(seq_scaled, verbose=0)[0, 0])

        rows.append((ear, p_mlp, p_lstm))
    return rows


def render(rows_pose, rows_nopose, load_mode: str) -> str:
    from src.fusion import HYSTERESIS_ON, EYE_CLOSED_THRESH

    lines = [
        "# Model calibration — MLP / LSTM theo EAR",
        "",
        f"Sinh tự động bởi `tools/model_calibration.py` lúc "
        f"{datetime.now().strftime('%Y-%m-%d %H:%M')} (load_mode=`{load_mode}`).",
        "",
        "> **Đây KHÔNG phải evaluation.** Không có nhãn người thật ở đây — chỉ là",
        "> phép quét feature tổng hợp để theo dõi hành vi model. Đánh giá đúng",
        "> nghĩa cần test set có nhãn (vấn đề C3).",
        "",
        f"Ngưỡng tham chiếu: `EYE_CLOSED_THRESH = {EYE_CLOSED_THRESH}` "
        f"(mắt nhắm), `HYSTERESIS_ON = {HYSTERESIS_ON}` (bật cảnh báo).",
        "",
    ]
    for title, rows, note in [
        ("Có pose vai (neck_tilt hợp lệ)", rows_pose, ""),
        ("Mất pose vai (neck_tilt = NaN)", rows_nopose,
         "Đây là trường hợp gây bug C2 trước khi sửa."),
    ]:
        lines += [f"## {title}", ""]
        if note:
            lines += [f"_{note}_", ""]
        lines += [
            "| EAR | Trạng thái mắt | p_mlp_drowsy | p_lstm_drowsy | Vượt HYSTERESIS_ON? |",
            "|---|---|---|---|---|",
        ]
        for ear, p_mlp, p_lstm in rows:
            state = "nhắm" if ear < EYE_CLOSED_THRESH else "mở"
            flag = "⚠️ CÓ" if max(p_mlp, p_lstm) >= HYSTERESIS_ON else "không"
            lines.append(
                f"| {ear:.2f} | {state} | {p_mlp:.3f} | {p_lstm:.3f} | {flag} |"
            )
        lines.append("")

    lines += ["## Nhận xét tự động", ""]
    open_rows = [r for r in rows_pose if r[0] >= 0.25]
    FATIGUE_ON = 0.40

    if open_rows:
        worst_mlp = max(open_rows, key=lambda r: r[1])
        worst_lstm = max(open_rows, key=lambda r: r[2])
        lines += [
            f"- Mắt mở rõ (EAR ≥ 0.25): `p_mlp_drowsy` cao nhất **{worst_mlp[1]:.3f}** "
            f"(EAR={worst_mlp[0]:.2f}); `p_lstm_drowsy` cao nhất "
            f"**{worst_lstm[2]:.3f}** (EAR={worst_lstm[0]:.2f}).",
        ]
        for name, worst, idx in [("MLP", worst_mlp, 1), ("LSTM", worst_lstm, 2)]:
            if worst[idx] >= FATIGUE_ON:
                lines.append(
                    f"- ⚠️ **{name}** vượt ngưỡng FATIGUE ({FATIGUE_ON}) DÙ MẮT ĐANG MỞ."
                )
            else:
                lines.append(f"- {name} không có bias rõ rệt ở dải mắt mở.")

    # Độ nhạy: model có PHÂN BIỆT được mắt nhắm với mắt mở không?
    for name, idx in [("MLP", 1), ("LSTM", 2)]:
        vals = [r[idx] for r in rows_pose]
        span = max(vals) - min(vals)
        if span < 0.15:
            lines.append(
                f"- ⚠️ **{name} gần như KHÔNG phản ứng với EAR** (biên độ chỉ "
                f"{span:.3f} trên toàn dải 0.08→0.35). Model này không đóng góp "
                f"tín hiệu hữu ích; `src/fusion.py` đã có guard bỏ qua LSTM khi nó "
                f"lệch MLP > 0.15, nhưng nó vẫn kéo `drowsiness_score` lên nền cao."
            )
        else:
            lines.append(f"- {name} phản ứng theo EAR với biên độ {span:.3f}. OK.")

    lines += [
        "",
        "**Cần retrain (C3) mới xử lý được gốc.** Trong lúc chờ, chạy "
        "`FORCE_RULE_ONLY=true` để vận hành thuần rule engine.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Đo bias MLP/LSTM theo EAR")
    parser.add_argument("--out", type=Path,
                        default=ROOT / "reports" / "model_calibration.md")
    args = parser.parse_args()

    from src.model_loader import load_drowsiness_bundle
    bundle = load_drowsiness_bundle(ROOT)
    print(f"load_mode = {bundle['load_mode']}")

    rows_pose = build_rows(bundle, with_pose=True)
    rows_nopose = build_rows(bundle, with_pose=False)

    for ear, p_mlp, p_lstm in rows_pose:
        print(f"  EAR={ear:.2f}  p_mlp={p_mlp:.3f}  p_lstm={p_lstm:.3f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        render(rows_pose, rows_nopose, bundle["load_mode"]), encoding="utf-8"
    )
    print(f"\nĐã ghi {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
