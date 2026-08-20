"""
Tìm các cell trong notebook nguồn chứa hằng số fusion (HYSTERESIS, EMA_ALPHA…).

Dùng để đối chiếu giá trị đang hard-code trong src/fusion.py với notebook đã
train ra model.

LƯU Ý: notebook nguồn KHÔNG nằm trong repo (`.gitignore` loại `results/*.ipynb`)
và hiện không có trên đĩa. Đây chính là vấn đề C3 trong báo cáo audit — model
được train ngoài repo nên không tái tạo được. Script này chỉ hữu ích khi bạn
có sẵn file notebook đó.

Usage:
  python search_nb.py --nb results/notebook6672d603fa.ipynb
  python search_nb.py --nb path/to/nb.ipynb --pattern LSTM --pattern scaler
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DEFAULT_PATTERNS = [
    'HYSTERESIS', 'EMA_ALPHA', 'MIN_ON', 'MIN_OFF', 'neck_alarm',
    'EMA_PROB_ON', 'neck_baseline', 'neck_recovered', 'def fusion',
    'class Fusion', 'def __init__',
]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Grep các cell notebook chứa hằng số fusion"
    )
    parser.add_argument(
        "--nb", type=Path,
        default=Path("results/notebook6672d603fa.ipynb"),
        help="Đường dẫn file .ipynb",
    )
    parser.add_argument(
        "--pattern", action="append", default=None,
        help="Từ khoá cần tìm (lặp lại được). Mặc định: bộ hằng số fusion.",
    )
    parser.add_argument("--max-chars", type=int, default=2500)
    args = parser.parse_args()

    if not args.nb.is_file():
        print(f"Không tìm thấy notebook: {args.nb}", file=sys.stderr)
        print(
            "Notebook nguồn không được commit vào repo (xem .gitignore: "
            "results/*.ipynb). Chỉ định đường dẫn bằng --nb.",
            file=sys.stderr,
        )
        return 1

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

    patterns = args.pattern or DEFAULT_PATTERNS
    nb = json.loads(args.nb.read_text(encoding="utf-8"))

    hits = 0
    for i, cell in enumerate(nb.get("cells", [])):
        src = "".join(cell.get("source", []))
        if any(k in src for k in patterns):
            hits += 1
            print(f"\n=== CELL {i} ({cell.get('cell_type')}) ===")
            print(src[:args.max_chars])
            print("---")

    print(f"\n{hits} cell khớp / {len(nb.get('cells', []))} cell.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
