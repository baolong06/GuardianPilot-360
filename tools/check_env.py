"""
Kiểm tra môi trường chạy (C1).

Vấn đề đã gặp: máy dev dùng Python 3.14 làm interpreter mặc định, trong khi
TensorFlow 2.17 và MediaPipe 0.10.14 chỉ có wheel tới Python 3.12 → `pip install
-r requirements.txt` fail, và `import app` fail theo.

Script này KHÔNG cài gì cả, chỉ đối chiếu và báo cáo.

Usage:
  python tools/check_env.py
  python tools/check_env.py --strict     # exit code 1 nếu có vấn đề
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

MIN_PY = (3, 9)
MAX_PY = (3, 12)   # trần do mediapipe 0.10.14 / tensorflow 2.17

# Package không bắt buộc — thiếu thì chỉ ghi chú, không tính là lỗi
OPTIONAL = {"pynvml", "tf2onnx", "playwright"}

# Tên import khác tên distribution
IMPORT_NAME = {
    "opencv-python": "cv2",
    "scikit-learn": "sklearn",
    "pillow": "PIL",
}

_REQ_LINE = re.compile(r"^\s*([A-Za-z0-9_.\-]+)\s*(?:([=<>!~]=|[<>])\s*([0-9A-Za-z.*+!-]+))?")


def parse_requirements(path: Path) -> list[tuple[str, str | None, str | None]]:
    out: list[tuple[str, str | None, str | None]] = []
    if not path.is_file():
        return out
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        m = _REQ_LINE.match(line)
        if m:
            out.append((m.group(1), m.group(2), m.group(3)))
    return out


def installed_version(dist: str) -> str | None:
    from importlib.metadata import PackageNotFoundError, version
    try:
        return version(dist)
    except PackageNotFoundError:
        return None


def check_python() -> list[str]:
    problems: list[str] = []
    v = sys.version_info
    print(f"Interpreter : {sys.executable}")
    print(f"Python      : {v.major}.{v.minor}.{v.micro}")
    if (v.major, v.minor) < MIN_PY:
        problems.append(
            f"Python {v.major}.{v.minor} quá cũ (cần >= {MIN_PY[0]}.{MIN_PY[1]})."
        )
    elif (v.major, v.minor) > MAX_PY:
        problems.append(
            f"Python {v.major}.{v.minor} quá mới — mediapipe 0.10.14 và "
            f"tensorflow 2.17 chỉ hỗ trợ tới {MAX_PY[0]}.{MAX_PY[1]}. "
            f"Tạo venv: py -3.11 -m venv .venv"
        )
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    print(f"Virtualenv  : {'có' if in_venv else 'KHÔNG (đang dùng Python hệ thống)'}")
    return problems


def check_requirements(files: list[Path]) -> list[str]:
    problems: list[str] = []
    rows: list[tuple[str, str, str, str]] = []
    seen: set[str] = set()

    for req_file in files:
        for dist, op, pinned in parse_requirements(req_file):
            if dist.lower() in seen:
                continue
            seen.add(dist.lower())
            actual = installed_version(dist)
            want = f"{op}{pinned}" if op else "(bất kỳ)"
            if actual is None:
                if dist.lower() in OPTIONAL:
                    status = "OPTIONAL"
                else:
                    status = "THIẾU"
                    problems.append(f"Thiếu package: {dist} {want}")
            elif op == "==" and actual != pinned:
                status = "LỆCH"
                problems.append(f"{dist}: cài {actual}, requirements ghim {pinned}")
            else:
                status = "OK"
            rows.append((dist, want, actual or "-", status))

    width = max((len(r[0]) for r in rows), default=10)
    print(f"\n{'PACKAGE'.ljust(width)}  {'YÊU CẦU'.ljust(12)}  {'ĐÃ CÀI'.ljust(12)}  TRẠNG THÁI")
    print("-" * (width + 42))
    for dist, want, actual, status in rows:
        print(f"{dist.ljust(width)}  {want.ljust(12)}  {actual.ljust(12)}  {status}")
    return problems


def check_imports() -> list[str]:
    """Import thử các module cốt lõi — bắt lỗi ABI (numpy 2 vs mediapipe…)."""
    import importlib
    problems: list[str] = []
    print("\nImport thử:")
    for dist in ("flask", "opencv-python", "mediapipe", "numpy",
                 "scikit-learn", "joblib", "tensorflow", "keras"):
        module = IMPORT_NAME.get(dist, dist)
        try:
            importlib.import_module(module)
            print(f"  {module:<14} OK")
        except Exception as exc:  # noqa: BLE001 - báo cáo mọi lỗi import
            print(f"  {module:<14} FAIL: {type(exc).__name__}: {str(exc)[:90]}")
            problems.append(f"import {module} thất bại: {type(exc).__name__}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description="GuardianPilot environment check")
    parser.add_argument("--strict", action="store_true",
                        help="exit 1 nếu phát hiện vấn đề")
    parser.add_argument("--no-imports", action="store_true",
                        help="bỏ qua bước import thử (nhanh hơn)")
    args = parser.parse_args()

    print("=== GuardianPilot — kiểm tra môi trường ===\n")
    problems = check_python()
    problems += check_requirements([
        ROOT / "requirements.txt",
        ROOT / "requirements-dev.txt",
    ])
    if not args.no_imports:
        problems += check_imports()

    print()
    if problems:
        print(f"PHÁT HIỆN {len(problems)} VẤN ĐỀ:")
        for p in problems:
            print(f"  - {p}")
        print("\nKhắc phục:")
        print("  py -3.11 -m venv .venv")
        print("  .venv\\Scripts\\python -m pip install -r requirements.txt -r requirements-dev.txt")
        return 1 if args.strict else 0

    print("Môi trường OK — đủ điều kiện chạy app.py và pytest.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
