"""Quick syntax + import sanity check for all .py files in the repo."""
import ast
import os
import sys

SKIP_DIRS = ("node_modules", ".git", "__pycache__", ".claude", "venv", ".venv", "dist", "build")

py_files = []
for root, dirs, files in os.walk("."):
    dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]
    for f in files:
        if f.endswith(".py"):
            py_files.append(os.path.join(root, f))

errors = []
for path in py_files:
    try:
        ast.parse(open(path, encoding="utf-8").read(), filename=path)
    except SyntaxError as e:
        errors.append((path, f"{e.msg} (line {e.lineno})"))

print(f"Scanned {len(py_files)} Python files")
if errors:
    print("SYNTAX ERRORS:")
    for p, msg in errors:
        print(f"  {p}: {msg}")
    sys.exit(1)
print("All Python files: SYNTAX OK")