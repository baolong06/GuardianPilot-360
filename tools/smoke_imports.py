"""Smoke-test critical imports from the project root."""
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ["ALLOW_RULE_ONLY_MODE"] = "true"  # dev-only

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root (parent of tools/)
print(f"DEBUG: ROOT={ROOT}\nDEBUG: src exists? {os.path.isdir(os.path.join(ROOT, 'src'))}")
sys.path.insert(0, ROOT)

import importlib

mods = [
    "src.fusion",
    "src.metrics",
    "src.perclos",
    "src.scoring",
    "src.frequency",
    "src.looking_away",
    "src.phone_distraction",
    "src.camera_obstruction",
    "src.context",  # driving context
    "src.trip_memory",
    "src.runtime_profile",
    "src.alert_manager",
    "src.thresholds",
    "src.landmarks",
    "src.pipeline",
    "src.event_logger",
    "src.model_loader",
]

ok, fail = [], []
for m in mods:
    try:
        importlib.import_module(m)
        ok.append(m)
    except Exception as exc:
        fail.append((m, type(exc).__name__, str(exc)[:200]))

print(f"Imported OK ({len(ok)}):")
for m in ok:
    print(f"  + {m}")

if fail:
    print("\nFAILED imports:")
    for m, t, e in fail:
        print(f"  ! {m}: {t}: {e}")
    sys.exit(1)
print("\nAll critical imports: OK")