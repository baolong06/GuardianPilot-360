import os, glob

print("=== MODEL WEIGHTS (.pth files) ===")
pth_files = glob.glob("models/**/*.pth", recursive=True)
if pth_files:
    for f in pth_files:
        size = os.path.getsize(f)
        print(f"  {f}  ({size/1024:.1f} KB)")
else:
    print("  NONE FOUND")

print()
print("=== PROCESSED DATA ===")
checks = [
    "data/processed/unified_train.json",
    "data/processed/unified_val.json",
    "data/processed/pedestrian_behavior/train.json",
    "data/processed/pedestrian_behavior/val.json",
    "data/processed/pie_trajectory/vehicle_train.json",
    "data/processed/pie_trajectory/pedestrian_train.json",
    "data/processed/pie_behavior/vehicle_behavior_train.json",
    "data/processed/pie/traffic_light_images",
    "data/processed/pie_intention/train.json",
]
for p in checks:
    if os.path.isdir(p):
        n = len(os.listdir(p))
        status = "OK  " if n > 0 else "EMPTY"
        print(f"  {status} {p}  ({n} files)")
    elif os.path.exists(p):
        sz = os.path.getsize(p)
        print(f"  OK   {p}  ({sz/1024:.1f} KB)")
    else:
        print(f"  MISS {p}")

print()
print("=== PIPELINE.PY imports check ===")
try:
    import sys
    sys.path.insert(0, ".")
    from pipeline.pipeline import PedestrianCVPipeline
    print("  OK pipeline.pipeline")
except Exception as e:
    print(f"  FAIL pipeline.pipeline: {e}")
