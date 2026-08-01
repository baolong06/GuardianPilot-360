"""
Temporary audit script - reads PIE annotation files and prints schemas.
Run: python audit_pie_schema.py
"""
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
import json
from pathlib import Path

BASE = Path("data/raw/pie")

def audit_spatial(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Meta
    meta = root.find("meta")
    task = meta.find("task") if meta is not None else None
    size = task.findtext("size") if task is not None else "?"
    orig = task.find("original_size") if task is not None else None
    w = orig.findtext("width") if orig is not None else "?"
    h = orig.findtext("height") if orig is not None else "?"
    print(f"  meta.task.size = {size}")
    print(f"  image = {w}x{h}")

    label_counts = Counter()
    attr_names_by_label = defaultdict(set)
    frame_ids = []
    sample_records = {}

    for track in root.findall(".//track"):
        label = track.get("label", "unknown")
        label_counts[label] += 1
        boxes = track.findall("box")
        for box in boxes[:3]:
            frame_no = int(box.get("frame", 0))
            frame_ids.append(frame_no)
            attrs = {}
            for attr in box.findall("attribute"):
                aname = attr.get("name", "")
                attr_names_by_label[label].add(aname)
                attrs[aname] = attr.text
            if label not in sample_records:
                sample_records[label] = {
                    "track_attribs": dict(track.attrib),
                    "box_attribs": dict(box.attrib),
                    "box_child_attrs": attrs
                }

    print(f"  === Label Counts ===")
    for lbl, cnt in sorted(label_counts.items()):
        print(f"    {lbl}: {cnt} tracks, attrs: {sorted(attr_names_by_label[lbl])}")

    if frame_ids:
        print(f"  Frame range: {min(frame_ids)} - {max(frame_ids)}")

    print(f"  === Sample Records ===")
    for lbl, rec in sample_records.items():
        print(f"  [{lbl}]")
        print(f"    track attribs: {rec['track_attribs']}")
        print(f"    box attribs:   {rec['box_attribs']}")
        print(f"    box attrs:     {rec['box_child_attrs']}")


def audit_attributes(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    print(f"  Root tag: {root.tag}")
    peds = root.findall("pedestrian")
    print(f"  Total pedestrian records: {len(peds)}")
    if peds:
        first = peds[0]
        print(f"  First attribs: {dict(first.attrib)}")
        # Collect all unique attribute names
        all_keys = set()
        crossing_vals = Counter()
        intention_vals = []
        for p in peds:
            all_keys.update(p.attrib.keys())
            crossing_vals[p.get("crossing")] += 1
            val = p.get("intention_prob", "")
            if val:
                try:
                    intention_vals.append(float(val))
                except ValueError:
                    pass
        print(f"  All attribute keys: {sorted(all_keys)}")
        print(f"  crossing value distribution: {dict(crossing_vals)}")
        if intention_vals:
            print(f"  intention_prob range: [{min(intention_vals):.4f}, {max(intention_vals):.4f}]")


def audit_obd(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    print(f"  Root tag: {root.tag}")
    frames = list(root)
    print(f"  Total frame elements: {len(frames)}")
    if frames:
        first = frames[0]
        last = frames[-1]
        print(f"  First frame id: {first.get('id')}, keys: {list(first.attrib.keys())}")
        print(f"  Last frame id:  {last.get('id')}")
        # Check speed values
        gps_speeds = [float(f.get("GPS_speed", 0)) for f in frames[:100]]
        obd_speeds = [float(f.get("OBD_speed", 0)) for f in frames[:100]]
        has_negatives_gps = any(s < 0 for s in gps_speeds)
        has_negatives_obd = any(s < 0 for s in obd_speeds)
        print(f"  GPS_speed sample range (first 100): [{min(gps_speeds):.2f}, {max(gps_speeds):.2f}] | negatives: {has_negatives_gps}")
        print(f"  OBD_speed sample range (first 100): [{min(obd_speeds):.2f}, {max(obd_speeds):.2f}] | negatives: {has_negatives_obd}")


def audit_csv(csv_path):
    with open(csv_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    print(f"  Total lines: {len(lines)}")
    if lines:
        print(f"  Header: {lines[0].strip()}")
        print(f"  Sample row: {lines[1].strip() if len(lines) > 1 else 'N/A'}")


def audit_pie_clips():
    clips_dir = BASE / "pie_clips"
    if not clips_dir.exists():
        print("  pie_clips/ directory not found!")
        return
    for set_dir in sorted(clips_dir.iterdir()):
        videos = list(set_dir.iterdir()) if set_dir.is_dir() else []
        exts = Counter(v.suffix.lower() for v in videos if v.is_file())
        dirs = [v.name for v in videos if v.is_dir()]
        print(f"  {set_dir.name}: {len(videos)} items | ext: {dict(exts)} | video dirs: {dirs[:5]}")


# ============================
# MAIN AUDIT
# ============================
print("=" * 60)
print("STEP 1: SPATIAL ANNOTATIONS")
print("=" * 60)
for set_dir in sorted((BASE / "annotations").iterdir()):
    for xml_file in sorted(set_dir.glob("*.xml")):
        print(f"\n[{set_dir.name}/{xml_file.name}]")
        audit_spatial(xml_file)
        break  # Only audit first xml per set for speed

print("\n" + "=" * 60)
print("STEP 2: OBJECT ATTRIBUTES (ped_attributes)")
print("=" * 60)
for set_dir in sorted((BASE / "annotations_attributes").iterdir()):
    for xml_file in sorted(set_dir.glob("*.xml")):
        print(f"\n[{set_dir.name}/{xml_file.name}]")
        audit_attributes(xml_file)
        break

print("\n" + "=" * 60)
print("STEP 3: EGO-VEHICLE OBD")
print("=" * 60)
for set_dir in sorted((BASE / "annotations_vehicle").iterdir()):
    for xml_file in sorted(set_dir.glob("*.xml")):
        print(f"\n[{set_dir.name}/{xml_file.name}]")
        audit_obd(xml_file)
        break

print("\n" + "=" * 60)
print("STEP 4: CSV ANNOTATED FRAMES")
print("=" * 60)
for set_dir in sorted((BASE / "annotations").iterdir()):
    for csv_file in sorted(set_dir.glob("*.csv")):
        print(f"\n[{set_dir.name}/{csv_file.name}]")
        audit_csv(csv_file)

print("\n" + "=" * 60)
print("STEP 5: PIE CLIPS DIRECTORY")
print("=" * 60)
audit_pie_clips()

print("\nAUDIT COMPLETE")
