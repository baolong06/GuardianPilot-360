import os
import json
import logging
from glob import glob
from typing import List, Dict, Any
from parse_bdd100k import parse_bdd100k_label_json
from parse_pie import parse_pie_xml

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("UnifiedDatasetBuilder")

def compute_velocities(sequence_frames: List[Dict[str, Any]], fps: float = 30.0) -> List[Dict[str, Any]]:
    """
    Computes velocity vectors [vx, vy] (in pixels/sec) for objects across sequential frames (e.g. PIE clips).
    """
    dt = 1.0 / fps
    id_prev_pos = {}

    for frame in sorted(sequence_frames, key=lambda x: x.get("frame_id", 0)):
        for obj in frame.get("objects", []):
            obj_id = obj["id"]
            if "center" in obj:
                cx, cy = obj["center"]
            elif "bbox" in obj:
                bbox = obj["bbox"]
                cx, cy = (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0
            else:
                continue

            if obj_id in id_prev_pos:
                prev_cx, prev_cy = id_prev_pos[obj_id]
                vx = round((cx - prev_cx) / dt, 2)
                vy = round((cy - prev_cy) / dt, 2)
                obj["velocity"] = [vx, vy]

            id_prev_pos[obj_id] = (cx, cy)

    return sequence_frames

def build_unified_dataset(output_json_path: str = "data/processed/unified_dataset.json"):
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    unified_records = []

    # 1. Parse BDD100K Annotations (Independent Images - No velocity calculation)
    bdd_val_path = "data/raw/BDD100K/val/annotations/bdd100k_labels_images_val.json"
    bdd_val_img_dir = "data/raw/BDD100K/val/images"
    if os.path.exists(bdd_val_path):
        logger.info("Parsing BDD100K validation labels...")
        bdd_frames = parse_bdd100k_label_json(bdd_val_path, base_img_dir=bdd_val_img_dir)
        for f in bdd_frames:
            f["dataset_source"] = "BDD100K"
        unified_records.extend(bdd_frames)

    # 2. Parse PIE Spatial Annotations (Sequential Clips - Compute velocities)
    pie_spatial_xmls = glob("data/raw/pie/annotations/**/*_annt.xml", recursive=True)
    if not pie_spatial_xmls:
        pie_spatial_xmls = glob("data/raw/pie/annotations/**/*.xml", recursive=True)
    logger.info(f"Found {len(pie_spatial_xmls)} PIE spatial XML annotation files.")

    from parse_pie import parse_pie_spatial_tracks
    for xml_file in pie_spatial_xmls:
        clip_name, pie_dict = parse_pie_spatial_tracks(xml_file)
        if not pie_dict:
            continue
        
        sequence_frames = []
        for frame_no, objs in pie_dict.items():
            sequence_frames.append({
                "frame_id": frame_no,
                "image_path": f"pie_clips/{clip_name}/{frame_no:05d}.jpg",
                "timestamp_sec": round(frame_no * 0.0333, 3),
                "dataset_source": "PIE",
                "sequence_id": clip_name,
                "objects": objs
            })
        
        sequence_frames = compute_velocities(sequence_frames)
        unified_records.extend(sequence_frames)

    logger.info(f"Total unified frame records collected: {len(unified_records)}")

    with open(output_json_path, 'w') as f:
        json.dump(unified_records, f, indent=2)

    logger.info(f"✅ Successfully exported unified dataset to {output_json_path}")

if __name__ == "__main__":
    build_unified_dataset()

