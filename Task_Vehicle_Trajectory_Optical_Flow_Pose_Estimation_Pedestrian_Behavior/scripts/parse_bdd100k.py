import os
import json
import logging
from typing import Dict, List, Any, Generator

try:
    import ijson
    HAS_IJSON = True
except ImportError:
    HAS_IJSON = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("BDD100K_Parser")

LABEL_MAPPING = {
    "car": "car",
    "truck": "truck",
    "bus": "bus",
    "motorcycle": "motorcycle",
    "bicycle": "bicycle",
    "pedestrian": "person",
    "person": "person",
    "rider": "person",
    "traffic light": "traffic light",
    "traffic sign": "traffic sign"
}

def parse_single_item(item: Dict[str, Any], idx: int, base_img_dir: str = "") -> Dict[str, Any]:
    """
    Parses a single BDD100K image item into a normalized ground-truth schema.
    Does NOT fabricate trajectory, velocity, pseudo-timestamps, or behavior labels.
    """
    image_name = item.get("name", f"frame_{idx}.jpg")
    img_attributes = item.get("attributes", {})
    labels = item.get("labels", [])

    # Normalize image-level metadata
    weather = img_attributes.get("weather", "unknown")
    scene = img_attributes.get("scene", "unknown")
    time_of_day = img_attributes.get("timeofday", "unknown")
    timestamp = item.get("timestamp", None)

    image_path = os.path.join(base_img_dir, image_name) if base_img_dir else image_name

    frame_objects = []
    for obj_idx, obj in enumerate(labels):
        category = obj.get("category", "unknown")
        mapped_class = LABEL_MAPPING.get(category, "unknown")
        if mapped_class == "unknown":
            continue

        box2d = obj.get("box2d", {})
        if not box2d:
            continue

        x1 = float(box2d.get("x1", 0))
        y1 = float(box2d.get("y1", 0))
        x2 = float(box2d.get("x2", 0))
        y2 = float(box2d.get("y2", 0))

        if x2 <= x1 or y2 <= y1:
            continue

        obj_id = int(obj.get("id", obj_idx))
        obj_attr = obj.get("attributes", {})

        # Extract genuine ground-truth attributes
        occluded = bool(obj_attr.get("occluded", False))
        truncated = bool(obj_attr.get("truncated", False))
        traffic_light_color = obj_attr.get("trafficLightColor", "none")

        frame_objects.append({
            "id": obj_id,
            "category": mapped_class,
            "original_category": category,
            "bbox": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
            "attributes": {
                "occluded": occluded,
                "truncated": truncated,
                "traffic_light_color": traffic_light_color
            }
        })

    return {
        "frame_name": image_name,
        "image_path": image_path,
        "timestamp": timestamp,
        "attributes": {
            "weather": weather,
            "scene": scene,
            "time_of_day": time_of_day
        },
        "objects": frame_objects
    }


def stream_bdd100k_label_json(json_path: str, base_img_dir: str = "") -> Generator[Dict[str, Any], None, None]:
    """
    Memory-efficient generator that streams BDD100K items using ijson (O(1) RAM usage).
    Falls back to standard json.load iterator if ijson is not installed.
    """
    if not os.path.exists(json_path):
        logger.error(f"File not found: {json_path}")
        return

    logger.info(f"Streaming BDD100K annotations from: {json_path}")
    count = 0

    if HAS_IJSON:
        with open(json_path, 'rb') as f:
            for item in ijson.items(f, 'item'):
                count += 1
                if count % 5000 == 0:
                    logger.info(f"Processed {count} image records (streaming via ijson)...")
                yield parse_single_item(item, count, base_img_dir)
    else:
        logger.warning("ijson not installed; falling back to json.load (higher memory usage).")
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, dict):
                data = [data]
            for idx, item in enumerate(data):
                if (idx + 1) % 5000 == 0:
                    logger.info(f"Processed {idx + 1}/{len(data)} image records...")
                yield parse_single_item(item, idx, base_img_dir)

    logger.info(f"Completed processing {count} total frame records.")


def parse_bdd100k_label_json(json_path: str, base_img_dir: str = "") -> List[Dict[str, Any]]:
    """
    Wrapper function returning a full list of parsed frames for backward compatibility.
    """
    return list(stream_bdd100k_label_json(json_path, base_img_dir))


if __name__ == "__main__":
    sample_file = "data/raw/BDD100K/val/annotations/bdd100k_labels_images_val.json"
    val_img_dir = "data/raw/BDD100K/val/images"
    if os.path.exists(sample_file):
        res = parse_bdd100k_label_json(sample_file, base_img_dir=val_img_dir)
        print(f"\nSample parsed frames count: {len(res)}")
        if res:
            print("\nSample Parsed Record Schema:")
            print(json.dumps(res[0], indent=2))

