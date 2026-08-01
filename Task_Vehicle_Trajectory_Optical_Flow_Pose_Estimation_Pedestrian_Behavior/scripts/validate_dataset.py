import os
import json
import logging
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("DatasetValidator")

def validate_unified_dataset(json_path: str = "data/processed/unified_dataset.json"):
    if not os.path.exists(json_path):
        logger.error(f"Unified dataset file not found: {json_path}")
        return False

    with open(json_path, 'r') as f:
        data: List[Dict[str, Any]] = json.load(f)

    logger.info(f"Loaded {len(data)} frames for validation.")

    corrupted_frames = 0
    invalid_bboxes = 0
    missing_ids = 0
    valid_classes = {"car", "truck", "bus", "motorcycle", "bicycle", "person", "traffic light", "traffic sign", "crosswalk", "sign", "transit_station", "unknown"}

    for item_idx, item in enumerate(data):
        if ("frame_id" not in item and "frame_name" not in item) or "objects" not in item:
            corrupted_frames += 1
            continue

        for obj in item["objects"]:
            if "id" not in obj:
                missing_ids += 1

            bbox = obj.get("bbox", [])
            if len(bbox) != 4 or bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
                invalid_bboxes += 1
                logger.warning(f"Invalid BBox in Frame {item.get('frame_id')}, Obj ID {obj.get('id')}: {bbox}")

            cls_name = obj.get("class", "unknown")
            if cls_name not in valid_classes:
                logger.warning(f"Unexpected class label '{cls_name}' at Frame {item.get('frame_id')}")

    logger.info("=== VALIDATION SUMMARY ===")
    logger.info(f"Total Frames Validated: {len(data)}")
    logger.info(f"Corrupted Frame Records: {corrupted_frames}")
    logger.info(f"Invalid BBoxes Found: {invalid_bboxes}")
    logger.info(f"Objects Missing Track IDs: {missing_ids}")

    if corrupted_frames == 0 and invalid_bboxes == 0:
        logger.info("✅ DATASET PASSED ALL INTEGRITY CHECKS!")
        return True
    else:
        logger.error("❌ DATASET CONTAINS INTEGRITY ERRORS.")
        return False

if __name__ == "__main__":
    validate_unified_dataset()
