import os
import xml.etree.ElementTree as ET
import json
import logging
from glob import glob
from typing import Dict, List, Any, Tuple, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("PIE_Parser")

LABEL_MAPPING = {
    "pedestrian": "person",
    "person": "person",
    "vehicle": "car",
    "car": "car",
    "truck": "truck",
    "bus": "bus",
    "traffic_light": "traffic light",
    "crosswalk": "crosswalk",
    "sign": "sign",
    "transit_station": "transit_station",
}

PEDESTRIAN_ACTION_MAP = {
    "standing": "stop",
    "walking": "straight",
    "__undefined__": "unknown",
}

CROSS_MAP = {
    "crossing": 1,
    "not-crossing": 0,
    "crossing-irrelevant": -1,
}


def normalize_video_key(filename: str) -> str:
    """
    Extracts canonical video_id from filename by stripping suffixes (_annt, _attributes, _obd).
    Example: 'video_0001_attributes.xml' -> 'video_0001'
    """
    name = os.path.splitext(os.path.basename(filename))[0]
    for suffix in ("_annt", "_attributes", "_obd"):
        if name.endswith(suffix):
            name = name[:-len(suffix)]
    return name


def parse_pie_spatial_tracks(xml_path: str) -> Tuple[str, Dict[int, List[Dict[str, Any]]]]:
    """
    Parses spatial annotation XML (_annt.xml - CVAT format).
    Returns (video_id, frames_dict).
    
    Features:
    - Filters out outside="1" boxes.
    - Computes Foot-Point (bottom-center) [foot_x, foot_y] for pedestrian ground contact.
    - Keeps 4 independent pedestrian attributes: action, look, gesture, cross (no lossy merging).
    """
    if not os.path.exists(xml_path):
        logger.error(f"XML file not found: {xml_path}")
        return "", {}

    video_id = normalize_video_key(xml_path)

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception as e:
        logger.error(f"Failed to parse spatial XML {xml_path}: {e}")
        return video_id, {}

    frames_dict: Dict[int, List[Dict[str, Any]]] = {}

    for track in root.findall(".//track"):
        label_raw = track.get("label", "unknown")
        mapped_class = LABEL_MAPPING.get(label_raw, label_raw)

        for box in track.findall("box"):
            if box.get("outside", "0") == "1":
                continue

            frame_no = int(box.get("frame", 0))
            xtl = float(box.get("xtl", 0))
            ytl = float(box.get("ytl", 0))
            xbr = float(box.get("xbr", 0))
            ybr = float(box.get("ybr", 0))

            if xbr <= xtl or ybr <= ytl:
                continue

            attributes: Dict[str, str] = {}
            for attr in box.findall("attribute"):
                aname = attr.get("name", "")
                if aname:
                    attributes[aname] = attr.text or ""

            obj_id = attributes.get("id", "")
            cx = round((xtl + xbr) / 2.0, 2)
            cy = round((ytl + ybr) / 2.0, 2)
            foot_x = cx
            foot_y = round(ybr, 2)

            if label_raw == "pedestrian":
                action_raw = attributes.get("action", "__undefined__")
                cross_raw = attributes.get("cross", "not-crossing")
                look_raw = attributes.get("look", "not-looking")
                gesture_raw = attributes.get("gesture", "__undefined__")
                occlusion_level = attributes.get("occlusion", "none")

                behavior_dict = {
                    "action": action_raw,
                    "cross": cross_raw,
                    "look": look_raw,
                    "gesture": gesture_raw,
                    "occlusion_level": occlusion_level
                }

                extra = {
                    "behavior_details": behavior_dict,
                    "crossing_action": CROSS_MAP.get(cross_raw, 0),
                    "looking": look_raw,
                    "gesture": gesture_raw,
                    "action": action_raw
                }
            elif label_raw == "vehicle":
                behavior_dict = {"vehicle_type": attributes.get("type", "unknown")}
                extra = {"vehicle_type": attributes.get("type", "unknown")}
            elif label_raw == "traffic_light":
                behavior_dict = {"tl_type": attributes.get("type", "regular"), "tl_state": attributes.get("state", "__undefined__")}
                extra = {"tl_type": attributes.get("type", "regular"), "tl_state": attributes.get("state", "__undefined__")}
            else:
                behavior_dict = {}
                extra = {}

            is_occluded = box.get("occluded", "0") == "1"
            is_keyframe = box.get("keyframe", "1") == "1"

            obj_entry: Dict[str, Any] = {
                "id": obj_id,
                "bbox": [round(xtl, 2), round(ytl, 2), round(xbr, 2), round(ybr, 2)],
                "class": mapped_class,
                "original_label": label_raw,
                "center": [cx, cy],
                "foot_point": [foot_x, foot_y],
                "behavior": PEDESTRIAN_ACTION_MAP.get(attributes.get("action", "__undefined__"), "unknown") if label_raw == "pedestrian" else "moving",
                "occluded": is_occluded,
                "keyframe": is_keyframe,
            }
            obj_entry.update(extra)

            if frame_no not in frames_dict:
                frames_dict[frame_no] = []
            frames_dict[frame_no].append(obj_entry)

    logger.info(f"Parsed {len(frames_dict)} spatial frames for video '{video_id}' from {os.path.basename(xml_path)}")
    return video_id, frames_dict


def parse_pie_pedestrian_attributes(xml_path: str) -> Tuple[str, Dict[str, Dict[str, Any]]]:
    """
    Parses pedestrian global attributes XML (_attributes.xml).
    Returns (video_id, ped_attributes_dict).
    Key: pedestrian_id (e.g. '2_2_189')
    """
    if not os.path.exists(xml_path):
        return "", {}

    video_id = normalize_video_key(xml_path)

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception as e:
        logger.error(f"Failed to parse ped attributes XML {xml_path}: {e}")
        return video_id, {}

    ped_attributes: Dict[str, Dict[str, Any]] = {}

    for ped in root.findall(".//pedestrian"):
        ped_id = ped.get("id", "")
        if not ped_id:
            continue

        def safe_int(v, default=0):
            try: return int(v)
            except: return default

        def safe_float(v, default=0.0):
            try: return float(v)
            except: return default

        ped_attributes[ped_id] = {
            "pedestrian_id": ped_id,
            "age": ped.get("age", "unknown"),
            "gender": ped.get("gender", "unknown"),
            "crossing": safe_int(ped.get("crossing", 0)),
            "intention_prob": safe_float(ped.get("intention_prob", 0.0)),
            "exp_start_point": safe_int(ped.get("exp_start_point", 0)),
            "critical_point": safe_int(ped.get("critical_point", 0)),
            "crossing_point": safe_int(ped.get("crossing_point", 0)),
            "intersection": ped.get("intersection", "unknown"),
            "num_lanes": safe_int(ped.get("num_lanes", 0)),
            "signalized": ped.get("signalized", "unknown"),
            "traffic_direction": ped.get("traffic_direction", "unknown"),
        }

    logger.info(f"Parsed {len(ped_attributes)} pedestrian attribute records for video '{video_id}'")
    return video_id, ped_attributes


def parse_pie_ego_sensor(xml_path: str) -> Tuple[str, Dict[int, Dict[str, Any]]]:
    """
    Parses ego-vehicle CAN/OBD sensor state XML (_obd.xml).
    Returns (video_id, ego_frames_dict).
    Key: frame_id
    """
    if not os.path.exists(xml_path):
        return "", {}

    video_id = normalize_video_key(xml_path)

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception as e:
        logger.error(f"Failed to parse OBD XML {xml_path}: {e}")
        return video_id, {}

    ego_frames: Dict[int, Dict[str, Any]] = {}

    for frame in root.findall(".//frame"):
        frame_id = int(frame.get("id", 0))
        
        def sf(k):
            try: return float(frame.get(k, 0.0))
            except: return 0.0

        ego_frames[frame_id] = {
            "frame_id": frame_id,
            "gps_speed": sf("GPS_speed"),
            "obd_speed": sf("OBD_speed"),
            "heading_angle": sf("heading_angle"),
            "latitude": sf("latitude"),
            "longitude": sf("longitude"),
            "acceleration": [sf("accX"), sf("accY"), sf("accZ")],
            "gyroscope": [sf("gyroX"), sf("gyroY"), sf("gyroZ")],
            "orientation": {
                "pitch": sf("pitch"),
                "roll": sf("roll"),
                "yaw": sf("yaw")
            }
        }

    logger.info(f"Parsed {len(ego_frames)} ego-vehicle OBD frames for video '{video_id}'")
    return video_id, ego_frames


def parse_pie_xml(xml_path: str) -> Dict[int, List[Dict[str, Any]]]:
    """
    Backward compatibility wrapper: parses spatial tracks XML directly.
    """
    _, frames_dict = parse_pie_spatial_tracks(xml_path)
    return frames_dict


if __name__ == "__main__":
    xml_files = glob("data/raw/pie/annotations/**/*.xml", recursive=True)
    if xml_files:
        spatial_xmls = [x for x in xml_files if x.endswith("_annt.xml")]
        sample_xml = spatial_xmls[0] if spatial_xmls else xml_files[0]
        v_id, res = parse_pie_spatial_tracks(sample_xml)
        print(f"\nParsed PIE Video '{v_id}' with {len(res)} frames.")
        first_frame = min(res.keys()) if res else 0
        sample = res.get(first_frame, [])
        print(f"\nSample Spatial Frame {first_frame} ({len(sample)} objects):")
        print(json.dumps(sample[:2], indent=2, ensure_ascii=False))
    else:
        print("No PIE XML files found in data/raw/pie/annotations/")

