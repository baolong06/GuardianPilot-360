"""
app.py — Drowsiness Detection Web Server
Chạy: python app.py [--port 5000] [--host 127.0.0.1]
"""
from __future__ import annotations

import argparse
import base64
import os
import sys
import time
import threading
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request, send_file

# Đảm bảo import được src/
BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

from src.landmarks import extract_features
from src.fusion import FusionState
from src.pipeline import run_holistic
from src.alert_manager import AlertManager, ALERT_MESSAGES, channels_for_level
from src.event_logger import EventLogger
from src.scoring import DriverState
from src.metrics import collect_metrics, InferenceWatchdog
from src.camera_obstruction import CameraObstructionDetector
from src.context import DrivingContext
from src.trip_memory import TripMemory
from src.phone_distraction import (
    PhoneDistractionDetector,
    wrists_from_pose,
    face_geometry_from_landmarks,
)
from src import thresholds as threshold_store
from src.model_loader import load_drowsiness_bundle
from src.runtime_profile import apply_process_limits, get_runtime_profile
from src.video_output import VideoOutputError, VideoOutputStore

apply_process_limits()

# ── Paths ────────────────────────────────────────────────────────────────────
MODELS_DIR = BASE_DIR / "models"
_holistic_task_path: Path = MODELS_DIR / "holistic_landmarker.task"

app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "web" / "templates"),
    static_folder=str(BASE_DIR / "web" / "static"),
)

_video_outputs = VideoOutputStore(BASE_DIR / "output")

# ── State ────────────────────────────────────────────────────────────────────
_lock         = threading.Lock()
_mlp_model    = None
_lstm_model   = None
_mlp_scaler   = None
_lstm_scaler  = None
_fusion       = FusionState()
_alert_mgr    = AlertManager()
_event_logger = EventLogger()
_initialized  = False
_init_error: str | None = None
_rule_only_mode: bool = False
_model_load_mode: str | None = None
# Session metadata (MVP — có thể set qua /api/init)
_driver_id: str | None = "driver_demo"
_vehicle_id: str | None = "vehicle_demo"
# GPS giả lập (API-04) — None nếu chưa có hardware
_gps_lat: float | None = None
_gps_lng: float | None = None


def _watchdog_reload():
    """SYS-05: khi inference stale → thử reload model."""
    logger = __import__("logging").getLogger("watchdog")
    logger.warning("Watchdog triggering model reload…")
    try:
        with _lock:
            _load_models()
        logger.info("Watchdog model reload OK")
    except Exception as exc:
        logger.error("Watchdog model reload failed: %s", exc)


_watchdog = InferenceWatchdog(stale_sec=5.0, on_stale=_watchdog_reload)
_camera_obs = CameraObstructionDetector(threshold_sec=10.0)
_driving_ctx = DrivingContext()
_trip_memory = TripMemory()
_phone_det = PhoneDistractionDetector()
_inference_fps: float = 0.0
_last_infer_ts: float | None = None


def _apply_runtime_thresholds():
    """Push HITL knobs into live detectors (best-effort)."""
    t = threshold_store.get_thresholds()
    _fusion.looking_away.yaw_thresh_deg = t["yaw_thresh_deg"]
    _fusion.looking_away.min_duration_ms = t["looking_away_min_sec"] * 1000.0
    _phone_det.near_frac = t["phone_near_frac"]
    _phone_det.min_duration_ms = t["phone_min_sec"] * 1000.0
    _driving_ctx.high_speed_kmh = t["high_speed_kmh"]
    _driving_ctx.long_drive_sec = t["long_drive_sec"]
    from src.scoring import DriverState as DS
    _fusion.scorer.THRESHOLDS[DS.FATIGUE] = (t["fatigue_on"], t["fatigue_on"] - 0.05)
    _fusion.scorer.THRESHOLDS[DS.DROWSY] = (t["drowsy_on"], t["drowsy_on"] - 0.05)
    _fusion.scorer.THRESHOLDS[DS.MICROSLEEP] = (t["microsleep_on"], t["microsleep_on"] - 0.05)


def _prepare_fusion_inputs(result, w: int, h: int, ts_ms: float) -> dict:
    """Phone heuristic + vehicle risk before FusionState.update()."""
    global _inference_fps, _last_infer_ts
    phone = _update_phone_from_result(result, w, h, ts_ms)
    veh = _driving_ctx.update()
    _fusion.phone_suspected = bool(phone.get("phone_suspected"))
    _fusion.risk_multiplier = veh.risk_multiplier
    if _last_infer_ts is not None:
        dt = max(1e-3, (ts_ms - _last_infer_ts) / 1000.0)
        inst = 1.0 / dt
        _inference_fps = 0.8 * _inference_fps + 0.2 * inst if _inference_fps else inst
    _last_infer_ts = ts_ms
    return phone


def _post_fusion(fused: dict, phone: dict, ts_ms: float, face_ok: bool) -> dict:
    fused.update({
        "phone_suspected": phone.get("phone_suspected", False),
        "phone_streak_ms": phone.get("phone_streak_ms", 0.0),
        "phone_count_window": phone.get("phone_count_window", 0),
        "inference_fps": round(_inference_fps, 1),
    })
    if face_ok:
        fused["camera_obstructed"] = _camera_obs.update(True, ts_ms)
    return _enrich_context_and_memory(fused)


def _session_reset():
    global _inference_fps, _last_infer_ts
    _fusion.reset()
    _alert_mgr.reset()
    _camera_obs.reset()
    _driving_ctx.reset()
    _trip_memory.reset()
    _phone_det.reset()
    _inference_fps = 0.0
    _last_infer_ts = None
    _apply_runtime_thresholds()


def _update_phone_from_result(result, w: int, h: int, ts_ms: float) -> dict:
    if result is None:
        return _phone_det.update(ts_ms, face_center=None, face_size=None, wrists=None)
    face_lm = getattr(result, "face_landmarks", None)
    pose_lm = getattr(result, "pose_landmarks", None)
    # TransformedResult stores list-of-lists
    face_pts = face_lm[0] if face_lm else None
    pose_pts = pose_lm[0] if pose_lm else None
    center, size = face_geometry_from_landmarks(face_pts, w, h)
    wrists = wrists_from_pose(pose_pts, w, h) if pose_pts else []
    return _phone_det.update(
        ts_ms, face_center=center, face_size=size, wrists=wrists
    )


def _enrich_context_and_memory(fused: dict) -> dict:
    veh = _driving_ctx.update()
    fused["vehicle_speed"] = veh.speed_kmh
    fused["driving_time_sec"] = veh.driving_time_sec
    fused["risk_multiplier"] = veh.risk_multiplier
    fused["channels"] = fused.get("channels") or channels_for_level(
        int(fused.get("alert_level", 0))
    )
    _trip_memory.update(
        perclos=float(fused.get("perclos_ratio") or fused.get("perclos") or 0.0),
        drowsiness_state=str(fused.get("drowsiness_state", "NORMAL")),
        alert_level=int(fused.get("alert_level", 0)),
        looking_away=bool(fused.get("looking_away")),
        phone_suspected=bool(fused.get("phone_suspected")),
    )
    fused["trip_summary_brief"] = {
        "perclos_peak": _trip_memory.perclos_peak,
        "alert_peak": _trip_memory.alert_peak,
        "samples": _trip_memory.samples,
    }
    return fused


def _apply_alert_and_log(
    fused: dict,
    frame: np.ndarray | None = None,
    feat: dict | None = None,
) -> dict:
    """Cập nhật AlertManager; ghi Event Log khi đổi cấp."""
    state_name = fused.get("drowsiness_state", "NORMAL")
    try:
        state = DriverState[state_name]
    except KeyError:
        state = DriverState(int(fused.get("alert_level", 0)))

    status = _alert_mgr.update(state)
    fused["alert_level"] = status.alert_level
    fused["alert_message"] = status.alert_message
    fused["drowsiness_state"] = status.drowsiness_state
    fused["channels"] = status.channels

    if status.changed:
        ear = None
        neck = None
        perclos = fused.get("perclos_ratio", fused.get("perclos"))
        if feat:
            ear_v = feat.get("ear_avg")
            neck_v = feat.get("neck_tilt")
            if ear_v is not None and ear_v == ear_v:  # not NaN
                ear = float(ear_v)
            if neck_v is not None and neck_v == neck_v:
                neck = float(neck_v)
        _event_logger.log_event(
            status.alert_level,
            driver_id=_driver_id,
            vehicle_id=_vehicle_id,
            ear_avg=ear,
            perclos=perclos,
            neck_tilt=neck,
            frame=frame,
            gps_lat=_gps_lat,
            gps_lng=_gps_lng,
        )
    return fused


def _load_models():
    global _mlp_model, _lstm_model, _mlp_scaler, _lstm_scaler
    global _initialized, _init_error, _rule_only_mode, _model_load_mode
    global _holistic_task_path

    class _IdentityScaler:
        def transform(self, x):
            return x

    class _ConstantNonDrowsyModel:
        def predict(self, x, verbose=0):
            # Output shape compatible with keras binary model: [[P(non_drowsy)]]
            return np.ones((x.shape[0], 1), dtype=np.float32)

    try:
        bundle = load_drowsiness_bundle(BASE_DIR)
        _mlp_model = bundle["mlp_model"]
        _lstm_model = bundle["lstm_model"]
        _mlp_scaler = bundle["mlp_scaler"]
        _lstm_scaler = bundle["lstm_scaler"]
        _holistic_task_path = bundle["holistic_task"]
        _model_load_mode = bundle["load_mode"]

        # warm-up inference để tránh timeout ở frame đầu
        dummy_mlp = np.zeros((1, 9), dtype=np.float32)
        dummy_seq = np.zeros((30, 6), dtype=np.float32)
        _mlp_model.predict(_mlp_scaler.transform(dummy_mlp), verbose=0)
        seq_scaled = _lstm_scaler.transform(dummy_seq).reshape(1, 30, 6)
        _lstm_model.predict(seq_scaled, verbose=0)

        _initialized = True
        _init_error = None
        _rule_only_mode = False
    except Exception as exc:
        # P0-6: production phải FAIL nếu full model không load được.
        # Default ALLOW_RULE_ONLY_MODE=false (block silent fallback).
        # Dev/CI muốn rule-only: ALLOW_RULE_ONLY_MODE=true python app.py
        allow_rule_only = os.getenv("ALLOW_RULE_ONLY_MODE", "false").lower() in {
            "1", "true", "yes", "on"
        }
        if not allow_rule_only:
            _init_error  = str(exc)
            _initialized = False
            _rule_only_mode = False
            raise

        # Fallback mode: keep service usable without trained artifacts.
        _mlp_model = _ConstantNonDrowsyModel()
        _lstm_model = _ConstantNonDrowsyModel()
        _mlp_scaler = _IdentityScaler()
        _lstm_scaler = _IdentityScaler()
        _initialized = True
        _rule_only_mode = True
        short_reason = str(exc).splitlines()[0][:240]
        _model_load_mode = "rule-only"
        _init_error = (
            f"Rule-only mode enabled: {short_reason}. "
            "Run: python tools/convert_models.py --in-place"
        )


def _decode_image(data: str) -> np.ndarray:
    if not data or not data.strip():
        raise ValueError("Image data is empty or whitespace.")
    if "," in data:
        data = data.split(",", 1)[1]
    raw   = base64.b64decode(data)
    if len(raw) == 0:
        raise ValueError("Decoded image is empty (0 bytes).")
    arr   = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError(f"Không giải mã được ảnh (raw bytes: {len(raw)})")
    return frame


# Màu cho annotation (BGR)
_CLR_NORMAL   = (100, 200, 100)   # xanh lá nhạt
_CLR_DROWSY   = (60,  60,  200)   # đỏ
_CLR_LANDMARK = (0,   200, 80)    # xanh lá đậm
_CLR_POSE     = (80,  160, 255)   # cam
_CLR_NECK     = (0,   165, 255)   # vàng cam
_CLR_EYE      = (40,  140, 255)   # cam đậm (eye-closure rule)

# Face mesh connections cho viền mắt và miệng
_EYE_L  = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
_EYE_R  = [33,  7,   163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
_MOUTH  = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
           375, 321, 405, 314, 17, 84, 181, 91, 146]

def _draw_contour(frame, pts_px, indices, color, closed=True):
    pts = np.array([(int(pts_px[i][0]), int(pts_px[i][1])) for i in indices
                    if 0 <= i < len(pts_px)], dtype=np.int32)
    if len(pts) >= 2:
        cv2.polylines(frame, [pts], closed, color, 1, cv2.LINE_AA)


def _annotate_frame(frame: np.ndarray, result, feat: dict | None,
                    fused: dict | None) -> np.ndarray:
    """Vẽ landmarks + trạng thái fusion lên frame. Trả về frame đã vẽ."""
    out = frame.copy()
    h, w = out.shape[:2]

    alarm_on   = fused["alarm_on"]    if fused else False
    ema_prob   = fused["ema_prob"]   if fused else 0.0
    neck_alarm = fused["neck_alarm"]  if fused else False
    eye_alarm  = fused.get("eye_alarm", False)   if fused else False
    yawn_alarm = fused.get("yawn_alarm", False)  if fused else False
    status_clr = _CLR_DROWSY if alarm_on else _CLR_NORMAL

    # ── Face landmarks ─────────────────────────────────────────────────────
    if result and result.face_landmarks:
        face_lm_raw = result.face_landmarks
        # TransformedResult: List[List[lm]] → unwrap to primary face's landmarks
        if isinstance(face_lm_raw, list) and face_lm_raw and isinstance(face_lm_raw[0], list):
            face_lm_raw = face_lm_raw[0]
        pts_px = [(lm.x * w, lm.y * h) for lm in face_lm_raw]

        # Chấm nhỏ cho toàn bộ face mesh (mỏng, không che mặt) — đổi cam khi eye-alarm
        dot_color = _CLR_EYE if eye_alarm else _CLR_LANDMARK
        for (px, py) in pts_px:
            cv2.circle(out, (int(px), int(py)), 1, dot_color, -1)

        # Viền mắt trái / phải / miệng rõ hơn
        contour_color = _CLR_EYE if eye_alarm else _CLR_LANDMARK
        _draw_contour(out, pts_px, _EYE_L, contour_color)
        _draw_contour(out, pts_px, _EYE_R, contour_color)
        _draw_contour(out, pts_px, _MOUTH, _CLR_LANDMARK)

    # ── Pose landmarks: vai + mũi ──────────────────────────────────────────
    if result and result.pose_landmarks:
        pose = result.pose_landmarks
        # TransformedResult wraps pose as List[List[lm]]; unwrap to first list
        if isinstance(pose, list) and pose and isinstance(pose[0], list):
            pose = pose[0]
        def ppt(i): return (int(pose[i].x * w), int(pose[i].y * h))
        try:
            nose, l_sh, r_sh = ppt(0), ppt(11), ppt(12)
            mid_sh = ((l_sh[0] + r_sh[0]) // 2, (l_sh[1] + r_sh[1]) // 2)
            cv2.line(out, l_sh, r_sh, _CLR_POSE, 2, cv2.LINE_AA)
            cv2.line(out, mid_sh, nose, _CLR_NECK if neck_alarm else _CLR_POSE,
                     2, cv2.LINE_AA)
            cv2.circle(out, nose, 4, _CLR_POSE, -1)
            cv2.circle(out, l_sh,  4, _CLR_POSE, -1)
            cv2.circle(out, r_sh,  4, _CLR_POSE, -1)
        except IndexError:
            pass

    # ── HUD text ───────────────────────────────────────────────────────────
    font = cv2.FONT_HERSHEY_SIMPLEX
    label = "DROWSY" if alarm_on else "NORMAL"

    # Tính chiều cao thanh HUD: mỗi alarm thêm 1 dòng
    n_alarm_lines = (1 if neck_alarm else 0) + (1 if eye_alarm else 0) + (1 if yawn_alarm else 0)
    bar_h = 70 + 25 * n_alarm_lines

    # Nền mờ cho text
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h),
                  (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, out, 0.45, 0, out)

    cv2.putText(out, f"STATUS: {label}  (p={ema_prob:.2f})",
                (10, 30), font, 0.8, status_clr, 2, cv2.LINE_AA)

    if fused and fused.get("p_mlp_drowsy") is not None:
        mlp_v  = fused["p_mlp_drowsy"]
        lstm_v = fused.get("p_lstm_drowsy")
        detail = f"MLP={mlp_v:.2f}"
        if lstm_v is not None:
            detail += f"  LSTM={lstm_v:.2f}"
        cv2.putText(out, detail, (10, 55), font, 0.55,
                    (160, 160, 160), 1, cv2.LINE_AA)

    # Cảnh báo rule-based
    y_off = 80
    if neck_alarm:
        cv2.putText(out, "NECK-TILT ALARM", (10, y_off), font, 0.65,
                    _CLR_NECK, 2, cv2.LINE_AA)
        y_off += 25
    if eye_alarm:
        cv2.putText(out, "EYE-CLOSED ALARM", (10, y_off), font, 0.65,
                    _CLR_EYE, 2, cv2.LINE_AA)
        y_off += 25
    if yawn_alarm:
        cv2.putText(out, "YAWN DETECTED", (10, y_off), font, 0.65,
                    (0, 200, 255), 2, cv2.LINE_AA)
        y_off += 25

    # EAR/MAR mini readout góc phải
    if feat:
        ear = feat.get("ear_avg")
        mar = feat.get("mar")
        if ear is not None:
            cv2.putText(out, f"EAR {ear:.3f}", (w - 120, 25),
                        font, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
        if mar is not None:
            cv2.putText(out, f"MAR {mar:.3f}", (w - 120, 45),
                        font, 0.5, (180, 180, 180), 1, cv2.LINE_AA)

    return out


def _encode_frame(frame: np.ndarray, quality: int = 80) -> str:
    """Encode BGR frame → base64 JPEG string."""
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")


@app.route("/api/init", methods=["POST"])
def api_init():
    """Load model + warm-up. Gọi 1 lần trước khi phân tích."""
    global _fusion, _alert_mgr, _driver_id, _vehicle_id, _gps_lat, _gps_lng
    body = request.get_json(silent=True) or {}
    with _lock:
        _session_reset()
        if "driver_id" in body:
            _driver_id = body.get("driver_id")
        if "vehicle_id" in body:
            _vehicle_id = body.get("vehicle_id")
        if "gps_lat" in body:
            _gps_lat = body.get("gps_lat")
        if "gps_lng" in body:
            _gps_lng = body.get("gps_lng")
        if "speed_kmh" in body:
            _driving_ctx.set_speed(float(body["speed_kmh"]))
        try:
            _load_models()
            if _rule_only_mode:
                return jsonify({
                    "ok": True,
                    "message": "Initialized in rule-only mode (models not found).",
                    "rule_only_mode": True,
                    "warning": _init_error,
                })
            return jsonify({
                "ok": True,
                "message": "Models loaded.",
                "rule_only_mode": False,
                "load_mode": _model_load_mode,
            })
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/api/runtime-profile")
def api_runtime_profile():
    """Cấu hình runtime theo EDGE_PROFILE (dev vs automotive edge)."""
    return jsonify({"ok": True, **get_runtime_profile()})


@app.route("/api/status")
def api_status():
    return jsonify({
        "ok":          True,
        "initialized": _initialized,
        "error":       _init_error,
        "alert_level": _alert_mgr.alert_level,
        "rule_only_mode": _rule_only_mode,
        "load_mode": _model_load_mode,
        "runtime_profile": get_runtime_profile().get("profile"),
    })


@app.route("/api/video-output/start", methods=["POST"])
def api_video_output_start():
    """Create an MP4 writer for a browser-driven video analysis session."""
    if not _initialized:
        return jsonify({"ok": False, "error": "Hệ thống chưa khởi tạo."}), 400

    body = request.get_json(silent=True) or {}
    try:
        info = _video_outputs.start(
            original_name=str(body.get("filename") or "video"),
            width=int(body.get("width", 0)),
            height=int(body.get("height", 0)),
            fps=float(body.get("fps", 0)),
        )
    except (TypeError, ValueError, VideoOutputError) as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    return jsonify({"ok": True, **info})


@app.route("/api/video-output/<output_id>/finish", methods=["POST"])
def api_video_output_finish(output_id: str):
    """Finalize an output file so it can be downloaded or played."""
    try:
        info = _video_outputs.finish(output_id)
    except VideoOutputError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 404

    if info["download_ready"]:
        info["download_url"] = f"/api/video-output/{output_id}/download"
    return jsonify({"ok": True, **info})


@app.route("/api/video-output/<output_id>/download")
def api_video_output_download(output_id: str):
    """Download a completed annotated video without exposing server paths."""
    try:
        path, filename = _video_outputs.get_download(output_id)
    except VideoOutputError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 404
    return send_file(
        path,
        mimetype="video/mp4",
        as_attachment=True,
        download_name=filename,
        conditional=True,
    )


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    """Nhận 1 frame base64 → trả về kết quả fusion (+ annotated frame nếu annotate=true)."""
    if not _initialized:
        return jsonify({"ok": False, "error": "Hệ thống chưa khởi tạo."}), 400

    body = request.get_json(silent=True) or {}
    img_data = body.get("image")
    if not img_data:
        return jsonify({"ok": False, "error": "Thiếu trường 'image'."}), 400

    reset_state = body.get("reset_state", False)
    annotate    = body.get("annotate", False)
    output_id   = str(body.get("output_id") or "").strip() or None

    source_timestamp_ms = body.get("source_timestamp_ms")
    if source_timestamp_ms is not None:
        try:
            source_timestamp_ms = float(source_timestamp_ms)
            if not np.isfinite(source_timestamp_ms) or source_timestamp_ms < 0:
                raise ValueError
        except (TypeError, ValueError):
            return jsonify({"ok": False, "error": "source_timestamp_ms không hợp lệ."}), 400

    try:
        frame = _decode_image(img_data)
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

    with _lock:
        if reset_state:
            _session_reset()

        h, w = frame.shape[:2]
        # Uploaded video must use its media timeline. Using server wall-clock time
        # would make duration-based rules depend on inference speed.
        ts_ms = source_timestamp_ms if source_timestamp_ms is not None else time.time() * 1000.0

        try:
            result = run_holistic(frame, str(_holistic_task_path))
        except Exception as exc:
            return jsonify({"ok": False, "error": f"MediaPipe: {exc}"}), 500

        feat = extract_features(result, w, h)
        phone = _prepare_fusion_inputs(result, w, h, ts_ms)
        veh = _driving_ctx.snapshot()

        if feat is None:
            _fusion.touch(ts_ms)  # sync last_ts_ms — prevent stale-dt on next face
            obstructed = _camera_obs.update(False, ts_ms)
            resp = {
                "ok":               True,
                "face_found":       False,
                "alarm_on":         _fusion.alarm_on,
                "ema_prob":         round(_fusion.ema_prob or 0.0, 4),
                "neck_alarm":       False,
                "eye_alarm":        False,
                "yawn_alarm":       False,
                "perclos":          round(_fusion.perclos_tracker.get_perclos(), 4),
                "perclos_ratio":    round(_fusion.perclos_tracker.get_perclos(), 4),
                "drowsiness_state": _fusion.scorer.get_state_name(),
                "drowsiness_score": 0.0,
                "alert_level":      _alert_mgr.alert_level,
                "alert_message":    ALERT_MESSAGES.get(_alert_mgr.alert_level, ""),
                "channels":         channels_for_level(_alert_mgr.alert_level),
                "camera_obstructed": obstructed,
                "looking_away":     False,
                "phone_suspected":  phone.get("phone_suspected", False),
                "vehicle_speed":    veh.speed_kmh,
                "driving_time_sec": veh.driving_time_sec,
                "risk_multiplier":  veh.risk_multiplier,
            }
            if annotate:
                fused_stub = {
                    "alarm_on": _fusion.alarm_on,
                    "ema_prob": round(_fusion.ema_prob or 0.0, 4),
                    "neck_alarm": False,
                    "eye_alarm": False,
                    "yawn_alarm": False,
                    "p_mlp_drowsy": None, "p_lstm_drowsy": None,
                    "ear_smooth": getattr(_fusion, "ear_smooth", None),
                    "eyes_open_streak_ms": round(_fusion.eyes_open_streak_ms, 1),
                    "eye_closed_streak_ms": round(_fusion.eye_closed_streak_ms, 1),
                }
                annotated = _annotate_frame(frame, result, None, fused_stub)
                if output_id:
                    try:
                        resp["output_frame_count"] = _video_outputs.append(output_id, annotated)
                    except VideoOutputError as exc:
                        return jsonify({"ok": False, "error": str(exc)}), 400
                resp["annotated_frame"] = _encode_frame(annotated)
            return jsonify(resp)

        fused = _fusion.update(
            feat, _mlp_model, _lstm_model,
            _mlp_scaler, _lstm_scaler,
            timestamp_ms=ts_ms,
        )
        fused = _apply_alert_and_log(fused, frame=frame, feat=feat)
        fused = _post_fusion(fused, phone, ts_ms, face_ok=True)

        # ── Debug extras ──────────────────────────────────────────────────
        fused["ear_smooth"] = round(_fusion.ear_smooth, 3) if _fusion.ear_smooth is not None else None
        fused["eyes_open_streak_ms"] = round(_fusion.eyes_open_streak_ms, 1)
        fused["eye_closed_streak_ms"] = round(_fusion.eye_closed_streak_ms, 1)
        fused["neck_recovered_streak_ms"] = round(_fusion.neck_recovered_streak_ms, 1)

        _watchdog.heartbeat()

        resp = {
            "ok":       True,
            "face_found": True,
            "features": {k: (None if (isinstance(v, float) and v != v) else v)
                         for k, v in feat.items()},
            **fused,
        }
        if annotate:
            annotated = _annotate_frame(frame, result, feat, fused)
            if output_id:
                try:
                    resp["output_frame_count"] = _video_outputs.append(output_id, annotated)
                except VideoOutputError as exc:
                    return jsonify({"ok": False, "error": str(exc)}), 400
            resp["annotated_frame"] = _encode_frame(annotated)
        return jsonify(resp)


@app.route("/api/analyze_lite", methods=["POST"])
def api_analyze_lite():
    """
    Endpoint nhẹ dành riêng cho Web Worker (inference loop).
    - Không trả annotated_frame (tiết kiệm encode ~10-15ms)
    - Trả landmark coords để frontend tự vẽ lên canvas
    - Ảnh đầu vào thường nhỏ (320×240) → MediaPipe nhanh hơn
    """
    if not _initialized:
        return jsonify({"ok": False, "error": "Hệ thống chưa khởi tạo."}), 400

    body = request.get_json(silent=True) or {}
    img_data = body.get("image")
    if not img_data:
        return jsonify({"ok": False, "error": "Thiếu trường 'image'."}), 400

    try:
        frame = _decode_image(img_data)
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

    with _lock:
        h, w = frame.shape[:2]
        ts_ms = time.time() * 1000.0

        try:
            result = run_holistic(frame, str(_holistic_task_path))
        except Exception as exc:
            return jsonify({"ok": False, "error": f"MediaPipe: {exc}"}), 500

        feat = extract_features(result, w, h)
        phone = _prepare_fusion_inputs(result, w, h, ts_ms)
        veh = _driving_ctx.snapshot()

        # ── Serialize landmarks (normalized 0-1 coords) ──────────────────
        face_lm = None
        pose_lm = None

        if result and result.face_landmarks:
            pts = result.face_landmarks
            # TransformedResult: list-of-faces; take primary
            if pts and isinstance(pts[0], (list, tuple)):
                pts = pts[0]
            face_lm = [round(coord, 4)
                       for lm in pts
                       for coord in (lm.x, lm.y)]

        if result and result.pose_landmarks:
            pose = result.pose_landmarks
            if pose and isinstance(pose[0], (list, tuple)):
                pose = pose[0]
            def _p(i):
                if i < len(pose):
                    lm = pose[i]
                    return [round(lm.x, 4), round(lm.y, 4),
                            round(getattr(lm, 'visibility', 1.0), 3)]
                return None
            pose_lm = {
                "nose":  _p(0),
                "l_eye": _p(2),
                "r_eye": _p(5),
                "l_ear": _p(7),
                "r_ear": _p(8),
                "l_sh":  _p(11),
                "r_sh":  _p(12),
                "l_el":  _p(13),
                "r_el":  _p(14),
                "l_wr":  _p(15),
                "r_wr":  _p(16),
            }

        if feat is None:
            _fusion.touch(ts_ms)  # sync last_ts_ms — prevent stale-dt on next face
            obstructed = _camera_obs.update(False, ts_ms)
            return jsonify({
                "ok":               True,
                "face_found":       False,
                "alarm_on":         _fusion.alarm_on,
                "ema_prob":         round(_fusion.ema_prob or 0.0, 4),
                "neck_alarm":       False,
                "eye_alarm":        False,
                "yawn_alarm":       False,
                "perclos":          round(_fusion.perclos_tracker.get_perclos(), 4),
                "perclos_ratio":    round(_fusion.perclos_tracker.get_perclos(), 4),
                "drowsiness_state": _fusion.scorer.get_state_name(),
                "drowsiness_score": 0.0,
                "alert_level":      _alert_mgr.alert_level,
                "alert_message":    ALERT_MESSAGES.get(_alert_mgr.alert_level, ""),
                "channels":         channels_for_level(_alert_mgr.alert_level),
                "camera_obstructed": obstructed,
                "looking_away":     False,
                "phone_suspected":  phone.get("phone_suspected", False),
                "vehicle_speed":    veh.speed_kmh,
                "driving_time_sec": veh.driving_time_sec,
                "features":         None,
                "face_landmarks":   face_lm,
                "pose_landmarks":   pose_lm,
            })

        fused = _fusion.update(
            feat, _mlp_model, _lstm_model,
            _mlp_scaler, _lstm_scaler,
            timestamp_ms=ts_ms,
        )
        fused = _apply_alert_and_log(fused, frame=frame, feat=feat)
        fused = _post_fusion(fused, phone, ts_ms, face_ok=True)

        _watchdog.heartbeat()

        return jsonify({
            "ok":       True,
            "face_found": True,
            "features": {k: (None if (isinstance(v, float) and v != v) else v)
                         for k, v in feat.items()},
            "face_landmarks": face_lm,
            "pose_landmarks": pose_lm,
            **fused,
        })


@app.route("/api/metrics")
def api_metrics():
    """SYS-04: CPU/RAM/(GPU)/uptime + trạng thái watchdog."""
    data = collect_metrics()
    data["watchdog"] = _watchdog.status()
    data["initialized"] = _initialized
    data["inference_fps"] = round(_inference_fps, 1)
    data["save_face_snapshots"] = _event_logger.save_face_snapshots
    return jsonify({"ok": True, **data})


@app.route("/api/vehicle", methods=["GET", "POST"])
def api_vehicle():
    """Mock CAN: GET current speed/context; POST {speed_kmh} to update."""
    if request.method == "POST":
        body = request.get_json(silent=True) or {}
        if "speed_kmh" not in body:
            return jsonify({"ok": False, "error": "Thiếu speed_kmh"}), 400
        with _lock:
            _driving_ctx.set_speed(float(body["speed_kmh"]))
            snap = _driving_ctx.update()
        return jsonify({"ok": True, **snap.__dict__})
    with _lock:
        snap = _driving_ctx.snapshot()
    return jsonify({"ok": True, **snap.__dict__})


@app.route("/api/trip/summary")
def api_trip_summary():
    with _lock:
        summary = _trip_memory.summary(
            driving_time_sec=_driving_ctx.driving_time_sec
        )
    return jsonify({"ok": True, **summary})


@app.route("/api/thresholds", methods=["GET", "PUT"])
def api_thresholds():
    """HITL: GET/PUT runtime thresholds (Safety Engineer)."""
    if request.method == "GET":
        return jsonify({
            "ok": True,
            "thresholds": threshold_store.get_thresholds(),
            "defaults": threshold_store.get_defaults(),
            "audit": threshold_store.audit_log(),
        })
    body = request.get_json(silent=True) or {}
    actor = body.pop("actor", "engineer")
    if body.get("reset"):
        th = threshold_store.reset_thresholds()
    else:
        patch = body.get("thresholds", body)
        th = threshold_store.update_thresholds(patch, actor=actor)
    with _lock:
        _apply_runtime_thresholds()
    return jsonify({"ok": True, "thresholds": th, "audit": threshold_store.audit_log()})


@app.route("/api/events")
def api_events():
    """GET /api/events?driver_id=&date=&limit= — danh sách event log."""
    driver_id = request.args.get("driver_id") or None
    date = request.args.get("date") or None
    try:
        limit = int(request.args.get("limit", 50))
    except ValueError:
        limit = 50
    events = _event_logger.get_events(driver_id=driver_id, date=date, limit=limit)
    return jsonify({"ok": True, "count": len(events), "events": events})


@app.route("/api/events/<int:event_id>/snapshot")
def api_event_snapshot(event_id: int):
    """Trả ảnh snapshot — chỉ có khi DEBUG SAVE_FACE_SNAPSHOTS=true."""
    if not _event_logger.save_face_snapshots:
        return jsonify({
            "ok": False,
            "error": "Privacy mode: face snapshots disabled (metadata-only).",
        }), 403

    event = _event_logger.get_event(event_id)
    if event is None:
        return jsonify({"ok": False, "error": "Event không tồn tại."}), 404
    path = event.get("snapshot_path")
    if not path or not Path(path).is_file():
        return jsonify({"ok": False, "error": "Không có snapshot."}), 404
    return send_file(path, mimetype="image/jpeg")


@app.route("/api/events/sync", methods=["POST"])
def api_events_sync():
    """
    Mock batch upload — metadata only (no face bytes / snapshot_path).
    """
    body = request.get_json(silent=True) or {}
    ids = body.get("event_ids")
    if ids:
        pending = [_event_logger.get_event(int(i)) for i in ids]
        pending = [e for e in pending if e]
        updated = _event_logger.mark_uploaded([int(i) for i in ids])
    else:
        pending = _event_logger.get_pending_upload(limit=int(body.get("limit", 100)))
        ids = [e["id"] for e in pending]
        updated = _event_logger.mark_uploaded(ids)

    payload = _event_logger.to_sync_payload(pending)
    print(f"[events/sync] mock upload {updated} metadata events (no faces)")
    return jsonify({
        "ok": True,
        "uploaded_count": updated,
        "event_ids": ids,
        "events": payload,
        "message": "Mock sync complete — metadata only, no face images",
    })


@app.route("/api/reset", methods=["POST"])
def api_reset():
    with _lock:
        _session_reset()
    return jsonify({"ok": True})


def main():
    parser = argparse.ArgumentParser(description="Drowsiness Detection Server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    _watchdog.start()
    print(f"\n=== Drowsiness Detection ===")
    print(f"    http://{args.host}:{args.port}\n")
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == "__main__":
    main()
