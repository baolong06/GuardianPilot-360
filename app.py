"""
app.py — Drowsiness Detection Web Server
Chạy: python app.py [--port 5000] [--host 127.0.0.1]
"""
from __future__ import annotations

import argparse
import base64
import logging
import math
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
from src.pipeline import run_holistic
from src.alert_manager import ALERT_MESSAGES, channels_for_level
from src.auth import auth_status, request_is_authorized, require_api_key
from src.camera import describe as describe_camera
from src.event_logger import EventLogger
from src.scoring import DriverState
from src.metrics import collect_metrics, InferenceWatchdog
from src.phone_distraction import (
    wrists_from_pose,
    face_geometry_from_landmarks,
)
from src.session import DEFAULT_SESSION_ID, SessionStore
from src import thresholds as threshold_store
from src.model_loader import load_drowsiness_bundle
from src.runtime_profile import apply_process_limits, get_runtime_profile
from src.video_output import VideoOutputError, VideoOutputStore

apply_process_limits()

logger = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────────
MODELS_DIR = BASE_DIR / "models"
_holistic_task_path: Path = MODELS_DIR / "holistic_landmarker.task"

app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "web" / "templates"),
    static_folder=str(BASE_DIR / "web" / "static"),
)

# M3: chặn payload base64 khổng lồ. `/api/analyze` giữ khoá inference suốt thời
# gian xử lý, nên một request vài trăm MB đủ để treo toàn bộ service.
MAX_UPLOAD_MB = float(os.getenv("MAX_UPLOAD_MB", "12"))
app.config["MAX_CONTENT_LENGTH"] = int(MAX_UPLOAD_MB * 1024 * 1024)
MAX_FRAME_WIDTH = 3840
MAX_FRAME_HEIGHT = 2160

_video_outputs = VideoOutputStore(BASE_DIR / "output")

# ── State ────────────────────────────────────────────────────────────────────
# H2: state nhận diện KHÔNG còn là biến toàn cục dùng chung. Mỗi session_id có
# FusionState/AlertManager/TripMemory… riêng (src/session.py). Request không gửi
# session_id thì rơi vào session "default" ⇒ tương thích ngược hoàn toàn.
#
# `_infer_lock` vẫn là khoá TOÀN CỤC vì nó bảo vệ tài nguyên dùng chung:
# MediaPipe landmarker là singleton và không thread-safe, model Keras cũng dùng
# chung. Tách state theo session giải quyết đúng vấn đề trộn dữ liệu; nó không
# nhằm tăng throughput.
_infer_lock   = threading.Lock()
_sessions     = SessionStore()
_mlp_model    = None
_lstm_model   = None
_mlp_scaler   = None
_lstm_scaler  = None
_event_logger: EventLogger | None = None
_event_logger_lock = threading.Lock()
_initialized  = False
_init_error: str | None = None
_rule_only_mode: bool = False
_model_load_mode: str | None = None

_server_start_time = time.time()

# Giữ alias cho tương thích ngược với script/test cũ đang đọc `_lock`.
_lock = _infer_lock


@app.after_request
def _add_production_headers(response):
    response.headers["X-GuardianPilot-Version"] = "1.0.0"
    response.headers["X-GuardianPilot-System"] = "DMS-Automotive-Edge"
    return response


def _get_event_logger() -> EventLogger:
    """
    M6: khởi tạo EventLogger lười.

    Trước đây nó được tạo ở module scope, nên chỉ cần `import app` (pytest,
    tooling, sphinx…) là đã tạo `data/events.db` + `data/snapshots/` trong cây
    làm việc — và fail hẳn nếu filesystem read-only trong container.
    """
    global _event_logger
    if _event_logger is None:
        with _event_logger_lock:
            if _event_logger is None:
                _event_logger = EventLogger()
    return _event_logger


def _watchdog_reload():
    """SYS-05: khi inference stale → thử reload model."""
    wd_logger = logging.getLogger("watchdog")
    wd_logger.warning("Watchdog triggering model reload…")
    try:
        with _infer_lock:
            _load_models()
        wd_logger.info("Watchdog model reload OK")
    except Exception as exc:
        wd_logger.error("Watchdog model reload failed: %s", exc)


_watchdog = InferenceWatchdog(stale_sec=5.0, on_stale=_watchdog_reload)


# ── Session helpers (H2) ─────────────────────────────────────────────────────
def _session_id_from_request(body: dict | None = None) -> str:
    """Ưu tiên header X-Session-Id, sau đó body.session_id, cuối cùng 'default'."""
    sid = request.headers.get("X-Session-Id")
    if not sid and body:
        sid = body.get("session_id")
    if not sid:
        sid = request.args.get("session_id")
    return sid or DEFAULT_SESSION_ID


def _get_session(body: dict | None = None):
    session = _sessions.get(_session_id_from_request(body))
    session.apply_thresholds(threshold_store.get_thresholds())
    return session


def _apply_runtime_thresholds():
    """
    Push HITL knobs xuống mọi session đang sống.

    H3/H4: việc áp ngưỡng nay đi qua `DriverSession.apply_thresholds()` →
    `FusionState.apply_thresholds()` → `DrowsinessScorer.set_threshold()`.
    Trước đây hàm này ghi thẳng vào `DrowsinessScorer.THRESHOLDS` (class
    attribute) và bỏ quên hoàn toàn 3 knob eye-closure.
    """
    _sessions.apply_thresholds_all(threshold_store.get_thresholds())


def _json_safe(value):
    """
    C2: đổi mọi float không hữu hạn thành None trước khi jsonify.

    `json.dumps` của Python (và do đó `flask.jsonify`) mặc định sinh literal
    `NaN`/`Infinity` — đó KHÔNG phải JSON hợp lệ, `JSON.parse` của trình duyệt
    ném lỗi và cả vòng lặp live chết. Đây là lưới an toàn tầng hai; tầng một là
    các guard ngay tại nguồn trong src/fusion.py.
    """
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.floating):
        v = float(value)
        return v if math.isfinite(v) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def _json_response(payload: dict, status: int = 200):
    return jsonify(_json_safe(payload)), status


def _prepare_fusion_inputs(session, result, w: int, h: int, ts_ms: float) -> dict:
    """Phone heuristic + vehicle risk before FusionState.update()."""
    phone = _update_phone_from_result(session, result, w, h, ts_ms)
    veh = session.driving_context.update()
    session.fusion.phone_suspected = bool(phone.get("phone_suspected"))
    session.fusion.risk_multiplier = veh.risk_multiplier
    # M2: FPS đo bằng đồng hồ thực trong session.note_inference(), KHÔNG dùng
    # ts_ms — với video upload ts_ms là media timeline nên FPS báo ra sẽ là FPS
    # của video chứ không phải tốc độ xử lý của server.
    session.note_inference()
    return phone


def _post_fusion(session, fused: dict, phone: dict, ts_ms: float,
                 face_ok: bool) -> dict:
    fused.update({
        "phone_suspected": phone.get("phone_suspected", False),
        "phone_streak_ms": phone.get("phone_streak_ms", 0.0),
        "phone_count_window": phone.get("phone_count_window", 0),
        "inference_fps": round(session.inference_fps, 1),
    })
    if face_ok:
        fused["camera_obstructed"] = session.camera_obstruction.update(True, ts_ms)
    return _enrich_context_and_memory(session, fused)


def _session_reset(session):
    session.reset()
    session.apply_thresholds(threshold_store.get_thresholds())


def _update_phone_from_result(session, result, w: int, h: int, ts_ms: float) -> dict:
    detector = session.phone_detector
    if result is None:
        return detector.update(ts_ms, face_center=None, face_size=None, wrists=None)
    face_lm = getattr(result, "face_landmarks", None)
    pose_lm = getattr(result, "pose_landmarks", None)
    # TransformedResult stores list-of-lists
    face_pts = face_lm[0] if face_lm else None
    pose_pts = pose_lm[0] if pose_lm else None
    center, size = face_geometry_from_landmarks(face_pts, w, h)
    wrists = wrists_from_pose(pose_pts, w, h) if pose_pts else []
    return detector.update(
        ts_ms, face_center=center, face_size=size, wrists=wrists
    )


def _enrich_context_and_memory(session, fused: dict) -> dict:
    veh = session.driving_context.update()
    fused["vehicle_speed"] = veh.speed_kmh
    fused["driving_time_sec"] = veh.driving_time_sec
    fused["risk_multiplier"] = veh.risk_multiplier
    fused["channels"] = fused.get("channels") or channels_for_level(
        int(fused.get("alert_level", 0))
    )
    session.trip_memory.update(
        perclos=float(fused.get("perclos_ratio") or fused.get("perclos") or 0.0),
        drowsiness_state=str(fused.get("drowsiness_state", "NORMAL")),
        alert_level=int(fused.get("alert_level", 0)),
        looking_away=bool(fused.get("looking_away")),
        phone_suspected=bool(fused.get("phone_suspected")),
    )
    fused["trip_summary_brief"] = {
        "perclos_peak": session.trip_memory.perclos_peak,
        "alert_peak": session.trip_memory.alert_peak,
        "samples": session.trip_memory.samples,
    }
    fused["session_id"] = session.session_id
    return fused


def _apply_alert_and_log(
    session,
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

    status = session.alert_manager.update(state)
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
        _get_event_logger().log_event(
            status.alert_level,
            driver_id=session.driver_id,
            vehicle_id=session.vehicle_id,
            ear_avg=ear,
            perclos=perclos,
            neck_tilt=neck,
            frame=frame,
            gps_lat=session.gps_lat,
            gps_lng=session.gps_lng,
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

    def _enable_rule_only(reason: str, model_load_mode: str = "rule-only"):
        """Chạy bằng rule engine, model trả P(non-drowsy)=1.0 (tức p_drowsy=0)."""
        globals().update(
            _mlp_model=_ConstantNonDrowsyModel(),
            _lstm_model=_ConstantNonDrowsyModel(),
            _mlp_scaler=_IdentityScaler(),
            _lstm_scaler=_IdentityScaler(),
            _initialized=True,
            _rule_only_mode=True,
            _model_load_mode=model_load_mode,
            _init_error=reason,
        )

    # ── C4 (giảm thiểu): chủ động bỏ qua model ────────────────────────────
    # Khác hẳn ALLOW_RULE_ONLY_MODE — cái đó chỉ là fallback KHI load thất bại.
    # FORCE_RULE_ONLY là lựa chọn có chủ đích: báo cáo reports/live_diagnostic.md
    # ghi nhận MLP trả p_drowsy≈0.585 khi mắt đang mở bình thường (EAR=0.30).
    # Chừng nào model chưa được retrain + đánh giá trên test set có nhãn, vận
    # hành bằng rule engine (eye-closure / neck-tilt / yawn / PERCLOS) là lựa
    # chọn hợp lệ và đáng tin hơn.
    if os.getenv("FORCE_RULE_ONLY", "false").lower() in {"1", "true", "yes", "on"}:
        # Vẫn cần file .task của MediaPipe để trích landmark.
        bundle = load_drowsiness_bundle(BASE_DIR)
        _holistic_task_path = bundle["holistic_task"]
        _enable_rule_only(
            "FORCE_RULE_ONLY=true — bỏ qua MLP/LSTM theo cấu hình, "
            "chỉ dùng rule engine.",
            model_load_mode="rule-only-forced",
        )
        return

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
        short_reason = str(exc).splitlines()[0][:240]
        _enable_rule_only(
            f"Rule-only mode enabled: {short_reason}. "
            "Run: python tools/convert_models.py --in-place"
        )


def _decode_image(data: str) -> np.ndarray:
    if not data or not data.strip():
        raise ValueError("Image data is empty or whitespace.")
    if "," in data:
        data = data.split(",", 1)[1]
    # M3: chặn base64 quá khổ trước khi tốn RAM giải mã. MAX_CONTENT_LENGTH của
    # Flask đã chặn ở tầng HTTP; đây là chốt thứ hai cho các caller nội bộ.
    max_raw_bytes = app.config["MAX_CONTENT_LENGTH"]
    if len(data) > max_raw_bytes * 2:
        raise ValueError(f"Ảnh vượt giới hạn {MAX_UPLOAD_MB:g} MB.")
    raw   = base64.b64decode(data)
    if len(raw) == 0:
        raise ValueError("Decoded image is empty (0 bytes).")
    if len(raw) > max_raw_bytes:
        raise ValueError(f"Ảnh vượt giới hạn {MAX_UPLOAD_MB:g} MB.")
    arr   = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError(f"Không giải mã được ảnh (raw bytes: {len(raw)})")
    h, w = frame.shape[:2]
    if w > MAX_FRAME_WIDTH or h > MAX_FRAME_HEIGHT:
        raise ValueError(
            f"Độ phân giải {w}x{h} vượt giới hạn "
            f"{MAX_FRAME_WIDTH}x{MAX_FRAME_HEIGHT}."
        )
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
    body = request.get_json(silent=True) or {}
    with _infer_lock:
        session = _get_session(body)
        _session_reset(session)
        if "driver_id" in body:
            session.driver_id = body.get("driver_id")
        if "vehicle_id" in body:
            session.vehicle_id = body.get("vehicle_id")
        if "gps_lat" in body:
            session.gps_lat = body.get("gps_lat")
        if "gps_lng" in body:
            session.gps_lng = body.get("gps_lng")
        if "speed_kmh" in body:
            session.driving_context.set_speed(float(body["speed_kmh"]))
        try:
            _load_models()
            if _rule_only_mode:
                return jsonify({
                    "ok": True,
                    "message": "Initialized in rule-only mode.",
                    "rule_only_mode": True,
                    "load_mode": _model_load_mode,
                    "warning": _init_error,
                    "session_id": session.session_id,
                })
            return jsonify({
                "ok": True,
                "message": "Models loaded.",
                "rule_only_mode": False,
                "load_mode": _model_load_mode,
                "session_id": session.session_id,
            })
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/api/runtime-profile")
def api_runtime_profile():
    """Cấu hình runtime theo EDGE_PROFILE (dev vs automotive edge)."""
    return jsonify({
        "ok": True,
        **get_runtime_profile(),
        # H5: frontend lấy ngưỡng "mắt nhắm" từ đây thay vì hard-code 0.20,
        # để UI và backend luôn nói cùng một con số.
        "eye_closed_thresh": threshold_store.get_thresholds()["eye_closed_thresh"],
        "camera": describe_camera(),
        **auth_status(),
    })


@app.route("/api/health")
def api_health():
    """Health check endpoint cho container probes / orchestrator."""
    metrics = collect_metrics()
    return jsonify({
        "status": "healthy" if (_initialized or _rule_only_mode) else "starting",
        "ready": _initialized,
        "rule_only_mode": _rule_only_mode,
        "version": "1.0.0",
        "uptime_sec": round(time.time() - _server_start_time, 1),
        "backend": os.getenv("HOLISTIC_BACKEND", "legacy"),
        "profile": get_runtime_profile().get("profile", "dev"),
        "system": {
            "cpu_percent": metrics.get("cpu_percent"),
            "memory_mb": metrics.get("rss_mb"),
        },
    })


@app.route("/api/session/info")
def api_session_info():
    """Lấy thông tin chi tiết của phiên hiện tại."""
    session = _get_session()
    return jsonify({
        "ok": True,
        "session_id": session.session_id,
        "driver_id": session.driver_id,
        "vehicle_id": session.vehicle_id,
        "alert_level": session.alert_manager.alert_level,
        "inference_fps": round(session.inference_fps, 1),
        "trip_memory": {
            "perclos_peak": session.trip_memory.perclos_peak,
            "alert_peak": session.trip_memory.alert_peak,
            "samples": session.trip_memory.samples,
        },
    })


@app.route("/api/status")
def api_status():
    session = _get_session()
    return jsonify({
        "ok":          True,
        "initialized": _initialized,
        "error":       _init_error,
        "alert_level": session.alert_manager.alert_level,
        "rule_only_mode": _rule_only_mode,
        "load_mode": _model_load_mode,
        "runtime_profile": get_runtime_profile().get("profile"),
        "session_id": session.session_id,
        "sessions": _sessions.stats(),
        **auth_status(),
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

    with _infer_lock:
        session = _get_session(body)
        fusion = session.fusion
        if reset_state:
            _session_reset(session)

        h, w = frame.shape[:2]
        # Uploaded video must use its media timeline. Using server wall-clock time
        # would make duration-based rules depend on inference speed.
        ts_ms = source_timestamp_ms if source_timestamp_ms is not None else time.time() * 1000.0

        try:
            result = run_holistic(frame, str(_holistic_task_path))
        except Exception as exc:
            return jsonify({"ok": False, "error": f"MediaPipe: {exc}"}), 500

        feat = extract_features(result, w, h)
        phone = _prepare_fusion_inputs(session, result, w, h, ts_ms)
        veh = session.driving_context.snapshot()
        alert_level = session.alert_manager.alert_level

        if feat is None:
            fusion.touch(ts_ms)  # sync last_ts_ms — prevent stale-dt on next face
            obstructed = session.camera_obstruction.update(False, ts_ms)
            resp = {
                "ok":               True,
                "face_found":       False,
                "session_id":       session.session_id,
                "alarm_on":         fusion.alarm_on,
                "ema_prob":         round(fusion.ema_prob or 0.0, 4),
                "neck_alarm":       False,
                "eye_alarm":        False,
                "yawn_alarm":       False,
                "perclos":          round(fusion.perclos_tracker.get_perclos(), 4),
                "perclos_ratio":    round(fusion.perclos_tracker.get_perclos(), 4),
                "drowsiness_state": fusion.scorer.get_state_name(),
                "drowsiness_score": 0.0,
                "alert_level":      alert_level,
                "alert_message":    ALERT_MESSAGES.get(alert_level, ""),
                "channels":         channels_for_level(alert_level),
                "camera_obstructed": obstructed,
                "looking_away":     False,
                "phone_suspected":  phone.get("phone_suspected", False),
                "vehicle_speed":    veh.speed_kmh,
                "driving_time_sec": veh.driving_time_sec,
                "risk_multiplier":  veh.risk_multiplier,
            }
            if annotate:
                fused_stub = {
                    "alarm_on": fusion.alarm_on,
                    "ema_prob": round(fusion.ema_prob or 0.0, 4),
                    "neck_alarm": False,
                    "eye_alarm": False,
                    "yawn_alarm": False,
                    "p_mlp_drowsy": None, "p_lstm_drowsy": None,
                    "ear_smooth": getattr(fusion, "ear_smooth", None),
                    "eyes_open_streak_ms": round(fusion.eyes_open_streak_ms, 1),
                    "eye_closed_streak_ms": round(fusion.eye_closed_streak_ms, 1),
                }
                annotated = _annotate_frame(frame, result, None, fused_stub)
                if output_id:
                    try:
                        resp["output_frame_count"] = _video_outputs.append(output_id, annotated)
                    except VideoOutputError as exc:
                        return jsonify({"ok": False, "error": str(exc)}), 400
                resp["annotated_frame"] = _encode_frame(annotated)
            return _json_response(resp)

        fused = fusion.update(
            feat, _mlp_model, _lstm_model,
            _mlp_scaler, _lstm_scaler,
            timestamp_ms=ts_ms,
        )
        fused = _apply_alert_and_log(session, fused, frame=frame, feat=feat)
        fused = _post_fusion(session, fused, phone, ts_ms, face_ok=True)

        # ── Debug extras ──────────────────────────────────────────────────
        fused["ear_smooth"] = round(fusion.ear_smooth, 3) if fusion.ear_smooth is not None else None
        fused["eyes_open_streak_ms"] = round(fusion.eyes_open_streak_ms, 1)
        fused["eye_closed_streak_ms"] = round(fusion.eye_closed_streak_ms, 1)
        fused["neck_recovered_streak_ms"] = round(fusion.neck_recovered_streak_ms, 1)

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
        return _json_response(resp)


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

    with _infer_lock:
        session = _get_session(body)
        fusion = session.fusion
        h, w = frame.shape[:2]
        ts_ms = time.time() * 1000.0

        try:
            result = run_holistic(frame, str(_holistic_task_path))
        except Exception as exc:
            return jsonify({"ok": False, "error": f"MediaPipe: {exc}"}), 500

        feat = extract_features(result, w, h)
        phone = _prepare_fusion_inputs(session, result, w, h, ts_ms)
        veh = session.driving_context.snapshot()
        alert_level = session.alert_manager.alert_level

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
            fusion.touch(ts_ms)  # sync last_ts_ms — prevent stale-dt on next face
            obstructed = session.camera_obstruction.update(False, ts_ms)
            return _json_response({
                "ok":               True,
                "face_found":       False,
                "session_id":       session.session_id,
                "alarm_on":         fusion.alarm_on,
                "ema_prob":         round(fusion.ema_prob or 0.0, 4),
                "neck_alarm":       False,
                "eye_alarm":        False,
                "yawn_alarm":       False,
                "perclos":          round(fusion.perclos_tracker.get_perclos(), 4),
                "perclos_ratio":    round(fusion.perclos_tracker.get_perclos(), 4),
                "drowsiness_state": fusion.scorer.get_state_name(),
                "drowsiness_score": 0.0,
                "alert_level":      alert_level,
                "alert_message":    ALERT_MESSAGES.get(alert_level, ""),
                "channels":         channels_for_level(alert_level),
                "camera_obstructed": obstructed,
                "looking_away":     False,
                "phone_suspected":  phone.get("phone_suspected", False),
                "vehicle_speed":    veh.speed_kmh,
                "driving_time_sec": veh.driving_time_sec,
                "features":         None,
                "face_landmarks":   face_lm,
                "pose_landmarks":   pose_lm,
            })

        fused = fusion.update(
            feat, _mlp_model, _lstm_model,
            _mlp_scaler, _lstm_scaler,
            timestamp_ms=ts_ms,
        )
        fused = _apply_alert_and_log(session, fused, frame=frame, feat=feat)
        fused = _post_fusion(session, fused, phone, ts_ms, face_ok=True)

        _watchdog.heartbeat()

        return _json_response({
            "ok":       True,
            "face_found": True,
            "features": {k: (None if (isinstance(v, float) and v != v) else v)
                         for k, v in feat.items()},
            "face_landmarks": face_lm,
            "pose_landmarks": pose_lm,
            **fused,
        })


@app.route("/api/metrics")
@require_api_key
def api_metrics():
    """SYS-04: CPU/RAM/(GPU)/uptime + trạng thái watchdog."""
    session = _get_session()
    data = collect_metrics()
    data["watchdog"] = _watchdog.status()
    data["initialized"] = _initialized
    data["inference_fps"] = round(session.inference_fps, 1)
    data["save_face_snapshots"] = _get_event_logger().save_face_snapshots
    data["sessions"] = _sessions.stats()
    return _json_response({"ok": True, **data})


@app.route("/api/vehicle", methods=["GET", "POST"])
def api_vehicle():
    """Mock CAN: GET current speed/context; POST {speed_kmh} to update."""
    if request.method == "POST":
        body = request.get_json(silent=True) or {}
        if "speed_kmh" not in body:
            return jsonify({"ok": False, "error": "Thiếu speed_kmh"}), 400
        session = _get_session(body)
        with _infer_lock:
            session.driving_context.set_speed(float(body["speed_kmh"]))
            snap = session.driving_context.update()
        return _json_response({"ok": True, "session_id": session.session_id,
                               **snap.__dict__})
    session = _get_session()
    with _infer_lock:
        snap = session.driving_context.snapshot()
    return _json_response({"ok": True, "session_id": session.session_id,
                           **snap.__dict__})


@app.route("/api/trip/summary")
@require_api_key
def api_trip_summary():
    session = _get_session()
    with _infer_lock:
        summary = session.trip_memory.summary(
            driving_time_sec=session.driving_context.driving_time_sec
        )
    return _json_response({"ok": True, "session_id": session.session_id, **summary})


@app.route("/api/thresholds", methods=["GET", "PUT"])
def api_thresholds():
    """HITL: GET/PUT runtime thresholds (Safety Engineer)."""
    if request.method == "GET":
        return _json_response({
            "ok": True,
            "thresholds": threshold_store.get_thresholds(),
            "defaults": threshold_store.get_defaults(),
            "audit": threshold_store.audit_log(),
        })
    # H6: chỉ PUT mới cần key — GET để frontend đọc ngưỡng hiển thị (H5).
    if not request_is_authorized():
        return jsonify({
            "ok": False,
            "error": "Unauthorized — thiếu hoặc sai header X-API-Key.",
        }), 401
    body = request.get_json(silent=True) or {}
    actor = body.pop("actor", "engineer")
    if body.get("reset"):
        th = threshold_store.reset_thresholds()
    else:
        patch = body.get("thresholds", body)
        th = threshold_store.update_thresholds(patch, actor=actor)
    with _infer_lock:
        _apply_runtime_thresholds()
    return _json_response({"ok": True, "thresholds": th,
                           "audit": threshold_store.audit_log()})


@app.route("/api/events")
@require_api_key
def api_events():
    """GET /api/events?driver_id=&date=&limit= — danh sách event log."""
    driver_id = request.args.get("driver_id") or None
    date = request.args.get("date") or None
    try:
        limit = int(request.args.get("limit", 50))
    except ValueError:
        limit = 50
    events = _get_event_logger().get_events(
        driver_id=driver_id, date=date, limit=limit
    )
    return _json_response({"ok": True, "count": len(events), "events": events})


@app.route("/api/events/<int:event_id>/snapshot")
@require_api_key
def api_event_snapshot(event_id: int):
    """Trả ảnh snapshot — chỉ có khi DEBUG SAVE_FACE_SNAPSHOTS=true."""
    event_logger = _get_event_logger()
    if not event_logger.save_face_snapshots:
        return jsonify({
            "ok": False,
            "error": "Privacy mode: face snapshots disabled (metadata-only).",
        }), 403

    event = event_logger.get_event(event_id)
    if event is None:
        return jsonify({"ok": False, "error": "Event không tồn tại."}), 404
    path = event.get("snapshot_path")
    if not path or not Path(path).is_file():
        return jsonify({"ok": False, "error": "Không có snapshot."}), 404
    return send_file(path, mimetype="image/jpeg")


@app.route("/api/events/sync", methods=["POST"])
@require_api_key
def api_events_sync():
    """
    Mock batch upload — metadata only (no face bytes / snapshot_path).
    """
    event_logger = _get_event_logger()
    body = request.get_json(silent=True) or {}
    ids = body.get("event_ids")
    if ids:
        pending = [event_logger.get_event(int(i)) for i in ids]
        pending = [e for e in pending if e]
        updated = event_logger.mark_uploaded([int(i) for i in ids])
    else:
        pending = event_logger.get_pending_upload(limit=int(body.get("limit", 100)))
        ids = [e["id"] for e in pending]
        updated = event_logger.mark_uploaded(ids)

    payload = event_logger.to_sync_payload(pending)
    logger.info("[events/sync] mock upload %d metadata events (no faces)", updated)
    return _json_response({
        "ok": True,
        "uploaded_count": updated,
        "event_ids": ids,
        "events": payload,
        "message": "Mock sync complete — metadata only, no face images",
    })


@app.route("/api/reset", methods=["POST"])
def api_reset():
    body = request.get_json(silent=True) or {}
    session = _get_session(body)
    with _infer_lock:
        _session_reset(session)
    return jsonify({"ok": True, "session_id": session.session_id})


@app.errorhandler(413)
def api_payload_too_large(_exc):
    """M3: trả JSON thay vì trang HTML mặc định của Werkzeug."""
    return jsonify({
        "ok": False,
        "error": f"Payload vượt giới hạn {MAX_UPLOAD_MB:g} MB.",
    }), 413


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
