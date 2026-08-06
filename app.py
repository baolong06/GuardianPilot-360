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
from flask import Flask, jsonify, render_template, request

# Đảm bảo import được src/
BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

from src.landmarks import extract_features
from src.fusion import FusionState
from src.pipeline import run_holistic

# ── Paths ────────────────────────────────────────────────────────────────────
MODELS_DIR    = BASE_DIR / "models"
MLP_PATH      = MODELS_DIR / "mlp_drowsiness_landmark.keras"
LSTM_PATH     = MODELS_DIR / "lstm_drowsiness_landmark.keras"
MLP_SCALER    = MODELS_DIR / "landmark_scaler.pkl"
LSTM_SCALER   = MODELS_DIR / "lstm_seq_scaler.pkl"
HOLISTIC_TASK = MODELS_DIR / "holistic_landmarker.task"

app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "web" / "templates"),
    static_folder=str(BASE_DIR / "web" / "static"),
)

# ── State ────────────────────────────────────────────────────────────────────
_lock         = threading.Lock()
_mlp_model    = None
_lstm_model   = None
_mlp_scaler   = None
_lstm_scaler  = None
_fusion       = FusionState()
_initialized  = False
_init_error: str | None = None


def _load_models():
    global _mlp_model, _lstm_model, _mlp_scaler, _lstm_scaler, _initialized, _init_error
    try:
        import keras
        _mlp_model  = keras.saving.load_model(str(MLP_PATH))
        _lstm_model = keras.saving.load_model(str(LSTM_PATH))
        import joblib
        _mlp_scaler  = joblib.load(str(MLP_SCALER))
        _lstm_scaler = joblib.load(str(LSTM_SCALER))
        # warm-up inference để tránh timeout ở frame đầu
        dummy_mlp = np.zeros((1, 9),            dtype=np.float32)
        dummy_seq = np.zeros((1, 30, 6),        dtype=np.float32)
        _mlp_model.predict(dummy_mlp,  verbose=0)
        _lstm_model.predict(dummy_seq, verbose=0)
        _initialized = True
        _init_error  = None
    except Exception as exc:
        _init_error  = str(exc)
        _initialized = False
        raise


def _decode_image(data: str) -> np.ndarray:
    if "," in data:
        data = data.split(",", 1)[1]
    raw   = base64.b64decode(data)
    arr   = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Không giải mã được ảnh")
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
        pts_px = [(lm.x * w, lm.y * h) for lm in result.face_landmarks]

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


@app.route("/api/init", methods=["POST"])
def api_init():
    """Load model + warm-up. Gọi 1 lần trước khi phân tích."""
    global _fusion
    with _lock:
        _fusion.reset()
        try:
            _load_models()
            return jsonify({"ok": True, "message": "Models loaded."})
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/api/status")
def api_status():
    return jsonify({
        "ok":          True,
        "initialized": _initialized,
        "error":       _init_error,
    })


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

    try:
        frame = _decode_image(img_data)
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

    with _lock:
        if reset_state:
            _fusion.reset()

        h, w = frame.shape[:2]
        ts_ms = time.time() * 1000.0

        try:
            result = run_holistic(frame, str(HOLISTIC_TASK))
        except Exception as exc:
            return jsonify({"ok": False, "error": f"MediaPipe: {exc}"}), 500

        feat = extract_features(result, w, h)

        if feat is None:
            resp = {
                "ok":         True,
                "face_found": False,
                "alarm_on":   _fusion.alarm_on,
                "ema_prob":   round(_fusion.ema_prob or 0.0, 4),
                "neck_alarm": False,
                "eye_alarm":  False,
                "yawn_alarm": False,
            }
            if annotate:
                fused_stub = {"alarm_on": _fusion.alarm_on,
                              "ema_prob": round(_fusion.ema_prob or 0.0, 4),
                              "neck_alarm": False,
                              "eye_alarm": False,
                              "yawn_alarm": False,
                              "p_mlp_drowsy": None, "p_lstm_drowsy": None,
                              "ear_smooth": getattr(_fusion, "ear_smooth", None),
                              "eyes_open_streak_ms": round(_fusion.eyes_open_streak_ms, 1),
                              "eye_closed_streak_ms": round(_fusion.eye_closed_streak_ms, 1)}
                resp["annotated_frame"] = _encode_frame(
                    _annotate_frame(frame, result, None, fused_stub))
            return jsonify(resp)

        fused = _fusion.update(
            feat, _mlp_model, _lstm_model,
            _mlp_scaler, _lstm_scaler,
            timestamp_ms=ts_ms,
        )

        # ── Debug extras ──────────────────────────────────────────────────
        fused["ear_smooth"] = round(_fusion.ear_smooth, 3) if _fusion.ear_smooth is not None else None
        fused["eyes_open_streak_ms"] = round(_fusion.eyes_open_streak_ms, 1)
        fused["eye_closed_streak_ms"] = round(_fusion.eye_closed_streak_ms, 1)
        fused["neck_recovered_streak_ms"] = round(_fusion.neck_recovered_streak_ms, 1)

        resp = {
            "ok":       True,
            "face_found": True,
            "features": {k: (None if (isinstance(v, float) and v != v) else v)
                         for k, v in feat.items()},
            **fused,
        }
        if annotate:
            resp["annotated_frame"] = _encode_frame(
                _annotate_frame(frame, result, feat, fused))
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
            result = run_holistic(frame, str(HOLISTIC_TASK))
        except Exception as exc:
            return jsonify({"ok": False, "error": f"MediaPipe: {exc}"}), 500

        feat = extract_features(result, w, h)

        # ── Serialize landmarks (normalized 0-1 coords) ──────────────────
        face_lm = None
        pose_lm = None

        if result and result.face_landmarks:
            # Gửi tất cả 468 điểm dưới dạng flat array [x0,y0,x1,y1,...]
            pts = result.face_landmarks
            face_lm = [round(coord, 4)
                       for lm in pts
                       for coord in (lm.x, lm.y)]

        if result and result.pose_landmarks:
            pose = result.pose_landmarks
            def _p(i):
                if i < len(pose):
                    lm = pose[i]
                    return [round(lm.x, 4), round(lm.y, 4),
                            round(getattr(lm, 'visibility', 1.0), 3)]
                return None
            # Gửi nhiều điểm pose để frontend vẽ skeleton đầy đủ
            pose_lm = {
                "nose":  _p(0),   # mũi
                "l_eye": _p(2),   # mắt trái
                "r_eye": _p(5),   # mắt phải
                "l_ear": _p(7),   # tai trái
                "r_ear": _p(8),   # tai phải
                "l_sh":  _p(11),  # vai trái
                "r_sh":  _p(12),  # vai phải
                "l_el":  _p(13),  # khuỷu tay trái (bonus)
                "r_el":  _p(14),  # khuỷu tay phải (bonus)
            }

        if feat is None:
            return jsonify({
                "ok":         True,
                "face_found": False,
                "alarm_on":   _fusion.alarm_on,
                "ema_prob":   round(_fusion.ema_prob or 0.0, 4),
                "neck_alarm": False,
                "eye_alarm":  False,
                "yawn_alarm": False,
                "features":   None,
            })

        fused = _fusion.update(
            feat, _mlp_model, _lstm_model,
            _mlp_scaler, _lstm_scaler,
            timestamp_ms=ts_ms,
        )

        return jsonify({
            "ok":       True,
            "face_found": True,
            "features": {k: (None if (isinstance(v, float) and v != v) else v)
                         for k, v in feat.items()},
            **fused,
        })


@app.route("/api/reset", methods=["POST"])
def api_reset():
    with _lock:
        _fusion.reset()
    return jsonify({"ok": True})


def main():
    parser = argparse.ArgumentParser(description="Drowsiness Detection Server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    print(f"\n=== Drowsiness Detection ===")
    print(f"    http://{args.host}:{args.port}\n")
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == "__main__":
    main()
