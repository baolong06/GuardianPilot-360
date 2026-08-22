# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Ràng buộc môi trường (đọc trước khi làm bất cứ việc gì)

Python mặc định của máy dev là **3.14** nhưng mediapipe 0.10.14 và tensorflow 2.17
**không có wheel cho 3.13/3.14**. Repo dùng venv Python 3.11 riêng — luôn gọi qua
đường dẫn tuyệt đối của nó, đừng gọi `python` trần:

```bash
.venv/Scripts/python -m pytest tests/ -q      # Git Bash
.venv\Scripts\python -m pytest tests\ -q      # PowerShell
```

Nếu `.venv` chưa có: `py -3.11 -m venv .venv` rồi
`.venv\Scripts\python -m pip install -r requirements.txt -r requirements-dev.txt`
(tải ~2GB). Kiểm tra bằng `.venv\Scripts\python tools/check_env.py`.

`numpy` **phải** < 2 (ABI của mediapipe). `scikit-learn` **phải** là 1.6.1 — các
file `models/compatible/*.pkl` được pickle bằng đúng version đó.

Console Windows mặc định cp1252, script có tiếng Việt cần `PYTHONIOENCODING=utf-8`.

## Lệnh thường dùng

```bash
.venv\Scripts\python app.py --port 5000        # chạy server
.venv\Scripts\python -m pytest                 # pytest.ini đã set testpaths=tests
.venv\Scripts\python -m pytest tests/test_perclos.py::test_eyes_always_open -q
.venv\Scripts\python tools/check_env.py        # đối chiếu version với requirements
.venv\Scripts\python tools/smoke_imports.py    # smoke test import
.venv\Scripts\python tools/model_calibration.py # đo bias model → reports/
.venv\Scripts\python tools/convert_models.py --in-place  # .keras → .weights.h5
```

`tools/` có 8 file tên `test_*.py` **không phải pytest** — chúng là script chẩn đoán
cần server đang chạy ở `127.0.0.1:5000`. `pytest.ini` đặt `testpaths = tests` để
`pytest` trần không gom nhầm chúng.

## Bài toán

DMS (Driver Monitoring System) phát hiện buồn ngủ. **Chỉ có inference** — repo
không chứa training pipeline nào; model được train ở một notebook ngoài repo
(xem `Report.md` §7, vấn đề C3). Không có Object Detection / Tracking / Optical Flow /
Depth / Trajectory — đừng giả định chúng tồn tại.

## Kiến trúc

```
Browser (web/static/js/app.js)
  ├── display loop  — requestAnimationFrame, vẽ ở FPS camera, KHÔNG chờ inference
  └── worker.js     — POST /api/analyze_lite mỗi 100ms (dev) / 200ms (edge)
        ↓
app.py  ── _infer_lock (toàn cục) ──> src/pipeline.py ──> src/landmarks.py
        └── SessionStore (src/session.py) ──> DriverSession ──> src/fusion.py
```

**Tách biệt hai loại state — đây là điểm dễ sai nhất:**

- `_infer_lock` trong `app.py` là khoá **toàn cục**, bảo vệ MediaPipe landmarker
  (singleton, không thread-safe) và model Keras dùng chung.
- Mọi state nhận diện nằm trong `DriverSession`, cấp phát theo `session_id`
  (header `X-Session-Id` → `body.session_id` → `"default"`). Đừng thêm biến state
  ở module scope trong `app.py` — đó chính là bug H2 đã sửa.

**Luồng fusion** (`src/fusion.py::FusionState.update`) chạy song song hai hệ quyết định:

1. `alarm_on` — nhị phân: `combined → EMA(0.3) → hysteresis(0.65/0.35) → debounce 0.5s`
2. `DriverState` — 5 cấp qua `src/scoring.py` (NORMAL→FATIGUE→DROWSY→MICROSLEEP→CRITICAL)

Hai hệ này **cố ý không ép nhau** và có thể bất đồng (`alarm_on=False` trong khi
`drowsiness_state=DROWSY`). Đây là thiết kế có chủ đích, không phải bug.

Rule engine (eye-closure streak, neck-tilt/pitch nod với baseline EMA, yawn state
machine, PERCLOS 30s) chạy **độc lập** với model — nên khi model trả NaN hoặc bị
tắt qua `FORCE_RULE_ONLY=true`, hệ thống vẫn cảnh báo được.

## Quy ước bắt buộc nhớ

- **Model output = P(Non-Drowsy)**, nên `p_drowsy = 1 - output`. Sai chiều là đảo
  ngược toàn bộ hệ thống.
- **Thứ tự feature là hợp đồng với model đã train.** `MLP_FEAT_COLS` (9) và
  `LSTM_FEAT_COLS` (6) trong `src/fusion.py` — đổi thứ tự = model vô nghĩa mà không
  báo lỗi gì.
- **NaN phải bị chặn trước khi tới model.** `StandardScaler` cho NaN đi qua
  (`ensure_all_finite="allow-nan"`), Keras nhân ra NaN, `jsonify` sinh literal `NaN`
  = JSON không hợp lệ = frontend chết. `neck_tilt` là NaN bất cứ khi nào mất pose vai
  — trường hợp rất thường gặp. Luôn `np.nan_to_num` và guard `math.isfinite`.
- **Ngưỡng chỉ có MỘT nguồn:** `src/thresholds.py`. Sửa runtime qua
  `DriverSession.apply_thresholds()` → `FusionState.apply_thresholds()` →
  `DrowsinessScorer.set_threshold()`. Đừng ghi thẳng vào `DrowsinessScorer.THRESHOLDS`
  (class attribute — bug H3 đã sửa). Frontend đọc ngưỡng từ `/api/runtime-profile`.
- **Video dùng media timeline, không dùng wall-clock.** `/api/analyze` nhận
  `source_timestamp_ms` để các rule theo thời lượng không phụ thuộc tốc độ inference.
  Nhưng chỉ số FPS thì phải dùng `time.monotonic()` (`DriverSession.note_inference`).
- **KHÔNG dùng `HolisticLandmarker` (Tasks API).** Trong mediapipe 0.10.14 nó abort
  cả process trên khuôn mặt thật (`Check failed: holder_ != nullptr`) — là `CHECK` của
  C++ nên `try/except` vô dụng. `src/pipeline.py` mặc định `HOLISTIC_BACKEND=legacy`
  (`mp.solutions.holistic`). Đừng "tối ưu" bằng cách quay lại đường `.task`.
- **`mp.solutions.holistic` KHÔNG deterministic** — cùng một ảnh cho EAR lệch tới 0.044
  qua các lần gọi. Đừng viết test khẳng định giá trị EAR tuyệt đối từ ảnh thật.
- **HolisticLandmarker chỉ trả MỘT khuôn mặt.** `_score_person` trong `pipeline.py`
  còn đó để chấm điểm/debug, nhưng nhánh multi-person đã gỡ vì không bao giờ chạy.
- **pitch/yaw/roll là góc TƯƠNG ĐỐI** (camera chưa calib) — chỉ dùng qua
  delta-với-baseline. Xem `docs/CAMERA_CALIBRATION.md`.

## Model — đừng tin số liệu mà chưa đo

`reports/model_calibration.md` (sinh bằng `tools/model_calibration.py`) cho thấy
**LSTM kẹt trong dải 0.53–0.60 trên toàn bộ dải EAR** (biên độ 0.067) — gần như
không mang thông tin. MLP thì ổn (biên độ 0.808). Guard
`abs(p_lstm - p_mlp) > 0.15 → chỉ dùng MLP` trong `fusion.py` chính là thứ giữ hệ
thống chạy được. Chưa có evaluation trên test set có nhãn.

## Privacy & bảo mật

`SAVE_FACE_SNAPSHOTS=false` là mặc định và phải giữ nguyên. `to_sync_payload()`
strip `snapshot_path` — **không bao giờ** để ảnh khuôn mặt vào payload đồng bộ.

Auth (`src/auth.py`) tắt khi chưa set `GUARDIANPILOT_API_KEY`. Khi thêm endpoint mới
chạm dữ liệu tài xế, nhớ gắn `@require_api_key`. Endpoint phục vụ demo webcam
(`/api/analyze`, `/api/analyze_lite`, `GET /api/thresholds`) cố ý để mở.

## Tài liệu tham khảo trong repo

- `Report.md` — 24 vấn đề đã sửa, kèm bằng chứng và việc còn nợ
- `PRD_GuardianPilot360.md` — mô tả nhiều thứ **chưa được xây** (TensorRT, FastAPI,
  React). Code thực tế là Flask + vanilla JS. Đối chiếu code trước khi tin PRD.
- `docs/WEBCAM_E2E.md`, `docs/CAMERA_CALIBRATION.md`
