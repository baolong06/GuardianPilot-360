# GuardianPilot 360 — Drowsiness Detection

Phát hiện buồn ngủ realtime dùng MediaPipe Holistic Landmark + MLP + LSTM + Fusion.

## Yêu cầu môi trường

**Python 3.9 – 3.12** (khuyến nghị 3.11). MediaPipe 0.10.14 và TensorFlow 2.17
chưa có wheel cho Python 3.13/3.14 — cài trên đó sẽ fail.

```bash
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt -r requirements-dev.txt
.venv\Scripts\python tools/check_env.py     # kiểm tra version khớp requirements
```

## Cấu trúc

```
app.py              # Flask server
src/
  landmarks.py      # EAR, MAR, head-pose, neck-tilt
  camera.py         # camera intrinsics cho head-pose
  fusion.py         # FusionState (MLP + LSTM + neck rule + EMA + debounce)
  pipeline.py       # MediaPipe wrapper
  session.py        # state theo từng tài xế/tab (SessionStore)
  auth.py           # API key cho endpoint nhạy cảm
  scoring.py        # state machine 5 cấp
  perclos.py        # PERCLOS rolling window 30s
models/             # .keras + .pkl + .task (ưu tiên models/compatible/)
web/                # templates + static (JS/CSS)
tools/              # script tiện ích & chẩn đoán
results/            # output gốc từ notebook + script phân tích offline
```

## Chạy

```bash
.venv\Scripts\python app.py --port 5000
```

Mở trình duyệt: http://127.0.0.1:5000 · Dashboard: http://127.0.0.1:5000/dashboard

### Phân tích video và tải kết quả

1. Nhấn **Khởi tạo** để load MediaPipe + MLP + LSTM.
2. Chọn tab **Video**, kéo thả hoặc chọn file MP4/WebM/MOV.
3. Chọn FPS phân tích (mặc định 5 FPS), sau đó nhấn **Phân tích video**.
4. Video gốc và frame nhận diện được hiển thị cạnh nhau trong lúc xử lý.
5. Khi hoàn tất, nhấn **Tải xuống MP4**. Server đồng thời lưu file tại `output/`.

Video kết quả không giữ âm thanh gốc. Codec ưu tiên H.264 (`avc1`) để phát được
trực tiếp trên trình duyệt; nếu bản OpenCV không hỗ trợ, tự động lùi về `mp4v`
(file vẫn tải được nhưng cần trình xem ngoài). Khi chạy Docker Compose,
`output/` được mount ra host nên file còn lại sau khi container dừng.

## Nhiều tài xế / nhiều tab

Server giữ state nhận diện **riêng cho từng `session_id`**. Frontend tự sinh id
mỗi tab và gửi qua header `X-Session-Id`; request không gửi gì thì rơi vào
session `default`. Xem session đang sống: `GET /api/status` → `sessions`.

## Bảo mật

Mặc định mọi endpoint đều mở (phù hợp demo localhost). Đặt biến môi trường để
bật xác thực cho các endpoint chứa dữ liệu tài xế:

```bash
GUARDIANPILOT_API_KEY=<key-bí-mật> python app.py
```

Khi bật, các endpoint sau cần header `X-API-Key`: `PUT /api/thresholds`,
`GET /api/events`, `/api/events/<id>/snapshot`, `POST /api/events/sync`,
`GET /api/trip/summary`, `GET /api/metrics`. Dashboard đọc key từ
`localStorage.setItem('gp_api_key', '<key>')`.

`/api/analyze`, `/api/analyze_lite` và `GET /api/thresholds` luôn mở để demo
webcam chạy được.

## Model artifacts & Docker

Chạy convert **một lần** trước khi deploy Docker (full ML mode):

```bash
python tools/convert_models.py --in-place
```

File cần có (ưu tiên `models/compatible/`):

- `mlp_drowsiness_landmark.weights.h5`
- `lstm_drowsiness_landmark.weights.h5`
- `landmark_scaler.pkl`
- `lstm_seq_scaler.pkl`
- `holistic_landmarker.task`

Nếu thiếu hoặc lỗi tương thích Keras, server tự fallback **rule-only mode**
(eye/neck/yawn rules). Tắt fallback: `ALLOW_RULE_ONLY_MODE=false`.

**Test webcam end-to-end:** xem [docs/WEBCAM_E2E.md](docs/WEBCAM_E2E.md)
**Calib camera cho head-pose:** xem [docs/CAMERA_CALIBRATION.md](docs/CAMERA_CALIBRATION.md)

```bash
docker compose build && docker compose up -d
curl -X POST http://localhost:5000/api/init
```

Kỳ vọng: `"rule_only_mode": false`, `"load_mode": "weights"`.

## ⚠️ Backend MediaPipe — đừng đổi sang `task`

`HolisticLandmarker` (Tasks API, đọc file `.task`) trong mediapipe 0.10.14 là API
**chưa hoàn thiện**. Trên khuôn mặt thật nó làm **abort cả process**:

```
F0000 packet.cc:138] Check failed: holder_ != nullptr The packet is empty.
```

Đây là `CHECK` ở tầng C++ → `abort()` → Python chết ngay, **`try/except` không bắt được**.
Với server đang chạy: một frame có mặt người = mất toàn bộ process và mọi session.

Vì vậy mặc định là `HOLISTIC_BACKEND=legacy` (`mp.solutions.holistic`) — đã đo trên
cùng ảnh: 478 điểm mặt + 33 điểm pose, ~52 ms/frame, không crash.

`HOLISTIC_BACKEND=task` chỉ dành cho thử nghiệm khi nâng cấp mediapipe.

## Trạng thái model — đọc trước khi tin kết quả

Model MLP/LSTM đi kèm được train **ngoài repo này** (notebook nguồn không có
trong git) và **chưa từng được đánh giá trên test set có nhãn**. Chạy

```bash
python tools/model_calibration.py     # → reports/model_calibration.md
```

để xem model phản ứng thế nào theo EAR. Đo hiện tại cho thấy:

- **MLP** phản ứng đúng: `p_drowsy` 0.95 (mắt nhắm) → 0.15 (mắt mở rõ).
- **LSTM kẹt trong dải 0.53–0.60 trên toàn bộ dải EAR** — gần như không mang
  thông tin. `src/fusion.py` đã bỏ qua LSTM khi nó lệch MLP > 0.15, nhưng nó
  vẫn kéo `drowsiness_score` lên một nền cao.

Nếu cần vận hành thuần rule engine (bỏ hẳn MLP/LSTM) trong lúc chờ retrain:

```bash
FORCE_RULE_ONLY=true python app.py
```

## Biến môi trường

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `EDGE_PROFILE` | `dev` | `dev` \| `edge` — độ phân giải, FPS, bật/tắt LSTM |
| `HOLISTIC_BACKEND` | `legacy` | `legacy` \| `task` \| `auto` — **đừng đổi trừ khi biết rõ**, xem cảnh báo dưới |
| `GUARDIANPILOT_API_KEY` | *(trống)* | bật auth cho endpoint nhạy cảm |
| `FORCE_RULE_ONLY` | `false` | bỏ qua MLP/LSTM, chỉ chạy rule engine |
| `ALLOW_RULE_ONLY_MODE` | `false` | cho phép fallback rule-only khi load model fail |
| `SAVE_FACE_SNAPSHOTS` | `false` | lưu ảnh khuôn mặt khi có cảnh báo (DEBUG) |
| `MAX_UPLOAD_MB` | `12` | giới hạn kích thước ảnh gửi lên |
| `CAMERA_FOCAL_PX` … | *(trống)* | camera intrinsics — xem docs/CAMERA_CALIBRATION.md |

## Pipeline

```
Frame (webcam / ảnh / video)
  └── MediaPipe Holistic → EAR, MAR, pitch/yaw/roll, neck-tilt
        ├── MLP (per-frame, 9 features)  → p_mlp_drowsy
        ├── LSTM (window 30 frames, 6 features) → p_lstm_drowsy
        └── Neck-tilt rule (baseline EMA)  → neck_alarm
              └── fusion: max(p_mlp, p_lstm) OR neck_alarm
                    └── EMA(α=0.3) → hysteresis(ON=0.65, OFF=0.35) → alarm_on
```

Song song, `DrowsinessScorer` chạy state machine 5 cấp
(NORMAL → FATIGUE → DROWSY → MICROSLEEP → CRITICAL) và đẩy ra `alert_level`.

## Test

```bash
.venv\Scripts\python -m pytest          # testpaths=tests đã cấu hình trong pytest.ini
.venv\Scripts\python -m pytest tests/test_fusion.py -q
.venv\Scripts\python -m pytest tests/test_perclos.py::test_eyes_always_open -q
```
