# Drowsiness Detection

Phát hiện buồn ngủ realtime dùng MediaPipe Holistic Landmark + MLP + LSTM + Fusion.

## Cấu trúc

```
app.py              # Flask server
src/
  landmarks.py      # EAR, MAR, head-pose, neck-tilt
  fusion.py         # FusionState (MLP + LSTM + neck rule + EMA + debounce)
  pipeline.py       # MediaPipe wrapper
models/             # .keras + .pkl (copy từ results/)
web/
  templates/        # index.html
  static/css/       # style.css
  static/js/        # app.js
results/            # output gốc từ notebook (nguồn model)
```

## Chạy

```bash
pip install -r requirements.txt
python app.py --port 5000
```

Mở trình duyệt: http://127.0.0.1:5000

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

Nếu thiếu hoặc lỗi tương thích Keras, server tự fallback **rule-only mode** (eye/neck/yawn rules).
Tắt fallback: `ALLOW_RULE_ONLY_MODE=false`.

**Test webcam end-to-end:** xem [docs/WEBCAM_E2E.md](docs/WEBCAM_E2E.md)

```bash
docker compose build && docker compose up -d
curl -X POST http://localhost:5000/api/init
```

Kỳ vọng: `"rule_only_mode": false`, `"load_mode": "weights"`.

## Pipeline

```
Frame (webcam / ảnh / video)
  └── MediaPipe Holistic → EAR, MAR, pitch/yaw/roll, neck-tilt
        ├── MLP (per-frame, 9 features)  → p_mlp_drowsy
        ├── LSTM (window 30 frames, 6 features) → p_lstm_drowsy
        └── Neck-tilt rule (baseline EMA)  → neck_alarm
              └── fusion: max(p_mlp, p_lstm) OR neck_alarm
                    └── EMA(α=0.2) → hysteresis(ON=0.65, OFF=0.35) → alarm_on
```
