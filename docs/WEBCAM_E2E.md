# Hướng dẫn test end-to-end (Webcam qua trình duyệt)

Tài liệu này mô tả quy trình chạy **full ML mode** (MLP + LSTM + Fusion) và test realtime bằng webcam.

## 1. Chuẩn bị model (bắt buộc cho full ML)

Model gốc từ notebook thường được lưu bằng Keras mới (3.15+) và có thể **không load được** trong Docker (Keras 3.5).

Chạy script convert một lần trên máy dev (đã có model gốc trong `models/`):

```bash
pip install -r requirements.txt
python tools/convert_models.py --in-place
```

Script sẽ tạo:

```
models/compatible/
  mlp_drowsiness_landmark.weights.h5
  lstm_drowsiness_landmark.weights.h5
  landmark_scaler.pkl
  lstm_seq_scaler.pkl
  holistic_landmarker.task
```

Với `--in-place`, các file weights/scaler/task cũng được copy vào `models/`.

Kiểm tra nhanh:

```bash
python tools/convert_models.py --verify-only
```

Kỳ vọng: `verify OK (weights)` và in ra giá trị `mlp output`, `lstm output`.

## 2. Chạy local (không Docker)

```bash
python app.py --host 127.0.0.1 --port 5000
```

Mở trình duyệt: http://127.0.0.1:5000

## 3. Chạy bằng Docker

```bash
docker compose build
docker compose up -d
```

Kiểm tra API:

```bash
curl http://localhost:5000/api/status
curl -X POST http://localhost:5000/api/init -H "Content-Type: application/json" -d "{}"
curl http://localhost:5000/api/status
```

Kỳ vọng sau `/api/init`:

```json
{
  "ok": true,
  "initialized": true,
  "rule_only_mode": false,
  "error": null
}
```

Nếu `rule_only_mode: true` → chưa convert model hoặc thiếu file trong `models/compatible/`.

## 4. Test webcam trên UI

1. Mở http://localhost:5000 (hoặc http://127.0.0.1:5000 khi chạy local).
2. Nhấn **Khởi tạo** (góc phải) — đợi 30–60 giây lần đầu.
3. Badge chuyển sang **Sẵn sàng**.
4. Tab **Webcam** → nhấn **Bật webcam** → cho phép quyền camera.
5. Nhấn **Phân tích live**.
6. Quan sát:
   - Canvas overlay landmarks (mắt/miệng/cổ)
   - Panel kết quả bên phải: `MLP`, `LSTM`, `EMA`, trạng thái `NORMAL/DROWSY`
   - Thanh perf: Display FPS, Inference FPS

## 5. Kịch bản test nhanh

| Kịch bản | Cách làm | Kỳ vọng |
|---------|----------|---------|
| Tỉnh táo | Nhìn thẳng camera, mắt mở | `drowsiness_state: NORMAL`, `alarm_on: false` |
| Nhắm mắt lâu | Nhắm mắt > 1s | `eye_alarm: true`, có thể tăng `ema_prob` |
| Gật đầu | Gật đầu mạnh | `neck_alarm: true` |
| Ngáp | Ngáp rõ, giữ miệng mở | `yawn_alarm: true` (sau ~1.2s) |
| Không có mặt | Che camera / quay ra ngoài | `face_found: false`, có thể `camera_obstructed` |

## 6. Debug qua API (không cần UI)

```bash
# Metrics hệ thống
curl http://localhost:5000/api/metrics

# Reset state giữa các lần test
curl -X POST http://localhost:5000/api/reset
```

## 7. Xử lý sự cố

| Triệu chứng | Nguyên nhân | Cách xử lý |
|------------|-------------|------------|
| `rule_only_mode: true` | Model chưa convert / thiếu `.weights.h5` | Chạy `python tools/convert_models.py --in-place` |
| `Hệ thống chưa khởi tạo` | Chưa gọi `/api/init` | Nhấn **Khởi tạo** trên UI |
| Webcam không mở | Trình duyệt chặn quyền camera | Dùng `localhost`, bật quyền Camera |
| Inference FPS thấp | CPU + TensorFlow nặng | Giảm độ phân giải webcam, đóng tab khác |
| Docker volume lỗi | Chưa share ổ đĩa | Docker Desktop → File sharing → bật ổ chứa project |

## 8. Tắt rule-only fallback (production strict)

Trong `docker-compose.yml` hoặc env:

```yaml
environment:
  - ALLOW_RULE_ONLY_MODE=false
```

Khi đó `/api/init` sẽ fail nếu không load được full ML artifacts.
