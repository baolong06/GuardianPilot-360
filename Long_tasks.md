# Task Plan — Long (Backend / AI Pipeline / Alert Logic / Data)

**Dựa trên:** `PRD_GuardianPilot360.md` so với source code hiện tại (`app.py`, `src/`, `tests/`).
**Vai trò:** Long chịu trách nhiệm phần **AI Model, Feature Engineering, Fusion/Alert Logic, Event Logger, API, Dataset & Testing**.
**Trạng thái hiện tại (đã có):**
- MediaPipe Holistic wrapper với multi-person primary selection (`src/pipeline.py`)
- EAR/MAR/head-pose/neck-tilt feature extraction (`src/landmarks.py`)
- Fusion: MLP + LSTM + neck-tilt rule + eye-closure rule + yawn detector + EMA + hysteresis debounce (`src/fusion.py`)
- Flask API: `/api/init`, `/api/analyze`, `/api/analyze_lite`, `/api/reset`, `/api/status`
- 1 file test: `tests/test_eye_closure_rule.py`

---

## Gap Analysis so với PRD (phần của Long)

| PRD Requirement | Hiện trạng | Việc cần làm |
|---|---|---|
| DMS-06: PERCLOS rolling window 30s | ❌ Chưa có — chỉ có `eye_closed_streak_ms` (thời gian nhắm liên tục), không phải tỷ lệ % thời gian nhắm/tổng thời gian | Bổ sung module PERCLOS |
| DMS-11: Microsleep detection | ⚠️ Có nhưng gộp chung logic eye-closure (`eye_microsleep`), chưa tách thành class/module riêng có thể test độc lập | Tách rõ MicrosleepDetector |
| DMS-12/13: Drowsiness Score + phân loại NORMAL/FATIGUE/DROWSY/MICROSLEEP/CRITICAL | ❌ Hiện tại chỉ có `alarm_on` (boolean) — không có 5 cấp độ theo PRD | Bổ sung State Machine 5 cấp |
| ALT-01→05: Cảnh báo 4 cấp tăng dần + tự hạ cấp | ❌ Chỉ có 1 cấp (on/off) | Bổ sung Alert Manager multi-level |
| DMS-14 / SYS-06: Event Logger (ảnh + GPS + lưu trước khi upload) | ❌ Chưa có bất kỳ logging nào | Xây Event Logger (SQLite local) |
| API-01→05: REST API Event Log, retry, batch upload offline, GPS, encryption | ❌ Chưa có endpoint nào cho việc này | Xây API Event Log tối thiểu (MVP scope, có thể mock GPS) |
| SYS-04: Monitor CPU/RAM/GPU/nhiệt độ | ❌ Chưa có | Bổ sung endpoint `/api/metrics` |
| SYS-05: Watchdog restart | ❌ Chưa có | Bổ sung watchdog nhẹ (thread healthcheck) |
| 9.2 Accuracy: Face Detection >95%, Eye State >90%, Drowsiness Recall >90%, FP <5% | ⚠️ Chưa có script đánh giá định lượng | Viết eval script + report |
| Test coverage (mục 5.1.K) | ⚠️ Chỉ có test cho eye-closure rule | Bổ sung test cho neck-tilt, yawn, PERCLOS, drowsiness scoring, multi-person selection |
| requirements.txt thiếu `pytest`, không pin version | ⚠️ Version dùng `>=` (rủi ro breaking change) | Pin version cụ thể + thêm test deps |
| Dataset (mục 10.4) | ⚠️ Chỉ có ảnh/video lẻ trong `results/`, chưa có annotation/tổ chức theo chuẩn | Tổ chức lại dataset + annotation guideline |

---

## Danh sách Task

### P0 — Bắt buộc cho MVP đúng PRD

- [ ] **L1. PERCLOS Module** (`src/perclos.py`)
  - Tính PERCLOS = % thời gian mắt nhắm trong rolling window 30 giây (dùng `deque` lưu (timestamp, is_closed))
  - Input: `ear_smooth` mỗi frame + threshold hiện có (`EYE_CLOSED_THRESH`)
  - Output: `perclos_ratio` (0.0–1.0), đưa vào `FusionState.update()` trả về cùng các field khác
  - Unit test: mô phỏng chuỗi frame nhắm/mở mắt, assert PERCLOS đúng theo tỷ lệ kỳ vọng

- [ ] **L2. Drowsiness Scoring Engine + State Machine 5 cấp** (`src/scoring.py`)
  - Input tổng hợp: `p_mlp_drowsy`, `p_lstm_drowsy`, `perclos_ratio`, `eye_closed_streak_ms`, `yawn_alarm`/tần suất ngáp, `neck_alarm`/head-nod frequency, `has_pose` (face visibility)
  - Output trạng thái: `NORMAL | FATIGUE | DROWSY | MICROSLEEP | CRITICAL` (theo mục 6.1 DMS-13 và luồng nghiệp vụ #3)
  - Logic chuyển trạng thái phải dựa trên chuỗi thời gian (không dùng 1 frame), có debounce tương tự cơ chế hysteresis đã có
  - Unit test cho từng transition NORMAL→FATIGUE→DROWSY→MICROSLEEP→CRITICAL và recovery ngược lại

- [ ] **L3. Alert Manager đa cấp** (`src/alert_manager.py`)
  - Cảnh báo cấp 1 (FATIGUE) → cấp 2 (DROWSY) → cấp 3 (MICROSLEEP) → cấp 4 (CRITICAL, khi không phục hồi sau cảnh báo — theo US1.2 AC5)
  - Tự động hạ cấp khi tài xế phục hồi (US1.5)
  - Trả `alert_level` (0–4) + `alert_message` qua API để frontend hiển thị màu tương ứng (xanh/vàng/cam/đỏ theo UI-04)
  - Ghi lại mỗi lần đổi cấp vào Event Log (US1.2 AC7)

- [ ] **L4. Event Logger (SQLite)** (`src/event_logger.py`)
  - Schema tối thiểu: `id, timestamp, driver_id, vehicle_id, alert_level, ear_avg, perclos, neck_tilt, snapshot_path, gps_lat, gps_lng, uploaded (bool)`
  - Lưu snapshot ảnh (JPEG) khi alert_level ≥ 2 (DROWSY) — theo US1.4 AC5
  - GPS: MVP có thể dùng giá trị giả lập/None nếu chưa có hardware, nhưng field phải tồn tại sẵn (mục A7, API-04)
  - Hàm `log_event()` gọi từ `app.py` mỗi khi `alert_level` thay đổi

- [ ] **L5. REST API cho Event Log** (bổ sung vào `app.py`)
  - `GET /api/events?driver_id=&date=&limit=` — trả danh sách events (US1.8 AC1/AC2)
  - `GET /api/events/<id>/snapshot` — trả ảnh snapshot
  - `POST /api/events/sync` — mock batch upload lên "cloud" (ghi log, đánh dấu `uploaded=True`) — chuẩn bị cho API-02 (retry/batch upload)

- [ ] **L6. Cập nhật `/api/analyze` & `/api/analyze_lite`**
  - Trả thêm: `perclos_ratio`, `drowsiness_state` (NORMAL/FATIGUE/DROWSY/MICROSLEEP/CRITICAL), `alert_level`
  - Đảm bảo không tăng latency quá nhiều (giữ mục tiêu <200ms theo mục 9.1)

### P1 — Quan trọng nhưng có thể làm sau P0

- [ ] **L7. System Metrics Endpoint** (`/api/metrics`)
  - CPU%, RAM%, (GPU% nếu có), uptime — dùng `psutil` (thêm vào `requirements.txt`)
  - Phục vụ SYS-04 và mục 9.3 (Reliability)

- [ ] **L8. Watchdog nhẹ**
  - Thread giám sát: nếu vòng lặp inference không có kết quả mới trong >5s → log warning + tự động reload model
  - Phục vụ SYS-05

- [ ] **L9. Camera Obstruction Detection**
  - Nếu `n_faces == 0` liên tục >10s trong khi trước đó có face → cảnh báo "camera bị che / mất mặt" (A6, R6 trong PRD)
  - Trả `camera_obstructed: bool` trong response

- [ ] **L10. Head-Nod Frequency Tracking**
  - Đếm số lần `neck_alarm` trigger trong rolling window (VD 60s) → dùng cho Drowsiness Score (YHP-02 tương tự nhưng cho head-nod)

- [ ] **L11. Yawn Frequency Counter**
  - Đếm số lần ngáp (`yawn_alarm` trigger) trong cửa sổ thời gian (YHP-02), trả `yawn_count_window`

### P2 — Testing, Data, Docs (nên làm xuyên suốt)

- [ ] **L12. Unit tests bổ sung**
  - `tests/test_perclos.py`, `tests/test_scoring_state_machine.py`, `tests/test_alert_manager.py`, `tests/test_multiperson_selection.py` (cho `src/pipeline.py`)
  - Test riêng cho neck-tilt rule và yawn detector (hiện đang test lồng trong `test_eye_closure_rule.py`, nên tách file cho rõ ràng)

- [ ] **L13. Evaluation Script** (`tools/evaluate.py`)
  - Chạy trên tập video/ảnh có nhãn (từ `results/`) → tính Precision (face detection), Accuracy (eye state), Recall (drowsiness, yawn, head-nod), False Positive Rate
  - Xuất báo cáo Markdown/CSV để so với target ở mục 3.3 và 9.2 PRD

- [ ] **L14. requirements.txt — pin version + thêm deps thiếu**
  - Pin cụ thể (VD `flask==3.0.3`) thay vì `>=` để tránh breaking change
  - Thêm: `pytest`, `psutil`, `python-dotenv` (nếu cần config), `cryptography` (cho AES-256 nếu làm API-01 encryption)

- [ ] **L15. Dataset organization**
  - Tạo cấu trúc `data/{face_eye, yawn, head_pose, drowsiness_video}/{train,test}` theo mục 10.4
  - Viết `data/README.md` mô tả annotation guideline (D1–D6 trong PRD)

- [ ] **L16. requirements-dev.txt / CI**
  - Tách dependencies dev (pytest, linter) khỏi production
  - Cấu hình GitHub Actions chạy test khi push (đã có `.github/workflows` — kiểm tra và cập nhật cho phù hợp repo mới)

---

## Gợi ý thứ tự thực hiện
1. L1 (PERCLOS) → L2 (Scoring) → L3 (Alert Manager) — đây là chuỗi phụ thuộc, phải làm theo thứ tự
2. L4 (Event Logger) → L5 (API) → L6 (cập nhật response) — sau khi có L3
3. L12 (test) làm song song mỗi khi hoàn thành 1 module
4. L7–L11 làm sau khi core pipeline ổn định
5. L13–L16 làm cuối, trước khi release
