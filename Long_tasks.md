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
| DMS-06: PERCLOS rolling window 30s | ✅ `src/perclos.py` + fusion | — |
| DMS-11: Microsleep detection | ⚠️ Có nhưng gộp chung logic eye-closure (`eye_microsleep`), chưa tách thành class/module riêng có thể test độc lập | Tách rõ MicrosleepDetector |
| DMS-12/13: Drowsiness Score + phân loại NORMAL/FATIGUE/DROWSY/MICROSLEEP/CRITICAL | ✅ `src/scoring.py` + fusion | — |
| ALT-01→05: Cảnh báo 4 cấp tăng dần + tự hạ cấp | ✅ `src/alert_manager.py` | — |
| DMS-14 / SYS-06: Event Logger | ✅ SQLite metadata; snapshot **opt-in DEBUG** (privacy) | — |
| Privacy: no face images external | ✅ metadata-only default; sync strips snapshots | — |
| Looking away / phone distraction | ✅ L18–L19 | YOLO P1 later |
| Speed + driving time → risk | ✅ L20 `/api/vehicle` | — |
| Alert channels (sound/vibe/break) | ✅ L21 | — |
| Trip fatigue memory | ✅ L22 `/api/trip/summary` | — |
| HITL thresholds + Docker | ✅ L23–L24 | — |
| API-01→05: REST API Event Log, retry, batch upload offline, GPS, encryption | ✅ MVP: `/api/events`, sync mock (no face bytes) | Encryption còn lại |
| SYS-04: Monitor CPU/RAM/GPU/nhiệt độ | ✅ `/api/metrics` + `src/metrics.py` | — |
| SYS-05: Watchdog restart | ✅ `InferenceWatchdog` (stale >5s → reload) | — |
| 9.2 Accuracy: Face Detection >95%, Eye State >90%, Drowsiness Recall >90%, FP <5% | ⚠️ Chưa có script đánh giá định lượng | Viết eval script + report |
| Test coverage (mục 5.1.K) | ✅ PERCLOS/scoring/alert/event/metrics/camera/frequency/multiperson/neck/yawn | — |
| requirements.txt thiếu `pytest`, không pin version | ✅ pin + `requirements-dev.txt` + CI | cryptography/dotenv optional |
| Dataset (mục 10.4) | ✅ cấu trúc `data/` + README guideline | Cần fill dữ liệu thật |

---

## Danh sách Task

### P0 — Bắt buộc cho MVP đúng PRD

- [x] **L1. PERCLOS Module** (`src/perclos.py`)
  - Tính PERCLOS = % thời gian mắt nhắm trong rolling window 30 giây (dùng `deque` lưu (timestamp, is_closed))
  - Input: `ear_smooth` mỗi frame + threshold hiện có (`EYE_CLOSED_THRESH`)
  - Output: `perclos_ratio` (0.0–1.0), đưa vào `FusionState.update()` trả về cùng các field khác
  - Unit test: mô phỏng chuỗi frame nhắm/mở mắt, assert PERCLOS đúng theo tỷ lệ kỳ vọng

- [x] **L2. Drowsiness Scoring Engine + State Machine 5 cấp** (`src/scoring.py`)
  - Input tổng hợp: `p_mlp_drowsy`, `p_lstm_drowsy`, `perclos_ratio`, `eye_closed_streak_ms`, `yawn_alarm`/tần suất ngáp, `neck_alarm`/head-nod frequency, `has_pose` (face visibility)
  - Output trạng thái: `NORMAL | FATIGUE | DROWSY | MICROSLEEP | CRITICAL` (theo mục 6.1 DMS-13 và luồng nghiệp vụ #3)
  - Logic chuyển trạng thái phải dựa trên chuỗi thời gian (không dùng 1 frame), có debounce tương tự cơ chế hysteresis đã có
  - Unit test cho từng transition NORMAL→FATIGUE→DROWSY→MICROSLEEP→CRITICAL và recovery ngược lại

- [x] **L3. Alert Manager đa cấp** (`src/alert_manager.py`)
  - Cảnh báo cấp 1 (FATIGUE) → cấp 2 (DROWSY) → cấp 3 (MICROSLEEP) → cấp 4 (CRITICAL, khi không phục hồi sau cảnh báo — theo US1.2 AC5)
  - Tự động hạ cấp khi tài xế phục hồi (US1.5)
  - Trả `alert_level` (0–4) + `alert_message` qua API để frontend hiển thị màu tương ứng (xanh/vàng/cam/đỏ theo UI-04)
  - Ghi lại mỗi lần đổi cấp vào Event Log (US1.2 AC7)

- [x] **L4. Event Logger (SQLite)** (`src/event_logger.py`)
  - Schema tối thiểu: `id, timestamp, driver_id, vehicle_id, alert_level, ear_avg, perclos, neck_tilt, snapshot_path, gps_lat, gps_lng, uploaded (bool)`
  - **Privacy (outline):** mặc định **metadata-only** — không lưu JPEG; snapshot chỉ khi `SAVE_FACE_SNAPSHOTS=true` (DEBUG)
  - GPS: MVP có thể dùng giá trị giả lập/None nếu chưa có hardware, nhưng field phải tồn tại sẵn (mục A7, API-04)
  - Hàm `log_event()` gọi từ `app.py` mỗi khi `alert_level` thay đổi
  - `/api/events/sync` chỉ gửi metadata (không kèm face bytes)

- [x] **L5. REST API cho Event Log** (bổ sung vào `app.py`)
  - `GET /api/events?driver_id=&date=&limit=` — trả danh sách events (US1.8 AC1/AC2)
  - `GET /api/events/<id>/snapshot` — trả ảnh snapshot
  - `POST /api/events/sync` — mock batch upload lên "cloud" (ghi log, đánh dấu `uploaded=True`) — chuẩn bị cho API-02 (retry/batch upload)

- [x] **L6. Cập nhật `/api/analyze` & `/api/analyze_lite`**
  - Trả thêm: `perclos_ratio`, `drowsiness_state` (NORMAL/FATIGUE/DROWSY/MICROSLEEP/CRITICAL), `alert_level`
  - Đảm bảo không tăng latency quá nhiều (giữ mục tiêu <200ms theo mục 9.1)

### P1 — Quan trọng nhưng có thể làm sau P0

- [x] **L7. System Metrics Endpoint** (`/api/metrics`)
  - CPU%, RAM%, (GPU% nếu có), uptime — dùng `psutil` (thêm vào `requirements.txt`)
  - Phục vụ SYS-04 và mục 9.3 (Reliability)

- [x] **L8. Watchdog nhẹ**
  - Thread giám sát: nếu vòng lặp inference không có kết quả mới trong >5s → log warning + tự động reload model
  - Phục vụ SYS-05

- [x] **L9. Camera Obstruction Detection**
  - Nếu `n_faces == 0` liên tục >10s trong khi trước đó có face → cảnh báo "camera bị che / mất mặt" (A6, R6 trong PRD)
  - Trả `camera_obstructed: bool` trong response

- [x] **L10. Head-Nod Frequency Tracking**
  - Đếm số lần `neck_alarm` trigger trong rolling window (VD 60s) → dùng cho Drowsiness Score (YHP-02 tương tự nhưng cho head-nod)

- [x] **L11. Yawn Frequency Counter**
  - Đếm số lần ngáp (`yawn_alarm` trigger) trong cửa sổ thời gian (YHP-02), trả `yawn_count_window`

### P2 — Testing, Data, Docs (nên làm xuyên suốt)

- [x] **L12. Unit tests bổ sung**
  - [x] `tests/test_perclos.py`, `tests/test_scoring_state_machine.py`, `tests/test_alert_manager.py`, `tests/test_event_logger.py`
  - [x] `tests/test_metrics.py`, `tests/test_camera_and_frequency.py`
  - [x] `tests/test_multiperson_selection.py` (cho `src/pipeline.py`)
  - [x] `tests/test_neck_tilt.py`, `tests/test_yawn.py` (tách riêng khỏi eye-closure)

- [x] **L13. Evaluation Script** (`tools/evaluate.py`)
  - Scaffold: Precision/Recall/Accuracy/FPR từ CSV nhãn → Markdown/CSV report
  - Cần dataset gán nhãn thật để ra số so với mục 3.3 / 9.2

- [x] **L14. requirements.txt — pin version + thêm deps thiếu**
  - [x] Pin version cụ thể (flask/opencv/mediapipe/tf/keras/numpy/pandas/sklearn/joblib/psutil)
  - [x] Thêm: `pytest` (trong `requirements-dev.txt`), `psutil`
  - [ ] `python-dotenv`, `cryptography` (AES-256) — chưa cần cho MVP hiện tại

- [x] **L15. Dataset organization**
  - Tạo cấu trúc `data/{face_eye, yawn, head_pose, drowsiness_video}/{train,test}` theo mục 10.4
  - Viết `data/README.md` mô tả annotation guideline (D1–D6 trong PRD)

- [x] **L16. requirements-dev.txt / CI**
  - [x] Tách `requirements-dev.txt` (pytest)
  - [x] GitHub Actions `.github/workflows/ci.yml` chạy unit tests khi push/PR

---

### P0 — Mở rộng theo project outline (Brief Gap Alignment)

- [x] **L17. Privacy metadata-only** — `SAVE_FACE_SNAPSHOTS=false` mặc định; sync strip snapshot
- [x] **L18. LookingAwayDetector** (`src/looking_away.py`) — `|yaw|` + duration → `looking_away`
- [x] **L19. Phone distraction heuristic** (`src/phone_distraction.py`) — hand-near-face; YOLO later
- [x] **L20. Vehicle context + CAN mock** (`src/context.py`, `tools/can_sim.py`, `/api/vehicle`)
- [x] **L21. Alert channels** — `sound` / `vibration` / `break_suggested` trong AlertManager
- [x] **L22. Trip fatigue memory** (`src/trip_memory.py`, `GET /api/trip/summary`)

### P1 — Platform

- [x] **L23. HITL thresholds** — `GET/PUT /api/thresholds`
- [x] **L24. Docker** — `Dockerfile` + `docker-compose.yml`

## Gợi ý thứ tự thực hiện
1. L1→L6 core PRD (xong) → L7–L16 polish (xong)
2. L17 privacy → L18 looking-away → L19 phone → L20 context → L21 channels → L22 trip
3. L23–L24 HITL + Docker
