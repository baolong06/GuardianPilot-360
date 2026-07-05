# GuardianPilot 360 — Hướng dẫn chạy dự án

**Hệ thống giám sát trạng thái người lái (Driver Monitoring System)** dùng kiến trúc Multi-Agent với 4 model AI độc lập, tổng hợp qua Knowledge Graph và Orchestrator.

---

## Mục lục

1. [Tổng quan hệ thống](#1-tổng-quan-hệ-thống)
2. [Cấu trúc thư mục](#2-cấu-trúc-thư-mục)
3. [Yêu cầu cần có](#3-yêu-cầu-cần-có)
4. [Cài đặt và chạy](#4-cài-đặt-và-chạy)
5. [Các lệnh run.py chi tiết](#5-các-lệnh-runpy-chi-tiết)
6. [Đọc kết quả đầu ra](#6-đọc-kết-quả-đầu-ra)
7. [Chạy unit tests](#7-chạy-unit-tests)
8. [Xử lý lỗi thường gặp](#8-xử-lý-lỗi-thường-gặp)
9. [Kiến trúc tóm tắt](#9-kiến-trúc-tóm-tắt)

---

## 1. Tổng quan hệ thống

| Model | File | Đầu vào | Đầu ra |
|-------|------|---------|--------|
| **M1 — Drowsiness** | `task_1/dcnn_drowsiness_task1_baseline.keras` | Ảnh mặt 96×96 | P(Buồn ngủ) |
| **M2 — Microsleep** | `Task 2.../cnn_16s_best.keras` | Tín hiệu EEG/EOG 16s | 4-class: Wake/MSE/MSEc/ED |
| **M3 — Distracted** | `Task_3/dbmnet_full_task3.keras` | Ảnh cabin 224×224 | 10 hành vi mất tập trung |
| **M4 — Landmark** | `Task_4/lstm_landmark_task4_fixed.keras` | 15 frame × 1440-dim landmark | Alert / Drowsy |

Bốn model chạy song song → ghi kết quả vào **Knowledge Graph** → **Orchestrator** áp 7 luật ưu tiên → ra **1 mức cảnh báo duy nhất**.

```
Camera → [M1] [M3] [M4]  ──┐
EEG    → [M2]             ──┤→ Knowledge Graph → Orchestrator → Actuation
                            ┘
```

---

## 2. Cấu trúc thư mục

```
Model/
├── requirements.txt
├── run.py                        # Entry point Python
│
├── guardian_pilot/               # Source code chính
│   ├── core/
│   │   ├── schema.py             # Enums + dataclasses dùng chung
│   │   └── knowledge_graph.py    # Blackboard in-memory (networkx)
│   ├── agents/
│   │   ├── base_agent.py         # Abstract PerceptionAgent
│   │   ├── m1_drowsiness.py      # Agent M1
│   │   ├── m2_microsleep.py      # Agent M2
│   │   ├── m3_distracted.py      # Agent M3 + fallback
│   │   ├── m4_landmark.py        # Agent M4 LSTM/MLP
│   │   └── orchestrator.py       # 7 luật ưu tiên
│   ├── dispatcher.py             # ThreadPoolExecutor frame routing
│   ├── actuation.py              # Console output + audit log
│   └── system.py                 # Facade khởi tạo toàn hệ thống
│
├── tests/
│   ├── test_knowledge_graph.py   # 14 unit tests
│   └── test_orchestrator_rules.py   # 9 unit tests (7 luật)
│
├── task_1/                       # Model M1 (56MB)
├── Task 2 — .../                 # Model M2 (6.8MB)
├── Task_3/                       # Model M3 (18.5MB)
├── Task_4/                       # Model M4 + scaler + landmarker
│
├── data/                         # Đặt file video vào đây
└── logs/                         # Audit log output
```

---

## 3. Yêu cầu cần có

- Python **3.10 hoặc 3.11** (khuyến nghị 3.11 — MediaPipe chưa hỗ trợ 3.12+)
- RAM tối thiểu **8GB** (TensorFlow cần ~4GB khi load tất cả model)
- GPU NVIDIA + CUDA 11.8+ *(tùy chọn — tăng tốc đáng kể)*

---

## 4. Cài đặt và chạy

### Bước 1: Tạo virtual environment

```bash
# Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### Bước 2: Cài thư viện

```bash
pip install -r requirements.txt
```

> ⚠️ TensorFlow ~500MB, MediaPipe ~50MB — lần đầu mất 5–10 phút tuỳ mạng.

### Bước 3: Kiểm tra cài đặt (không cần model AI, không cần GPU)

```bash
python -m pytest tests/ -v
```

Kết quả đúng: `23 passed`

### Bước 4: Chạy hệ thống

```bash
# Trên file video
python run.py --video data/input.mp4

# Webcam real-time
python run.py --camera 0

# Tuỳ chỉnh FPS
python run.py --video data/input.mp4 --fps 20

# Headless (không hiển thị cửa sổ)
python run.py --video data/input.mp4 --no-display
```

---

## 5. Các lệnh run.py chi tiết

```
python run.py [OPTIONS]

Options:
  --video PATH        Đường dẫn file video để test offline
                      (nếu không đặt → dùng webcam)
  --camera INDEX      Camera index, mặc định 0
  --fps FLOAT         Số frame/giây xử lý, mặc định 15.0
  --eeg               Bật Agent M2 (cần sensor EEG/EOG thật)
  --no-display        Tắt cửa sổ OpenCV (chạy headless)
  --audit-log PATH    Đường dẫn file audit log JSONL
                      mặc định: guardian_pilot_audit.log
```

### Ví dụ thực tế

```bash
# Test nhanh với video 10 fps, không hiển thị
python run.py --video data/test.mp4 --fps 10 --no-display

# Webcam + bật sensor EEG (nếu đã cắm hardware)
python run.py --camera 0 --eeg --fps 15

# Headless server mode, log ra file riêng
python run.py --video data/input.mp4 --no-display --audit-log logs/run_audit.log
```

---

## 6. Đọc kết quả đầu ra

### Console output

Khi alert level thay đổi, hệ thống in ra:

```
============================================================
✅  ALERT: NORMAL
  Reason:  Tất cả chỉ số bình thường.
  Conf:    0.85
  Active:  ['M1_Drowsiness', 'M3_Distracted', 'M4_LandmarkGaze']
  Actions: ['system_idle']
============================================================

============================================================
⚠️   ALERT: MILD_WARNING
  Reason:  M1: Cảnh báo DROWSY đơn lẻ (conf=0.73). Chờ xác nhận 2 giây.
  Conf:    0.51
  Active:  ['M1_Drowsiness', 'M3_Distracted', 'M4_LandmarkGaze']
  Actions: ['beep_soft_1x', 'dashboard_amber']
============================================================
```

### 4 mức cảnh báo

| Màu | Mức | Hành động |
|-----|-----|-----------|
| 🟢 Xanh | `NORMAL` | Không có |
| 🟡 Vàng | `MILD_WARNING` | Beep nhẹ 1 lần, đèn vàng |
| 🔴 Đỏ | `SEVERE_WARNING` | Beep to 3 lần, rung ghế, đèn đỏ |
| 🟣 Tím | `EMERGENCY` | Còi liên tục, rung mạnh, đèn đỏ nhấp nháy, gửi telemetry |

> `EMERGENCY` chỉ trigger khi Agent M2 (EEG) phát hiện `PATHOLOGICAL_PROXY`
> với confidence > 0.6. Nếu không có sensor EEG → M2 tự tắt, `EMERGENCY` không bao giờ xảy ra.

### Audit log

Log được ghi vào `logs/guardian_pilot_audit.log`, mỗi dòng là 1 JSON event:

```json
{"ts": "2026-07-05T10:00:00Z", "alert": "MILD_WARNING", "reason": "M1: Drowsy conf=0.72", "confidence": 0.504, "agents": ["M1_Drowsiness", "M4_LandmarkGaze"], "actions": ["beep_soft_1x", "dashboard_amber"]}
```

---

## 7. Chạy unit tests

Tests **không cần model AI** — chạy nhanh, không cần GPU.

```bash
# Chạy tất cả tests
python -m pytest tests/ -v

# Chỉ test Knowledge Graph
python -m pytest tests/test_knowledge_graph.py -v

# Chỉ test 7 luật Orchestrator
python -m pytest tests/test_orchestrator_rules.py -v

# Test kèm output chi tiết khi fail
python -m pytest tests/ -v --tb=long
```

### Kết quả đúng

```
============================== 23 passed in 1.82s ==============================
```

---

## 8. Xử lý lỗi thường gặp

### ❌ `ModuleNotFoundError: No module named 'cv2'`
```bash
pip install opencv-python-headless
```

### ❌ `FileNotFoundError: landmark_scaler_task4.pkl`
Agent M4 yêu cầu file scaler tồn tại trước khi khởi động.
Kiểm tra thư mục `Task_4/`:
```bash
ls Task_4/
# Phải có: landmark_scaler_task4.pkl
```

### ❌ `No module named 'mediapipe'`
```bash
pip install mediapipe>=0.10.0
```
> MediaPipe không hỗ trợ Python 3.12+ — dùng **Python 3.11**.

### ❌ Agent M2 luôn offline
Đây là **hành vi đúng** khi không có sensor EEG/EOG.
Hệ thống tiếp tục chạy với M1, M3, M4. Chỉ dùng `--eeg` khi có hardware thật.

### ❌ `SEVERE_WARNING` ngay khi khởi động
Thường do model chưa load xong mà frame đã tới. Chờ vài giây để tất cả model load.
Log sẽ in `✓ Model loaded:` cho từng model.

### ❌ Video chạy quá chậm
Giảm FPS xuống:
```bash
python run.py --video data/input.mp4 --fps 5
```
Hoặc bật GPU (cần cài CUDA + TF-GPU).

---

## 9. Kiến trúc tóm tắt

```
┌──────────────────────────────────────────────────────────┐
│                   NGUỒN DỮ LIỆU                          │
│  Camera RGB ~30fps              Sensor EEG/EOG (tuỳ chọn)│
└──────────┬──────────────────────────────┬────────────────┘
           │                              │
           ▼                              ▼
┌─────────────────────┐        ┌──────────────────┐
│   Frame Dispatcher   │        │  Physio Buffer   │
│  (ThreadPoolExecutor)│        │  (M2 only)        │
└──┬───────┬───────┬──┘        └────────┬─────────┘
   │       │       │                    │
   ▼       ▼       ▼                    ▼
 [M1]    [M3]    [M4]                [M2]
 DCNN   DBMNet   LSTM              CNN_16s
           │ fallback                   │
        [baseline]               (offline nếu
                                  không có sensor)
   │       │       │                    │
   └───────┴───────┴────────────────────┘
                   │
                   ▼
        ┌─────────────────────┐
        │   Knowledge Graph    │  ← Blackboard dùng chung
        │  (networkx, thread-  │    (30 frame history/agent)
        │   safe RLock)        │
        └──────────┬──────────┘
                   │
                   ▼
        ┌─────────────────────┐
        │  Orchestrator Agent  │  ← 7 Priority Rules
        │  (không inference)   │    chỉ đọc KG + quyết định
        └──────────┬──────────┘
                   │
                   ▼
        ┌─────────────────────┐
        │   Actuation Layer    │  ← Console + Audit JSONL
        │  (MVP: console/log)  │    Production: CAN bus, loa, LED
        └─────────────────────┘
```

### 7 luật Orchestrator (thứ tự ưu tiên cao → thấp)

| # | Điều kiện | Kết quả |
|---|-----------|---------|
| R1 | M2 = PATHOLOGICAL + conf > 0.6 | **EMERGENCY** |
| R2 | M1 + M4 cùng DROWSY, conf > 0.5 | **SEVERE_WARNING** |
| R3 | Chỉ 1 agent DROWSY, conf > 0.6 | **MILD_WARNING** + chờ 2s |
| R4 | M1 vs M4 mâu thuẫn | **MILD_WARNING** + log ConflictEvent |
| R5 | M3 DISTRACTED, conf > 0.6 | max(current, **MILD_WARNING**) |
| R6 | Bất kỳ agent offline | Giảm confidence × 0.85 |
| R7 | Mặc định | **NORMAL** |

---

## Liên hệ & Tài liệu thêm

- Kiến trúc chi tiết: [`GuardianPilot360_AgentArchitecture.md`](./GuardianPilot360_AgentArchitecture.md)
- Source code agents: `guardian_pilot/agents/`
- Unit tests: `tests/`
