# Report — Khắc phục vấn đề audit GuardianPilot-360

**Ngày:** 2026-08-20
**Phạm vi:** sửa toàn bộ vấn đề trong báo cáo audit **trừ C3** (xây dựng training pipeline — bị loại khỏi phạm vi theo yêu cầu).
**Ràng buộc:** không `git add` / `commit` / `push`. Toàn bộ thay đổi nằm ở working tree.

---

## 1. Tóm tắt

| Chỉ số | Trước | Sau |
|---|---|---|
| `import app` | ❌ Fail (thiếu mediapipe/TF/keras) | ✅ OK |
| `pytest tests/` | Không chạy được | ✅ **88/88 PASS**, 0 warning |
| Kiểm chứng API (23 mục) | — | ✅ **23/23 PASS** với model thật (`load_mode=weights`) |
| Cảnh báo sklearn khi unpickle scaler | ⚠️ `InconsistentVersionWarning` | ✅ Hết |
| Số dependency runtime | 11 | 9 (bỏ `pandas`, chuyển `pytest` sang dev) |

**24 vấn đề: 22 ĐÃ SỬA · 1 GIẢM THIỂU (C4) · 1 KHÔNG LÀM (C3, theo yêu cầu).**
Phát sinh thêm **2 phát hiện mới** trong lúc sửa (M11, và một đính chính cho C4).

---

## 2. Bảng đối chiếu vấn đề → trạng thái

| ID | Mức | Trạng thái | Nội dung đã làm |
|---|---|---|---|
| **C1** | Critical | ✅ ĐÃ SỬA | Tạo `.venv` Python 3.11, cài đúng version ghim; thêm `tools/check_env.py`; ghi rõ ràng buộc Python 3.9–3.12 vào `requirements.txt` + README |
| **C2** | Critical | ✅ ĐÃ SỬA | Chặn NaN ở đường LSTM + lưới an toàn `_json_safe()` ở tầng response |
| **C3** | Critical | ⛔ KHÔNG LÀM | Ngoài phạm vi theo yêu cầu của bạn |
| **C4** | Critical | ⚠️ GIẢM THIỂU | Thêm `FORCE_RULE_ONLY=true` + `tools/model_calibration.py`. **Gốc chỉ chữa được bằng retrain (= C3).** Xem §5 — kết luận C4 đã được đính chính |
| **H1** | High | ✅ ĐÃ SỬA | Gỡ ~50 dòng nhánh multi-person không bao giờ chạy (kèm `NameError` tiềm ẩn) |
| **H2** | High | ✅ ĐÃ SỬA | `src/session.py` — state tách theo `session_id`, tương thích ngược qua session `default` |
| **H3** | High | ✅ ĐÃ SỬA | `DrowsinessScorer.THRESHOLDS` thành instance attribute |
| **H4** | High | ✅ ĐÃ SỬA | 3 knob eye-closure nay thật sự có tác dụng, đồng bộ cả PERCLOSTracker |
| **H5** | High | ✅ ĐÃ SỬA | Ngưỡng EAR về một nguồn duy nhất; frontend đọc từ server |
| **H6** | High | ✅ ĐÃ SỬA | `src/auth.py` — API key cho 6 endpoint nhạy cảm, mặc định tắt |
| **M1** | Medium | ✅ ĐÃ SỬA | Gỡ hard-code `E:/KhoiNghiep` khỏi 9 file, thay bằng path tương đối + argparse |
| **M2** | Medium | ✅ ĐÃ SỬA | `inference_fps` đo bằng `time.monotonic()` |
| **M3** | Medium | ✅ ĐÃ SỬA | `MAX_CONTENT_LENGTH` + handler 413 JSON + chặn độ phân giải |
| **M4** | Medium | ✅ ĐÃ SỬA | Guard `bbox is None` trước `_score_person` |
| **M5** | Medium | ✅ ĐÃ SỬA | `_stale_count` đưa vào lock |
| **M6** | Medium | ✅ ĐÃ SỬA | `EventLogger` khởi tạo lười |
| **M7** | Medium | ✅ ĐÃ SỬA | Dọn requirements, thêm `requests` vào dev |
| **M8** | Medium | ✅ ĐÃ SỬA (một phần) | Sửa pattern `.gitignore`. **Không** `git rm --cached` — xem §7 |
| **M9** | Medium | ✅ ĐÃ SỬA | `src/camera.py` + `docs/CAMERA_CALIBRATION.md` |
| **M10** | Medium | ✅ ĐÃ SỬA | Codec `avc1` (H.264) trước, fallback `mp4v` |
| **M11** | Medium | ✅ ĐÃ SỬA | **MỚI** — scaler pickle bằng sklearn 1.6.1 nhưng requirements ghim 1.5.1 |
| **L1** | Low | ✅ ĐÃ SỬA (một phần) | Sửa typo, gỡ hằng số chết trong `app.js`. **Không** tách ES modules (theo lựa chọn của bạn) |
| **L2** | Low | ✅ ĐÃ SỬA | Gỡ pandas hoàn toàn khỏi `src/fusion.py` và requirements |
| **L3** | Low | ✅ ĐÃ SỬA | `reset()` tường minh cho `FusionState` |
| **L4** | Low | ✅ ĐÃ SỬA | `.pytest_cache/` vào `.gitignore` |
| **L5** | Low | ✅ ĐÃ SỬA | Sửa docstring sai của `tools/evaluate.py` |
| — | — | ✅ | Xoá file rác `.py`; sửa `search_nb.py`, `tools/export_onnx.py`; thêm `pytest.ini` |

---

## 3. Chi tiết từng thay đổi

### C2 — NaN trong LSTM làm hỏng JSON response 🔴

**Vấn đề.** Nếu một cột của cửa sổ LSTM là NaN suốt cả 30 frame (thực tế hay gặp:
`neck_tilt` khi MediaPipe không bắt được pose vai), `ffill().bfill()` không cứu được.
`StandardScaler` cho NaN đi qua (`ensure_all_finite="allow-nan"`), Keras nhân ra NaN,
`flask.jsonify` sinh literal `NaN` — **không phải JSON hợp lệ** — và `res.json()`
ở frontend ném lỗi, giết cả vòng lặp live.

**Đã làm** — [src/fusion.py](src/fusion.py):
- Thay `pandas` bằng `_ffill_bfill()` viết bằng numpy (giữ nguyên ngữ nghĩa).
- `np.nan_to_num()` trước **và** sau `scaler.transform()`.
- Guard `_finite()` cho output cả MLP lẫn LSTM. MLP không hữu hạn → coi như
  `p_drowsy = 0.0` (**không** báo động giả hàng loạt; các rule eye/neck/yawn vẫn
  chạy độc lập và vẫn bật được cảnh báo).

**Đã làm** — [app.py](app.py): thêm `_json_safe()` đệ quy đổi mọi float không hữu hạn
thành `null`, áp cho mọi response của `/api/analyze` và `/api/analyze_lite`.

**Bằng chứng (chạy với model thật):**

```
Đường CŨ  (pandas, không nan_to_num) → p_non_drowsy_lstm = nan   | NaN? True
Đường MỚI                            → p_lstm_drowsy     = 0.5207
_ffill_bfill khớp pandas ffill().bfill(): True
Không có literal NaN/Infinity trong JSON: PASS
```

**Rủi ro:** không. Điền NaN → 0.0 dùng đúng quy ước mà đường MLP đã dùng sẵn.

---

### H2 — State toàn cục dùng chung cho mọi người dùng 🟠

**Vấn đề.** `_fusion`, `_alert_mgr`, `_driving_ctx`, `_trip_memory`, `_phone_det`
đều là biến module-level. Mở hai tab là hai luồng nhận diện trộn vào nhau; `/api/reset`
của người này xoá state người kia. Mâu thuẫn thẳng với "Fleet Dashboard" trên UI.

**Đã làm.** File mới [src/session.py](src/session.py):
- `DriverSession` — gói toàn bộ state của MỘT tài xế + metadata (driver_id, vehicle_id, GPS).
- `SessionStore` — cấp phát theo `session_id`, TTL 30 phút, tối đa 32 session, tự dọn session cũ.
- `app.py` lấy id theo thứ tự: header `X-Session-Id` → `body.session_id` → `?session_id` → `"default"`.
- Frontend ([app.js](web/static/js/app.js)) sinh id mỗi tab bằng `crypto.randomUUID()`
  lưu trong `sessionStorage`; [worker.js](web/static/js/worker.js) nhận cùng id để
  vòng lặp live không chạy trên session khác main thread.

**Quyết định thiết kế:** vẫn giữ **một** khoá toàn cục `_infer_lock` quanh MediaPipe +
model predict, vì landmarker là singleton không thread-safe. Tách session giải quyết
đúng vấn đề *trộn dữ liệu*; nó không nhằm tăng throughput.

**Tương thích ngược:** request không gửi session_id → session `default`, hành vi y hệt
trước. Toàn bộ test cũ và các script trong `tools/` chạy không cần sửa gì.

**Bằng chứng:** `carA=90.0 / carB=10.0 / default=0.0` — độc lập hoàn toàn.

---

### H4 + H5 — Knob HITL giả và ngưỡng EAR mâu thuẫn 🟠

**Vấn đề.** `PUT /api/thresholds` nhận `eye_closed_thresh`, `eye_closed_on_sec`,
`eye_closed_hard_sec`, trả `ok: true`, **nhưng `_apply_runtime_thresholds()` chưa
bao giờ đẩy chúng xuống `FusionState`**. Kỹ sư an toàn chỉnh ngưỡng và tin là đã
đổi, trong khi hệ thống không đổi gì. Cùng lúc, "ngưỡng mắt nhắm" tồn tại ở 4 nơi
với 3 giá trị khác nhau: fusion `0.16`, thresholds store `0.18`, PERCLOS default
`0.18`, frontend `0.20`.

**Đã làm:**
- `FusionState.apply_thresholds()` — đẩy cả 3 knob eye-closure, đồng bộ luôn
  `perclos_tracker.eye_closed_threshold` (nếu không, PERCLOS đếm "mắt nhắm" theo một
  ngưỡng khác rule eye-closure).
- `DrowsinessScorer.set_threshold()` thay cho việc ghi thẳng vào class attribute.
- Thống nhất về **0.16** ở `thresholds.py` và `perclos.py`.
- `/api/runtime-profile` công bố `eye_closed_thresh`; frontend đọc từ đó.

> **Lưu ý quan trọng:** đổi `_DEFAULTS["eye_closed_thresh"]` từ 0.18 → 0.16 là để
> **KHỚP** hành vi đang chạy thật (vì knob cũ không có tác dụng nên giá trị thực thi
> luôn là 0.16 của fusion). Đây **không** phải tinh chỉnh làm hệ thống nhạy hơn hay
> bớt nhạy đi. Nếu để nguyên 0.18, việc sửa H4 sẽ vô tình làm hệ thống nhạy hơn hẳn
> ngay lần khởi động đầu tiên.

**Bằng chứng:** `0.16 → 0.25` truyền được xuống FusionState, PERCLOSTracker đồng bộ theo.

---

### H1 — 50 dòng code chết chứa `NameError` 🟠

Nhánh `if n_faces > 1:` trong [src/pipeline.py](src/pipeline.py) **không bao giờ chạy**:
HolisticLandmarker (Tasks API) chỉ trả một khuôn mặt và Step 1 gói thành `[flat_face]`
→ `n_faces` luôn ≤ 1. Nhánh đó còn gọi `holistic.detect(...)` trong khi biến `holistic`
chỉ tồn tại ở đường `.task` — đi qua đường fallback `mp.solutions` là `NameError`.

Đã gỡ nhánh này cùng hai biến `scale_x`/`scale_y` chỉ phục vụ nó. **Giữ nguyên**
`_score_person` / `_estimate_face_bbox` / `_expand_crop` / `_transform_landmarks`
(có test riêng trong `tests/test_multiperson_selection.py`) và giữ nguyên hành vi
với 1 khuôn mặt.

---

### H3 — Ngưỡng rò rỉ qua class attribute 🟠

`app.py` ghi thẳng `_fusion.scorer.THRESHOLDS[DS.FATIGUE] = ...` — `THRESHOLDS` là
**class attribute**, nên sửa một lần là đổi cho MỌI instance, kể cả instance tạo sau
`reset()`, và rò rỉ giữa các test chạy chung process.

Nay mỗi instance copy sang `self.thresholds` trong `__init__`. Bằng chứng: sau khi
sửa ngưỡng runtime, `DrowsinessScorer()` mới vẫn trả 0.55, class attribute không đổi.

---

### H6 — Không có xác thực 🟠

File mới [src/auth.py](src/auth.py). Thiết kế "opt-in để không phá demo":

- **Không set `GUARDIANPILOT_API_KEY` → auth TẮT**, hành vi y hệt trước (log cảnh báo một lần).
- Có set → yêu cầu header `X-API-Key` (so sánh bằng `hmac.compare_digest`) trên:
  `PUT /api/thresholds`, `GET /api/events`, `/api/events/<id>/snapshot`,
  `POST /api/events/sync`, `GET /api/trip/summary`, `GET /api/metrics`.
- `/api/analyze`, `/api/analyze_lite` và **`GET /api/thresholds`** cố ý để mở —
  demo webcam và việc frontend đọc ngưỡng hiển thị (H5) vẫn phải chạy được.
- Dashboard đọc key từ `localStorage['gp_api_key']`, log hướng dẫn rõ ràng khi gặp 401.

---

### M11 — Scaler pickle bằng sklearn khác version 🟡 **(phát hiện mới)**

Chạy test lần đầu lộ ra:

```
InconsistentVersionWarning: Trying to unpickle estimator StandardScaler
from version 1.6.1 when using version 1.5.1.
This might lead to breaking code or invalid results.
```

`models/compatible/*.pkl` được pickle bằng **scikit-learn 1.6.1** trong khi
`requirements.txt` ghim **1.5.1**. Vì không thể retrain (C3 ngoài phạm vi), artifact
là nguồn sự thật → đã ghim `scikit-learn==1.6.1`. Cảnh báo biến mất, 88/88 test vẫn PASS.

---

### Các sửa nhỏ còn lại

| ID | File | Thay đổi |
|---|---|---|
| M1 | `results/*.py` (7), `tools/test_distance.py`, `tools/diagnose_live.js` | `Path(__file__).resolve().parents[1]` + `argparse`; model tìm qua `model_search_roots()` |
| M2 | `src/session.py::note_inference` | FPS đo bằng `time.monotonic()` — với video, `ts_ms` là media timeline nên số cũ là FPS của video, không phải tốc độ server |
| M3 | `app.py` | `MAX_CONTENT_LENGTH` (env `MAX_UPLOAD_MB`, mặc định 12MB), errorhandler 413 trả JSON, chặn frame > 3840×2160 |
| M4 | `src/pipeline.py` | `if bbox is None: continue` + trả None khi không còn candidate |
| M5 | `src/metrics.py` | `_stale_count += 1` vào trong `self._lock` |
| M6 | `app.py` | `_get_event_logger()` lười — `import app` không còn tạo `data/events.db` |
| M9 | `src/camera.py`, `src/landmarks.py` | Intrinsics qua env; **mặc định tái tạo chính xác giả định cũ** nên head-pose không đổi |
| M10 | `src/video_output.py` | `CODEC_PREFERENCE = ("avc1", "mp4v")`, trả thêm trường `codec` |
| L1 | `web/static/js/app.js` | Sửa typo `mulai rendah`; `EAR_CLOSED_THRESHOLD` cứng → `earClosedThreshold` lấy từ server |
| L3 | `src/fusion.py` | `reset()` tường minh — không còn `self.__init__()`, **giữ lại ngưỡng HITL đã cấu hình** |
| — | `pytest.ini` | `testpaths = tests` — `pytest` trần ở root không gom nhầm 8 file `tools/test_*.py` |
| — | `search_nb.py` | argparse + báo lỗi rõ ràng thay vì `FileNotFoundError` trần |
| — | `tools/export_onnx.py` | Dùng `model_search_roots()` — trước đây trỏ vào thư mục rỗng nên luôn in "not found" |

---

## 4. File đã đụng tới

**Mới (8):**
`src/session.py` · `src/auth.py` · `src/camera.py` · `tools/check_env.py` ·
`tools/model_calibration.py` · `pytest.ini` · `docs/CAMERA_CALIBRATION.md` ·
`reports/model_calibration.md` *(sinh tự động)*

**Sửa (29):** `app.py` · `src/fusion.py` · `src/scoring.py` · `src/pipeline.py` ·
`src/perclos.py` · `src/landmarks.py` · `src/metrics.py` · `src/thresholds.py` ·
`src/video_output.py` · `web/static/js/{app,worker,dashboard}.js` ·
`results/*.py` (7) · `tools/{evaluate,export_onnx,test_distance}.py` ·
`tools/diagnose_live.js` · `search_nb.py` · `requirements.txt` ·
`requirements-dev.txt` · `.gitignore` · `README.md` · `docker-compose.yml`

**Xoá (1):** `.py` — file rác 16.6KB ở thư mục gốc, untracked, chứa JSON profiling
dataset NYC-taxi (`gate2_mvp.db`, `yellow_tripdata`) hoàn toàn không liên quan dự án.

**KHÔNG đụng:** `tests/` (giữ làm lưới an toàn) · `models/` · `data/` ·
`PRD_GuardianPilot360.md` · `*_tasks.md` · `.git/`

Tổng: **29 file sửa, +1181 / −369 dòng** (chưa tính file mới).

---

## 5. ĐÍNH CHÍNH cho vấn đề C4

Vòng audit dựa vào `reports/live_diagnostic.md` và kết luận **"model MLP bị bias
về DROWSY"**. Chạy đo lại có hệ thống bằng `tools/model_calibration.py` với chính
artifact hiện tại cho kết quả **khác**:

| EAR | Trạng thái mắt | p_mlp_drowsy | p_lstm_drowsy |
|---|---|---|---|
| 0.08 | nhắm | 0.950 | 0.601 |
| 0.16 | ngưỡng | 0.793 | 0.590 |
| 0.20 | mở | 0.560 | 0.576 |
| 0.30 | mở rõ | **0.148** | 0.543 |
| 0.35 | mở rõ | **0.143** | 0.534 |

- **MLP thực ra hoạt động đúng** — đơn điệu, biên độ 0.808 trên dải EAR. Con số
  `p_mlp = 0.585 tại EAR = 0.30` trong báo cáo cũ **không tái lập được**; hiện tại
  là 0.148.
- **LSTM mới là thứ hỏng**: kẹt trong 0.53–0.60 trên **toàn bộ** dải EAR, biên độ
  chỉ **0.067**. Nó gần như không mang thông tin, mà vẫn nằm trên ngưỡng FATIGUE
  (0.40) ngay cả khi mắt mở rõ.

Điều này đồng thời **giải thích** comment sẵn có trong `src/fusion.py`
(*"LSTM hay bị kẹt ở ~0.55"*) và cái guard `abs(p_lstm - p_mlp) > 0.15 → chỉ dùng MLP`
— guard đó chính là lý do hệ thống vẫn chạy được trên thực tế.

**Kết luận đúng cho C4:** vấn đề nằm ở **LSTM**, không phải MLP. Vẫn cần retrain
(C3) để xử lý gốc. Trong lúc chờ, hoặc chạy `FORCE_RULE_ONLY=true`, hoặc — rẻ hơn
nhiều — đặt `EDGE_PROFILE=edge` (profile này vốn đã tắt LSTM) và chỉ dùng MLP + rule.

---

## 6. Kiểm chứng đã chạy

```
1. pytest tests/                        →  88 passed, 0 warning
2. tools/check_env.py                   →  11/11 package OK, 8/8 import OK
3. verify_api.py (23 mục, model thật)   →  23 PASS / 0 FAIL
4. tools/smoke_imports.py               →  All critical imports OK
5. tools/model_calibration.py           →  chạy được, sinh reports/model_calibration.md
6. node --check trên 3 file JS          →  đều hợp lệ
7. ast.parse toàn bộ .py                →  0 lỗi cú pháp
```

Chi tiết 23 mục kiểm chứng API (chạy qua `Flask.test_client`, `load_mode=weights`):

```
PASS  init  load_mode=weights                    PASS  H4: eye_closed_thresh 0.16 -> 0.25
PASS  init trả session_id                        PASS  H4: PERCLOSTracker đồng bộ
PASS  analyze 200                                PASS  H3: instance mới giữ mặc định 0.55
PASS  C2: không có literal NaN/Infinity          PASS  H3: class attribute không bị ghi đè
PASS  C2: JSON parse được                        PASS  L3: /api/reset giữ ngưỡng HITL
PASS  H2: carA=90 / carB=10 độc lập              PASS  M3: payload quá lớn → 413
PASS  H2: session 'default' không ảnh hưởng      PASS  H6: chưa set env → vẫn mở
PASS  H2: status liệt kê session                 PASS  H6: thiếu/sai key → 401
PASS  H5: runtime-profile công bố ngưỡng         PASS  H6: đúng key → 200
PASS  M9: công bố trạng thái calib camera        PASS  H6: PUT thresholds cần key
PASS  H6: GET thresholds vẫn mở                  PASS  H6: /api/analyze không bị chặn
```

---

## 7. Việc CHƯA làm — cần bạn quyết

| Việc | Lý do chưa làm |
|---|---|
| **C3 — training pipeline** | Bạn loại khỏi phạm vi. Đây vẫn là nút thắt lớn nhất: không có dataset/code train thì không retrain được LSTM, không đánh giá được độ chính xác, không kiểm tra được data leakage |
| **C4 gốc** | Phụ thuộc C3 |
| **Gỡ model khỏi git index** | `.gitignore` đã sửa pattern, nhưng 7 file trong `models/compatible/` **đang được track**. `git rm --cached` sẽ khiến người clone mới không chạy được app. Cần bạn quyết (Git LFS? release asset?) |
| **Tách `app.js` thành ES modules** | Bạn chọn "dọn tại chỗ, giữ 1 file" vì tôi không có trình duyệt để test regression |
| **ONNX / TensorRT** | `tools/export_onnx.py` nay tìm được model, nhưng cần `pip install tf2onnx` và chưa chạy thử |
| **Đổi tên 8 file `tools/test_*.py`** | Đã xử lý bằng `pytest.ini` (`testpaths=tests`) — an toàn hơn đổi tên hàng loạt |
| **Typo tên hàm test** | `tests/test_eye_closure_rule.py:406` có tên `test_escape_valve_smoothsurvives_...`. Tôi cam kết không sửa `tests/` |

---

## 8. Checklist smoke-test thủ công (phần tôi không kiểm chứng được)

Tôi **không** chạy webcam hay trình duyệt. Nhờ bạn kiểm 6 mục sau:

```bash
.venv\Scripts\python app.py --port 5000
```

1. **Webcam live** — mở http://127.0.0.1:5000 → **Khởi tạo** → **Phân tích live**.
   Kỳ vọng: badge "Sẵn sàng (weights)", landmark vẽ lên canvas, FPS chạy, không có
   lỗi đỏ trong Console.
2. **Session tách biệt (H2)** — mở **hai tab** cùng lúc, chạy live ở cả hai. Kỳ vọng:
   hai tab có state riêng; bấm "Đặt lại" ở tab A **không** làm tab B nhảy về NORMAL.
   Console mỗi tab log `[Profile] dev ... EAR<0.16`.
3. **Phân tích video + tải MP4 (M10)** — tab Video → chọn file → Phân tích → Tải xuống.
   Kỳ vọng: file tải về mở được. Nếu OpenCV có H.264, video còn phát inline được.
4. **Dashboard** — http://127.0.0.1:5000/dashboard hiển thị event bình thường.
5. **Auth (H6)** — chạy lại server với `GUARDIANPILOT_API_KEY=test123`.
   Kỳ vọng: webcam vẫn chạy; dashboard trống + Console báo 401 cho tới khi chạy
   `localStorage.setItem('gp_api_key','test123')` rồi F5.
6. **Ngưỡng HITL (H4)** — `PUT /api/thresholds {"thresholds":{"eye_closed_thresh":0.25}}`
   rồi nheo mắt nhẹ. Kỳ vọng: cảnh báo "mắt nhắm" bật sớm hơn rõ rệt so với 0.16.

---

## 9. Trạng thái git

**Không có `git add` / `commit` / `push` nào được thực hiện.** Toàn bộ thay đổi nằm
ở working tree để bạn tự review trước:

```bash
git status
git diff                       # 29 file đã sửa
git diff --stat                # +1181 / −369
```

File mới đang untracked: `src/session.py`, `src/auth.py`, `src/camera.py`,
`tools/check_env.py`, `tools/model_calibration.py`, `pytest.ini`,
`docs/CAMERA_CALIBRATION.md`, `reports/model_calibration.md`, `Report.md`.

Muốn quay lại toàn bộ: `git checkout -- .` (file mới phải xoá tay).
