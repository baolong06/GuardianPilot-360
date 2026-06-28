# GuardianPilot 360 — Kiến trúc AI-Agent Tổng hợp 4 Model DMS

## 1. Bối cảnh & Mục tiêu

Hệ thống có 4 model học sâu, mỗi model giải quyết một khía cạnh của giám sát trạng thái người lái (Driver Monitoring System — DMS), cùng nhận dữ liệu từ **một nguồn camera duy nhất** nhưng cần preprocessing khác nhau:

| Model | File | Input cần (từ cùng 1 frame camera) | Output |
|---|---|---|---|
| **M1 — Drowsiness DCNN** | `dcnn_drowsiness_task1_baseline.keras` | Ảnh mặt 96×96, normalize [0,1] | P(Drowsy) — xác suất buồn ngủ tức thời |
| **M2 — Microsleep CNN_16s** | `cnn_16s_best.keras` | Tín hiệu EEG/EOG 16s window — **lưu ý: cần sensor sinh lý riêng, không phải camera** | 4-class: Wake/MSE/MSEc/ED |
| **M3 — Distracted Driver (DBMNet + Baseline)** | `dbmnet_full_task3.keras` (chính) + `baseline_ghostnetlike_task3.keras` (fallback) | Ảnh toàn cảnh cabin 224×224, thang [0,255] | 10-class hành vi mất tập trung |
| **M4 — Landmark & Gaze (MLP/LSTM)** | `mlp_landmark_task4_fixed.keras` + `lstm_landmark_task4_fixed.keras` | Sequence 15 frame × 1440-dim landmark (qua `face_landmarker.task` + `landmark_scaler_task4.pkl`) | Binary: Alert/Drowsy |

**Vấn đề cốt lõi cần giải quyết:** 4 model này được huấn luyện **độc lập, trên dataset khác nhau, với giả định khác nhau** (M1 dùng ảnh crop mắt/mặt tĩnh, M3 dùng ảnh toàn cảnh, M4 dùng sequence landmark theo thời gian). Khi chạy đồng thời trên xe thật, chúng có thể:
- Đưa ra **kết luận mâu thuẫn** (M1 nói "tỉnh táo", M4 nói "buồn ngủ" cùng lúc).
- **Tranh giành tài nguyên** GPU/CPU nếu chạy vô tổ chức.
- Không có **cơ chế ưu tiên** khi cần đưa ra 1 cảnh báo duy nhất cho người lái.

Kiến trúc dưới đây giải quyết bằng mô hình **AI-Agent đa tác tử (Multi-Agent System) có Agent Chỉ huy (Orchestrator)**, ra quyết định dựa trên **Đồ thị Tri thức (Knowledge Graph)** chia sẻ giữa các agent.

---

## 2. Triết lý thiết kế: Tại sao cần Multi-Agent + Knowledge Graph?

### 2.1. Vấn đề nếu chạy 4 model độc lập (anti-pattern)
```
Camera → [M1, M2, M3, M4 chạy song song, không biết nhau] → 4 output rời rạc
                                                            → UI hiển thị lẫn lộn, người lái bối rối
```
Đây là cách **không nên làm**: không ai "chịu trách nhiệm" tổng hợp, dễ xung đột, dễ bỏ sót cảnh báo nghiêm trọng nếu 1 model bị nhiễu/lỗi.

### 2.2. Giải pháp: Agent hóa từng model + 1 Orchestrator
Mỗi model được bọc trong một **Perception Agent** riêng — không chỉ là "gọi model.predict()", mà có:
- Bộ nhớ ngắn hạn (lịch sử N frame gần nhất của riêng nó).
- Bộ ước lượng độ tin cậy (confidence/uncertainty) riêng.
- Khả năng tự báo "tôi không chắc" (degrade gracefully) khi input thiếu/nhiễu.

Một **Orchestrator Agent** (Agent Chỉ huy) không tự nó tính toán dự đoán nào cả — nó chỉ:
1. Đọc trạng thái mới nhất của tất cả Perception Agent từ Knowledge Graph.
2. Áp **luật ưu tiên + đối chiếu chéo (cross-validation)** để giải quyết xung đột.
3. Ra **1 quyết định cuối cùng duy nhất** (mức cảnh báo: Bình thường / Cảnh báo nhẹ / Cảnh báo nghiêm trọng / Khẩn cấp).
4. Ghi quyết định đó ngược lại vào Knowledge Graph (để các agent khác và lớp UI đọc).

Đây là kiến trúc **Blackboard System** kết hợp **Hierarchical Multi-Agent** — mẫu thiết kế kinh điển trong robotics/hệ thống giám sát an toàn (tương tự cấu trúc Subsumption Architecture của Rodney Brooks, áp dụng cho domain DMS).

---

## 3. Sơ đồ Kiến trúc Tổng thể

```
┌─────────────────────────────────────────────────────────────────────┐
│                         NGUỒN DỮ LIỆU (Sensors)                      │
│  Camera cabin (RGB, ~30fps)        Sensor EEG/EOG (nếu có lắp đặt)    │
└───────────────┬───────────────────────────────┬───────────────────────┘
                │                               │
                ▼                               ▼
┌───────────────────────────────┐   ┌───────────────────────────────┐
│   FRAME DISPATCHER (lõi xử lý) │   │  PHYSIO SIGNAL BUFFER (M2)     │
│  - Lấy 1 frame mới nhất         │   │  - Buffer tín hiệu 16s          │
│  - Tạo 3 bản preprocessing       │   │  - Chỉ hoạt động nếu có sensor  │
│    riêng cho M1/M3/M4           │   └──────────────┬────────────────┘
└─────┬──────────┬──────────┬────┘                  │
      │          │          │                       │
      ▼          ▼          ▼                       ▼
┌──────────┐┌──────────┐┌──────────┐         ┌──────────────┐
│ AGENT M1 ││ AGENT M3 ││ AGENT M4 │         │  AGENT M2     │
│Drowsiness││Distracted││ Landmark │         │  Microsleep   │
│  (DCNN)  ││ (DBMNet+ ││+Gaze MLP/│         │  (CNN_16s)    │
│          ││ baseline)││  LSTM    │         │               │
└────┬─────┘└────┬─────┘└────┬─────┘         └──────┬───────┘
     │            │            │                      │
     └────────────┴─────┬──────┴──────────────────────┘
                         ▼
         ┌───────────────────────────────────┐
         │      KNOWLEDGE GRAPH (KG)          │
         │  Lưu trạng thái real-time của      │
         │  toàn hệ thống (xem mục 4)          │
         └───────────────┬───────────────────┘
                         ▼
         ┌───────────────────────────────────┐
         │   ORCHESTRATOR AGENT (Agent Chỉ huy)│
         │  - Đọc KG                           │
         │  - Áp luật ưu tiên + đối chiếu chéo  │
         │  - Quyết định mức cảnh báo cuối      │
         │  - Ghi quyết định ngược vào KG       │
         └───────────────┬───────────────────┘
                         ▼
         ┌───────────────────────────────────┐
         │   ACTUATION LAYER (Lớp hành động)   │
         │  - Cảnh báo âm thanh/đèn/rung ghế   │
         │  - Ghi log an toàn (audit trail)     │
         │  - Gửi telemetry về trung tâm (nếu  │
         │    có kết nối)                       │
         └───────────────────────────────────┘
```

---

## 4. Đồ thị Tri thức (Knowledge Graph) — Cấu trúc chi tiết

Knowledge Graph là **bộ nhớ chia sẻ duy nhất** mà mọi agent đọc/viết — đây chính là cơ chế giúp các model "biết nhau" mà không cần gọi trực tiếp lẫn nhau (loose coupling).

### 4.1. Schema các Node

```
Node: DriverState (1 instance, cập nhật liên tục)
├── timestamp: ISO8601
├── current_alert_level: enum [NORMAL, MILD_WARNING, SEVERE_WARNING, EMERGENCY]
├── alert_reason: string (lý do agent Orchestrator chọn mức này)
└── confidence: float [0,1] (độ tin cậy tổng hợp của quyết định)

Node: PerceptionResult (1 instance / agent / frame, giữ lịch sử N=30 gần nhất — ~1s nếu 30fps)
├── source_agent: enum [M1_Drowsiness, M2_Microsleep, M3_Distracted, M4_LandmarkGaze]
├── timestamp: ISO8601
├── raw_output: dict (output gốc của model, ví dụ {"drowsy_prob": 0.82})
├── normalized_label: enum [ALERT, MILD_CONCERN, DROWSY, DISTRACTED, MICROSLEEP, PATHOLOGICAL_PROXY]
├── confidence: float [0,1]
├── input_quality: enum [GOOD, DEGRADED, MISSING]   # ví dụ: không detect được mặt → DEGRADED
└── processing_latency_ms: float

Node: ConflictEvent (tạo khi 2+ agent mâu thuẫn nhau)
├── timestamp: ISO8601
├── agents_involved: list[string]
├── conflicting_labels: dict (agent → label)
├── resolution_strategy: string (luật nào được áp dụng để giải quyết)
└── final_label: string

Node: AgentHealth (1 instance / agent, theo dõi tình trạng vận hành)
├── agent_id: string
├── is_alive: bool (heartbeat — agent có đang chạy bình thường không)
├── last_successful_inference: timestamp
├── consecutive_failures: int
└── degraded_mode: bool (đã chuyển sang fallback chưa, ví dụ M3 dùng baseline thay DBMNet)
```

### 4.2. Cấu trúc Edge (quan hệ)

```
PerceptionResult --[CONTRIBUTES_TO]--> DriverState
PerceptionResult --[CONFLICTS_WITH]--> PerceptionResult   (khi 2 agent ra label khác hướng cùng thời điểm)
AgentHealth --[MONITORS]--> PerceptionResult
ConflictEvent --[RESOLVED_BY]--> Orchestrator_Rule_ID
DriverState --[TRIGGERS]--> ActuationEvent
```

### 4.3. Lý do dùng đồ thị (không chỉ là database thường)

- **Truy vấn quan hệ nhanh:** Orchestrator cần hỏi "trong 2 giây gần nhất, có agent nào báo DROWSY mà M4 (landmark, đáng tin nhất cho trạng thái mắt) không xác nhận không?" — đây là truy vấn dạng graph traversal (so khớp pattern), tự nhiên hơn nhiều so với JOIN nhiều bảng SQL.
- **Audit trail tự nhiên:** Mỗi `ConflictEvent` lưu lại đúng *tại sao* một quyết định được đưa ra — bắt buộc cho compliance an toàn ô tô (truy vết khi có sự cố).
- **Mở rộng dễ:** Thêm Model M5 trong tương lai chỉ cần thêm 1 loại Node `PerceptionResult` mới, không cần đổi schema toàn hệ thống.

### 4.4. Công nghệ triển khai gợi ý

| Quy mô | Công nghệ |
|---|---|
| Demo/MVP nhanh | In-memory graph bằng Python `networkx`, hoặc dict lồng nhau có TTL (time-to-live) tự xoá node cũ |
| Production trên xe (edge, tài nguyên hạn chế) | SQLite + lớp truy vấn graph tự viết (nhẹ, không cần network), hoặc embedded graph DB như **Kùzu** (chạy được offline, nhẹ) |
| Production có kết nối cloud (fleet management) | Neo4j hoặc Amazon Neptune ở backend trung tâm, đồng bộ định kỳ từ buffer cục bộ trên xe |

---

## 5. Vai trò chi tiết từng Agent

### 5.1. Perception Agent — Mẫu chung (Template)

Mọi Perception Agent (M1–M4) tuân theo cùng 1 interface để Orchestrator xử lý đồng nhất:

```python
class PerceptionAgent(ABC):
    @abstractmethod
    def preprocess(self, raw_frame) -> ModelInput:
        """Biến frame thô thành input đúng format model cần."""

    @abstractmethod
    def infer(self, model_input: ModelInput) -> RawOutput:
        """Gọi model.predict(), trả về output thô."""

    @abstractmethod
    def normalize(self, raw_output: RawOutput) -> PerceptionResult:
        """Map output thô về schema chung (normalized_label, confidence)."""

    @abstractmethod
    def estimate_confidence(self, raw_output, input_quality) -> float:
        """Tính độ tin cậy — KHÔNG chỉ dựa vào softmax score, mà còn xét
        input_quality (ví dụ: không detect được mặt → confidence thấp dù
        model vẫn trả ra một số nào đó)."""

    def run(self, raw_frame) -> PerceptionResult:
        """Pipeline chuẩn: preprocess → infer → normalize → ghi vào KG."""
        model_input = self.preprocess(raw_frame)
        if model_input is None:           # input quality kém (ví dụ mất mặt)
            return self._emit_degraded_result()
        raw_output = self.infer(model_input)
        result = self.normalize(raw_output)
        result.confidence = self.estimate_confidence(raw_output, model_input.quality)
        self.knowledge_graph.write(result)
        return result
```

### 5.2. Agent M1 — Drowsiness (DCNN)
- **Vai trò:** tín hiệu nhanh, tần suất cao (mỗi frame), độ trễ thấp — "cảm biến cảnh báo sớm".
- **Đặc thù xử lý xung đột:** vì DCNN train trên ảnh crop mắt/mặt tĩnh, **dễ bị fooled** bởi ánh sáng/góc camera xấu → `estimate_confidence` PHẢI giảm mạnh nếu phát hiện ảnh quá tối/quá sáng (kiểm tra histogram trước khi đưa vào model).

### 5.3. Agent M2 — Microsleep (CNN_16s)
- **Vai trò:** tín hiệu "chậm nhưng sâu" — chỉ tin cậy khi có sensor EEG/EOG lắp đặt thật (nhiều xe demo/MVP sẽ KHÔNG có).
- **Quan trọng:** nếu không có sensor, Agent M2 phải tự báo `AgentHealth.is_alive = False` ngay khi khởi động, **không được giả lập input** — Orchestrator cần biết rõ M2 không tham gia, không phải M2 đang chạy nhưng cho kết quả rác.

### 5.4. Agent M3 — Distracted Driver (DBMNet + Baseline)
- **Vai trò:** phát hiện hành vi (cầm điện thoại, quay đầu, v.v.) — bổ trợ ngữ cảnh cho M1/M4.
- **Cơ chế fallback nội bộ (đã thống nhất với Long):**
  ```python
  def infer(self, model_input):
      action_out, view_out, f, f_hat = self.dbmnet.predict(model_input)
      dbmnet_confidence = np.max(action_out)
      if dbmnet_confidence < CONFIDENCE_THRESHOLD:   # ví dụ 0.5
          # DBMNet không chắc chắn -> dùng baseline làm tiếng nói thứ 2
          baseline_out = self.baseline.predict(model_input)
          if np.argmax(baseline_out) == np.argmax(action_out):
              return action_out, confidence=dbmnet_confidence * 1.1  # 2 model đồng thuận -> tăng nhẹ tin cậy
          else:
              return action_out, confidence=dbmnet_confidence * 0.7  # mâu thuẫn nội bộ -> giảm tin cậy, ghi ConflictEvent
      return action_out, confidence=dbmnet_confidence
  ```

### 5.5. Agent M4 — Landmark & Gaze (MLP/LSTM)
- **Vai trò:** tín hiệu "đáng tin nhất cho trạng thái mắt/đầu" vì input là landmark hình học, ít bị ảnh hưởng ánh sáng hơn ảnh thô.
- **Lưu ý vận hành quan trọng:** cần `face_landmarker.task` + `landmark_scaler_task4.pkl` (StandardScaler đã fit lúc train) — **phải dùng đúng scaler đã lưu, không fit lại**, nếu không sẽ lệch phân phối input so với lúc train.
- **LSTM vs MLP:** dùng **LSTM làm chính** (xét đúng tinh thần temporal/online monitoring), MLP làm tín hiệu phụ đối chiếu nhanh (1 frame, không cần đợi đủ 15 frame buffer).

### 5.6. Orchestrator Agent — Bộ não trung tâm

**Đầu vào:** toàn bộ `PerceptionResult` mới nhất từ KG (4 agent, hoặc ít hơn nếu agent nào degraded/offline).

**Luật ưu tiên (Priority Rules) — thiết kế theo nguyên tắc "an toàn trước, mượt mà sau":**

```
RULE 1 (Khẩn cấp tuyệt đối):
  IF M2.normalized_label == PATHOLOGICAL_PROXY AND M2.confidence > 0.6
  THEN alert_level = EMERGENCY   # ưu tiên cao nhất, không cần đồng thuận agent khác
  (lý do: nguy cơ sức khỏe nghiêm trọng, không thể chờ xác nhận chéo)

RULE 2 (Đồng thuận tăng cường — Sensor Fusion cơ bản):
  IF (M1.normalized_label == DROWSY AND M1.confidence > 0.5)
     AND (M4.normalized_label == DROWSY AND M4.confidence > 0.5)
  THEN alert_level = SEVERE_WARNING
  (lý do: 2 nguồn độc lập — ảnh thô và landmark hình học — cùng đồng ý)

RULE 3 (Tín hiệu đơn lẻ — cảnh báo nhẹ, chờ xác nhận):
  IF EXACTLY ONE OF (M1, M4) báo DROWSY với confidence > 0.6
  THEN alert_level = MILD_WARNING
       start_confirmation_window(2 giây)   # chờ frame tiếp theo xác nhận thêm

RULE 4 (Xung đột trực tiếp — log + thiên về an toàn):
  IF M1.normalized_label == DROWSY AND M4.normalized_label == ALERT
     (hoặc ngược lại) với cả 2 confidence > 0.5
  THEN ghi ConflictEvent
       alert_level = MILD_WARNING   # thiên về phía cẩn trọng hơn, KHÔNG bỏ qua hoàn toàn
       resolution_strategy = "trust_landmark_over_raw_image_on_lighting_uncertainty"
       (lý do: M4 landmark ít bị ảnh hưởng ánh sáng hơn M1, nhưng vẫn không bỏ qua
        cảnh báo của M1 hoàn toàn — false negative nguy hiểm hơn false positive trong DMS)

RULE 5 (Distracted độc lập với Drowsy):
  IF M3.normalized_label == DISTRACTED AND M3.confidence > 0.6
  THEN alert_level = max(current_alert_level, MILD_WARNING)
       (lý do: mất tập trung và buồn ngủ là 2 trục khác nhau, không loại trừ nhau —
        cộng dồn mức cảnh báo, không để cái này che lấp cái kia)

RULE 6 (Agent offline/degraded — không được im lặng):
  IF any(AgentHealth.is_alive == False)
  THEN ghi log cảnh báo vận hành (KHÔNG phải cảnh báo cho người lái)
       hệ thống tiếp tục chạy với agent còn lại, nhưng giảm
       max_achievable_confidence xuống (vì thiếu 1 nguồn xác nhận chéo)

RULE 7 (Mặc định):
  ELSE alert_level = NORMAL
```

**Nguyên tắc tổng quát khi viết luật mới (để Long mở rộng sau):**
1. **An toàn > Tiện lợi:** khi nghi ngờ, luôn chọn mức cảnh báo cao hơn (giảm false negative), kể cả khi phải chịu thêm false positive.
2. **Không có model nào "luôn đúng tuyệt đối"** — mọi luật đều dùng `confidence`, không dùng `normalized_label` đơn lẻ làm quyết định cuối.
3. **Mọi quyết định phải ghi lại lý do** (`alert_reason`) — không có "hộp đen" trong hệ thống an toàn.

---

## 6. Luồng xử lý 1 chu kỳ (Tick) — Ví dụ cụ thể

```
T = 0ms    : Camera trả về frame mới
T = 5ms    : Frame Dispatcher tạo 3 bản preprocessing (96×96 cho M1, 224×224 cho M3,
             gửi vào buffer sequence cho M4)
T = 10ms   : M1, M3, M4 bắt đầu infer song song (3 thread/process riêng)
             M2 đọc buffer EEG/EOG riêng (độc lập hoàn toàn với camera tick)
T = 35ms   : M1 trả kết quả (DCNN nhẹ, nhanh nhất) → ghi PerceptionResult vào KG
T = 60ms   : M3 trả kết quả (DBMNet, MobileNetV3 nặng hơn) → ghi vào KG
             (Agent M3 tự kiểm tra confidence DBMNet, gọi baseline nếu cần — xem 5.4)
T = 80ms   : M4 đã đủ 15 frame trong buffer → LSTM trả kết quả → ghi vào KG
T = 85ms   : Orchestrator được trigger (sau khi ít nhất M1+M4 đã ghi xong) →
             đọc toàn bộ PerceptionResult mới nhất từ KG → áp luật (mục 5.6) →
             ra alert_level → ghi DriverState mới vào KG
T = 90ms   : Actuation Layer đọc DriverState mới → nếu alert_level thay đổi so với
             trước, kích hoạt cảnh báo tương ứng (âm thanh nhẹ / rung ghế / còi gấp)
T = 95ms   : Toàn bộ log của tick này được flush vào audit trail (cho compliance)

→ Tổng latency end-to-end: ~95ms, đủ nhanh cho real-time (mục tiêu <150ms/tick
  để không trễ hơn 1 khoảng phản ứng người lái trung bình ~200-300ms)
```

**Lưu ý quan trọng về đồng bộ:** Orchestrator **không chờ tất cả 4 agent xong mới chạy** (nếu vậy, agent chậm nhất — M2 nếu có sensor — sẽ làm trễ toàn hệ thống). Thay vào đó dùng cơ chế **"chạy với dữ liệu mới nhất có sẵn, đánh dấu rõ agent nào đã/chưa cập nhật trong tick này"** — đây chính là lý do KG cần trường `timestamp` ở mọi node, để Orchestrator biết dữ liệu nào còn "tươi" (trong ngưỡng vài trăm ms) và dữ liệu nào đã cũ cần bỏ qua.

---

## 7. Xử lý Lỗi & Suy giảm có Kiểm soát (Graceful Degradation)

Đây là phần **bắt buộc** cho sản phẩm thật chạy trên xe — không được để 1 model lỗi làm sập cả hệ thống.

| Tình huống lỗi | Hành vi mong đợi |
|---|---|
| Camera mất tín hiệu hoàn toàn | Tất cả M1/M3/M4 → `AgentHealth.is_alive=False`. Orchestrator chuyển `alert_level=SEVERE_WARNING` (không thấy gì = nguy hiểm, không phải = bình thường) |
| Không detect được mặt (ánh sáng/góc camera) | M1, M4 → `input_quality=DEGRADED`, confidence tự động giảm theo công thức ở 5.1, KHÔNG trả lỗi crash |
| M3 (DBMNet) lỗi runtime (OOM, exception) | Tự động fallback sang `baseline_ghostnetlike_task3.keras` hoàn toàn (không chỉ khi confidence thấp), ghi `AgentHealth.degraded_mode=True` |
| Sensor EEG/EOG không lắp đặt (M2) | M2 không khởi động, `is_alive=False` ngay từ đầu, Orchestrator hoạt động với 3 agent còn lại — RULE 1 (pathological) sẽ không thể kích hoạt, cần ghi rõ trong UI là "Microsleep monitoring: Không khả dụng" |
| `landmark_scaler_task4.pkl` không load được | M4 không khởi động — không được dùng input chưa chuẩn hóa đưa thẳng vào model (sẽ cho kết quả sai hoàn toàn, nguy hiểm hơn là không có M4) |

---

## 8. Roadmap Triển khai Gợi ý

### Giai đoạn 1 — MVP (validate kiến trúc, chưa cần real-time)
- Code 4 Perception Agent (Python class theo template mục 5.1).
- Knowledge Graph bằng `networkx` in-memory.
- Orchestrator chạy luật if/else đơn giản (mục 5.6).
- Test trên video ghi sẵn (offline), không cần camera thật.

### Giai đoạn 2 — Real-time trên máy tính (trước khi đưa lên xe)
- Chuyển sang xử lý đa luồng/đa tiến trình (Python `multiprocessing` hoặc `asyncio` cho I/O-bound phần camera).
- Đo latency thật từng agent, tối ưu (quantize model, dùng TensorRT/ONNX Runtime nếu cần).
- Viết Actuation Layer giả lập (in ra console/log thay vì còi/rung thật).

### Giai đoạn 3 — Triển khai Edge trên xe
- Đóng gói từng Agent (có thể dùng Docker container riêng hoặc tiến trình native nếu thiết bị edge hạn chế tài nguyên).
- Knowledge Graph chuyển sang SQLite/Kùzu (embedded, không cần network).
- Thêm watchdog process giám sát `AgentHealth` toàn hệ thống (nếu 1 agent crash, tự restart, không kéo sập cả hệ thống).
- Tích hợp Actuation Layer thật (CAN bus xe, loa, đèn cảnh báo).

### Giai đoạn 4 — Fleet Management (nếu mở rộng quy mô)
- Đồng bộ định kỳ Knowledge Graph cục bộ lên Neo4j/Neptune trung tâm.
- Dashboard giám sát nhiều xe, phân tích pattern cảnh báo theo thời gian/tài xế.

---

## 9. Tóm tắt Nguyên tắc Thiết kế Cốt lõi

1. **Mỗi model = 1 Agent có trạng thái, độ tin cậy, và khả năng tự báo "không chắc"** — không gọi model.predict() trần trụi.
2. **Knowledge Graph là nguồn sự thật duy nhất** — agent không gọi trực tiếp lẫn nhau, giảm coupling, dễ thêm/bớt model sau này.
3. **Orchestrator không tự suy luận từ ảnh/sensor** — nó chỉ đọc trạng thái đã được các Perception Agent chuẩn hóa, áp luật ưu tiên rõ ràng, có thể audit.
4. **An toàn được ưu tiên hơn độ chính xác đơn lẻ** — khi 2 model mâu thuẫn, chọn hướng cẩn trọng hơn, không lấy trung bình đơn giản.
5. **Suy giảm có kiểm soát** — mất 1 nguồn dữ liệu/model không làm sập hệ thống, nhưng phải được ghi nhận và phản ánh đúng vào mức tin cậy tổng thể.
