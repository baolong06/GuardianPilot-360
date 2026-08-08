# PRD - Product Requirement Document
## GuardianPilot360 - Hệ thống AI giám sát buồn ngủ và hỗ trợ an toàn cho tài xế

**Phiên bản:** MVP v1.0
**Trạng thái:** Đề xuất phát triển
**Phạm vi triển khai ban đầu**

---

## 1. Giới thiệu tài liệu

GuardianPilot360 là hệ thống Camera AI được thiết kế để giám sát trạng thái tỉnh táo của tài xế trong thời gian thực.

### 1.1. Mục đích

Hệ thống tập trung vào việc:
- Phát hiện dấu hiệu mệt mỏi và buồn ngủ.
- Phát hiện tài xế nhắm mắt kéo dài.
- Phát hiện ngủ gật hoặc gục đầu.
- Phát hiện tài xế không phản ứng sau cảnh báo.
- Phát cảnh báo theo nhiều mức độ tăng dần.
- Ghi lại sự kiện để người quản lý đội xe theo dõi.
- Trong tương lai, hỗ trợ phương tiện thực hiện quy trình đưa xe về trạng thái an toàn khi tài xế không thể tiếp tục điều khiển.

> Hệ thống không chẩn đoán tình trạng sức khỏe của tài xế. Kết quả AI chỉ được sử dụng để hỗ trợ cảnh báo an toàn.

### 1.2. Phạm vi áp dụng

Tài liệu áp dụng cho dự án phát triển MVP v1.0 của hệ thống, bao gồm:
- Camera hồng ngoại giám sát tài xế trong cabin.
- Phần mềm xử lý hình ảnh và video.
- Module phát hiện khuôn mặt và vùng mắt.
- Module tính EAR và PERCLOS.
- Module phát hiện ngáp.
- Module phân tích Head Pose và gục đầu.
- Module đánh giá mức độ buồn ngủ.
- Hệ thống cảnh báo âm thanh đa mức.
- HUD Dashboard hiển thị trạng thái tài xế.
- Event Logger lưu thông tin sự kiện.
- API gửi cảnh báo đến người quản lý.
- GPS ghi nhận vị trí xảy ra sự kiện.

### 1.3. Đối tượng đọc tài liệu

| Đối tượng | Vai trò |
|---|---|
| Ban Lãnh đạo / Product Owner | Phê duyệt chiến lược sản phẩm, nguồn lực và lộ trình |
| Technical Lead / Kiến trúc sư hệ thống | Thiết kế kiến trúc, lựa chọn công nghệ |
| Đội ngũ AI / ML Engineer | Phát triển và huấn luyện mô hình |
| Đội ngũ Embedded / Firmware | Triển khai trên Jetson, tối ưu hiệu năng |
| Đội ngũ Frontend / Backend | Phát triển UI Dashboard và Web App |
| Đội ngũ QA / Test | Xây dựng kế hoạch kiểm thử |
| Đội ngũ Triển khai / Bảo trì | Lắp đặt, hiệu chỉnh và vận hành hệ thống |

---

## 2. Bối cảnh và vấn đề (Background / Problem Statement)

### 2.1. Hiện trạng

**Thực trạng giao thông đường bộ:**
- Tài xế xe tải, xe khách và phương tiện đường dài thường phải lái xe trong thời gian dài, đặc biệt vào ban đêm.
- Mệt mỏi, buồn ngủ và ngủ gật làm giảm khả năng quan sát, phản ứng và kiểm soát phương tiện.
- Microsleep có thể xảy ra trong vài giây nhưng vẫn đủ để phương tiện di chuyển một khoảng cách lớn mà tài xế gần như không kiểm soát được.
- Camera hành trình thông thường chủ yếu ghi hình và chưa chủ động đánh giá mức độ tỉnh táo của tài xế.
- Việc phát hiện sớm dấu hiệu mệt mỏi có thể giúp cảnh báo tài xế trước khi trạng thái buồn ngủ chuyển sang mức nguy hiểm.

**Hạ tầng pháp lý:**
- Hệ thống phải tuân thủ các quy định hiện hành liên quan đến thiết bị giám sát hành trình và ghi nhận hình ảnh người lái xe.
- Video và dữ liệu tài xế phải được xử lý theo nguyên tắc bảo mật và giới hạn quyền truy cập.
- Các chức năng trực tiếp điều khiển phương tiện trong roadmap tương lai phải được đánh giá riêng về tiêu chuẩn an toàn chức năng, phương tiện và quy định pháp luật trước khi triển khai.

**Thị trường:**
- Nhiều camera hành trình hiện nay cung cấp cảnh báo ADAS cơ bản.
- Các giải pháp DMS chuyên sâu có khả năng đánh giá liên tục EAR, PERCLOS, trạng thái đầu và hành vi tài xế vẫn còn hạn chế trong phân khúc phổ thông.
- Cơ hội của GuardianPilot360 là tập trung sâu vào bài toán Driver Monitoring System thay vì tích hợp quá nhiều module không liên quan.
- Trong dài hạn, GuardianPilot360 có thể mở rộng từ DMS sang Driver Emergency Assistance.

### 2.2. Pain points

| Nhóm đối tượng | Pain Point |
|---|---|
| Nhà vận tải / Chủ xe | Không biết tài xế có đang mệt mỏi hoặc ngủ gật khi thực hiện hành trình dài; khó giám sát nhiều phương tiện đồng thời |
| Tài xế | Mệt mỏi sau nhiều giờ lái xe; đôi khi không tự nhận biết mình đang bắt đầu ngủ gật; cảnh báo thông thường chưa thích ứng theo mức độ nguy hiểm |
| Kỹ thuật viên | Khó cân bằng giữa độ nhạy phát hiện buồn ngủ và tỷ lệ cảnh báo sai |
| Hành khách | Lo ngại tài xế mệt mỏi hoặc ngủ gật trên các chuyến xe dài |

### 2.3. Cơ hội

- **Cơ hội an toàn:** Phát hiện sớm dấu hiệu buồn ngủ có thể giúp tài xế chủ động nghỉ ngơi trước khi xảy ra tình huống nguy hiểm.
- **Cơ hội công nghệ:** Camera IR, Face Landmark, YOLO, MediaPipe, PERCLOS và TensorRT cho phép triển khai DMS Real-Time trên thiết bị Edge.
- **Cơ hội thị trường:** Doanh nghiệp vận tải có nhu cầu giám sát trạng thái tài xế trên xe tải, xe khách và xe đường dài.
- **Cơ hội phát triển:** Hệ thống có thể mở rộng từ DMS sang Emergency Driver Assistance bằng camera 360 độ, radar và dữ liệu phương tiện.

---

## 3. Mục tiêu sản phẩm và Success Metric

### 3.1. Mục tiêu kinh doanh

| # | Mục tiêu | Chỉ tiêu |
|---|---|---|
| BG1 | Xây dựng MVP phát hiện buồn ngủ tài xế trên nền tảng AI Edge | MVP v1.0 hoàn thành và lắp đặt thử nghiệm trên 2-3 xe |
| BG2 | Giảm rủi ro tai nạn liên quan đến buồn ngủ và mất tỉnh táo | Phát hiện và cảnh báo trước khi trạng thái chuyển sang nguy hiểm |
| BG3 | Tạo sản phẩm DMS chuyên biệt cho doanh nghiệp vận tải | Hỗ trợ Dashboard và lịch sử cảnh báo theo tài xế/xe |
| BG4 | Tạo nền tảng cho Driver Emergency Assistance trong tương lai | Kiến trúc MVP có khả năng mở rộng sang camera 360 và hệ thống điều khiển phương tiện |

### 3.2. Mục tiêu sản phẩm (Product Goals)

| # | Mục tiêu | Mô tả |
|---|---|---|
| PG1 | Phát hiện buồn ngủ | Nhận diện sớm dấu hiệu mệt mỏi qua mắt, PERCLOS, ngáp và Head Pose |
| PG2 | Cảnh báo tăng dần | Cảnh báo tài xế theo mức độ từ nhắc nhở đến nguy hiểm |
| PG3 | Đánh giá phản ứng | Theo dõi xem tài xế có mở mắt và phục hồi sau cảnh báo hay không |
| PG4 | Quản lý đội xe | Dashboard giám sát, Event Log, GPS và thống kê cảnh báo |
| PG5 | Khả năng mở rộng | Chuẩn bị kiến trúc cho camera 360 và Driver Emergency Assistance trong tương lai |

### 3.3. Success Metric

| Metric | Mô tả | Target |
|---|---|---|
| FPS | Tốc độ xử lý khung hình trên Jetson | 15–30 FPS ổn định |
| Latency end-to-end | Capture → inference → alert | < 200ms cho DMS |
| Face Detection | Nhận diện khuôn mặt tài xế | Precision > 95% |
| Eye State Detection | Phân loại mắt mở/nhắm | Accuracy > 90% |
| Drowsiness Recall | Khả năng phát hiện trường hợp buồn ngủ | > 90% |
| PERCLOS Detection | Theo dõi tỷ lệ thời gian mắt nhắm | Rolling window 30s |
| Yawn Detection | Phát hiện hành vi ngáp | Recall > 85% |
| False Positive Rate | Tỷ lệ cảnh báo sai | < 5% |
| Head Nod Detection | Phát hiện gục đầu | Recall > 85% |
| System Uptime | Độ ổn định | > 99% trong 2 giờ chạy liên tục |
| Alert Latency | Thời gian từ khi đủ điều kiện đến cảnh báo | < 1s |
| Temperature | Nhiệt độ hoạt động | < 75°C |

---

## 4. Đối tượng người dùng (User Persona)

**Persona 1: Chủ doanh nghiệp vận tải**
- Nhu cầu: Kiểm soát trạng thái tài xế, giảm nguy cơ tai nạn, xem lại lịch sử cảnh báo theo xe và tài xế.
- Khó khăn: Không thể trực tiếp theo dõi trạng thái hàng chục hoặc hàng trăm tài xế trên đường.

**Persona 2: Tài xế xe đường dài**
- Nhu cầu: Được cảnh báo sớm khi bắt đầu mệt mỏi hoặc buồn ngủ.
- Khó khăn: Hành trình dài, lái xe ban đêm, dễ chủ quan và không nhận biết microsleep.

**Persona 3: Kỹ sư AI / Kỹ thuật viên lắp đặt**
- Nhu cầu: Cài đặt và hiệu chỉnh camera cabin; theo dõi độ chính xác và hiệu năng của hệ thống.
- Khó khăn: Điều kiện ánh sáng, kính mắt, góc camera và đặc điểm khuôn mặt khác nhau giữa các tài xế.

---

## 5. Phạm vi sản phẩm

### 5.1. In Scope (Build Phase)

**A. Phần cứng và nền tảng**
- Lắp đặt và cấu hình NVIDIA Jetson Nano / TX2 / Orin Nano.
- Kết nối camera hồng ngoại trong cabin qua GStreamer pipeline.
- Hỗ trợ camera hoạt động ngày và đêm.
- Cài đặt heatsink, quạt tản nhiệt và giới hạn TDP.
- Tích hợp GPS và đồng bộ thời gian NTP.

**B. Module DMS (Driver Monitoring System)**
- Phát hiện khuôn mặt tài xế.
- Trích xuất Face Landmark.
- Phát hiện vùng mắt.
- Tính EAR.
- Phân loại mắt mở / nhắm.
- Tính PERCLOS theo rolling window.
- Phát hiện ngáp thông qua MAR hoặc mô hình AI.
- Phân tích Head Pose.
- Phát hiện gục đầu.
- Phát hiện microsleep.
- Kết hợp nhiều feature để tạo Drowsiness Score.

**C. Module Eye State Detection**
- Theo dõi trạng thái hai mắt theo thời gian thực.
- Phát hiện mắt nhắm liên tục.
- Tính thời gian nhắm mắt.
- Phân biệt chớp mắt bình thường và nhắm mắt kéo dài.
- Đưa output vào Drowsiness Scoring Engine.

**D. Module PERCLOS & Temporal Analysis**
- Tính tỷ lệ thời gian mắt nhắm trong rolling window 30 giây.
- Theo dõi xu hướng PERCLOS.
- Kết hợp nhiều frame thay vì quyết định từ một ảnh.
- Reset hoặc giảm mức cảnh báo khi tài xế phục hồi.

**E. Module Yawn Detection**
- Phát hiện trạng thái há miệng.
- Tính MAR hoặc sử dụng classifier.
- Đếm số lần ngáp trong cửa sổ thời gian.
- Kết hợp ngáp với PERCLOS để tăng độ tin cậy.

**F. Module Head Pose & Head Nod**
- Ước lượng Pitch, Yaw và Roll.
- Phát hiện cúi đầu kéo dài.
- Phát hiện gục đầu lặp lại.
- Phân biệt nhìn xuống tạm thời và hành vi ngủ gật.

**G. Module Drowsiness Scoring**
- Tổng hợp EAR, PERCLOS, Eye Closure Duration, Yawn Frequency, Head Pose, Head Nod Frequency, Face visibility.
- Tạo trạng thái NORMAL / FATIGUE / DROWSY / MICROSLEEP / CRITICAL.

**H. Module Hệ thống & Tích hợp**
- Pipeline xử lý DMS Real-Time.
- Hệ thống cảnh báo âm thanh nhiều mức.
- TensorRT INT8 quantization.
- Đạt 15–30 FPS ổn định.
- Watchdog restart khi module gặp lỗi.
- Event Logger lưu thông tin cảnh báo.

**I. UI Dashboard & Giám sát**
- HUD Dashboard trên xe: trạng thái tài xế, mức cảnh báo và thông tin hệ thống.
- Web App: live status, replay log, thống kê theo xe/tài xế/ngày.
- Event Logger.

**J. Cloud & API**
- API truyền dữ liệu hành trình lên Cloud / Cục CSGT.
- REST API.
- Mã hóa dữ liệu khi truyền.
- Retry khi mất mạng.
- Batch upload khi offline.
- GPS tracking.
- Dashboard cho người quản lý.

**K. Kiểm thử & Triển khai**
- Test DMS với clip tài xế tỉnh táo, mệt mỏi, chớp mắt bình thường, mắt nhắm kéo dài, ngáp, gục đầu, microsleep mô phỏng.
- Test ban ngày và ban đêm.
- Test tài xế đeo kính.
- Test thay đổi góc camera.
- Test tích hợp pipeline end-to-end.
- Lắp đặt thử nghiệm 2-3 xe.

### 5.2. Out-of-Scope (Current phase)

| Tính năng | Lý do |
|---|---|
| Tự động điều khiển vô-lăng | Không thuộc MVP DMS |
| Tự động điều khiển chân ga | Yêu cầu tích hợp sâu với hệ thống xe |
| Tự động phanh | Yêu cầu CAN Bus và hệ thống an toàn phương tiện |
| Tự động chuyển làn | Thuộc Driver Emergency Assistance trong tương lai |
| Tự động đỗ vào làn bên phải | Thuộc roadmap tương lai |
| Tự lái hoàn toàn | Vượt phạm vi MVP |
| Camera 360 điều khiển xe | Phase sau |
| Radar / LiDAR Sensor Fusion | Phase sau |
| Nhận diện biển báo | Không phục vụ trực tiếp bài toán DMS MVP |
| Nhận diện ổ gà | Không phục vụ trực tiếp bài toán DMS MVP |
| LLM tra cứu luật giao thông | Không thuộc phạm vi chính |
| Nhận diện hiệu lệnh CSGT | Không thuộc phạm vi chính |

---

## 6. Yêu cầu tính năng

### 6.1. Tính năng cơ bản

**Module DMS**

| ID | Yêu cầu | Ưu tiên |
|---|---|---|
| DMS-01 | Hệ thống phát hiện khuôn mặt tài xế | P0 |
| DMS-02 | Phát hiện Face Landmark và vùng mắt | P0 |
| DMS-03 | Tính EAR cho cả hai mắt | P0 |
| DMS-04 | Phân loại trạng thái mắt mở / nhắm | P0 |
| DMS-05 | Tính thời gian mắt nhắm liên tục | P0 |
| DMS-06 | Tính PERCLOS theo rolling window 30 giây | P0 |
| DMS-07 | Phát hiện hành vi ngáp | P0 |
| DMS-09 | Ước lượng Head Pose | P0 |
| DMS-10 | Phát hiện gục đầu | P0 |
| DMS-11 | Phát hiện microsleep | P0 |
| DMS-12 | Tính Drowsiness Score | P0 |
| DMS-13 | Phân loại mức độ cảnh báo | P1 |
| DMS-14 | Ghi nhận sự kiện kèm ảnh và GPS | P1 |

**Module Eye & PERCLOS**

| ID | Yêu cầu | Ưu tiên |
|---|---|---|
| EYE-01 | Theo dõi trạng thái hai mắt theo thời gian thực | P0 |
| EYE-02 | Phân biệt blink và eye closure kéo dài | P0 |
| EYE-03 | Tính blink duration | P0 |
| EYE-04 | Tính PERCLOS | P0 |
| EYE-05 | Reset trạng thái khi tài xế phục hồi | P0 |

**Module Yawn & Head Pose**

| ID | Yêu cầu | Ưu tiên |
|---|---|---|
| YHP-01 | Phát hiện miệng và hành vi ngáp | P0 |
| YHP-02 | Đếm số lần ngáp theo cửa sổ thời gian | P1 |
| YHP-03 | Ước lượng Pitch/Yaw/Roll | P0 |
| YHP-04 | Phát hiện cúi hoặc gục đầu | P0 |
| YHP-05 | Phát hiện đầu không hướng về phía trước kéo dài | P1 |

**Module Alert**

| ID | Yêu cầu | Ưu tiên |
|---|---|---|
| ALT-01 | Cảnh báo cấp 1 khi xuất hiện dấu hiệu mệt mỏi | P0 |
| ALT-02 | Cảnh báo cấp 2 khi xác định buồn ngủ | P0 |
| ALT-03 | Cảnh báo cấp 4 khi tài xế không phục hồi | P0 |
| ALT-04 | Cảnh báo cấp 4 khi tài xế không phục hồi | P0 |
| ALT-05 | Tự động hạ cấp khi tài xế tỉnh táo trở lại | P0 |
| ALT-06 | Gửi cảnh báo cấp cao lên Dashboard | P1 |

**Module Hệ thống & Tích hợp**

| ID | Yêu cầu | Ưu tiên |
|---|---|---|
| SYS-01 | Pipeline xử lý DMS Real-Time | P0 |
| SYS-02 | Hệ thống cảnh báo âm thanh nhiều cấp độ | P0 |
| SYS-03 | TensorRT INT8, đạt 15-30 FPS | P0 |
| SYS-04 | Monitor CPU, GPU, RAM và nhiệt độ | P0 |
| SYS-05 | Watchdog restart tự động khi module bị treo | P1 |
| SYS-06 | Lưu Event Log trước khi upload Cloud | P1 |

**Module UX/UI**

| ID | Yêu cầu | Ưu tiên |
|---|---|---|
| UI-01 | HUD Dashboard dark mode, font lớn | P0 |
| UI-02 | Hiển thị trạng thái tài xế | P0 |
| UI-03 | Hiển thị mức độ NORMAL/FATIGUE/DROWSY/CRITICAL | P0 |
| UI-04 | Màu cảnh báo xanh/vàng/cam/đỏ | P0 |
| UI-05 | Web App quản lý đội xe | P1 |
| UI-06 | Replay Event Log | P1 |

**Cloud & API**

| ID | Yêu cầu | Ưu tiên |
|---|---|---|
| API-01 | REST API truyền Event Log | P0 |
| API-01 | Retry khi mất mạng | P0 |
| API-01 | Batch upload khi offline | P0 |
| API-01 | GPS tracking + NTP sync | P0 |
| API-01 | Authentication và mã hóa dữ liệu | P1 |

**UI/UX (bổ sung)**

| ID | Yêu cầu | Ưu tiên |
|---|---|---|
| UI-01 | HUD Dashboard chuẩn ADAS: dark mode, font lớn | P0 |
| UI-02 | Hiển thị tốc độ hiện tại, cảnh báo, biển báo, trạng thái tài xế | P0 |
| UI-03 | Màu cảnh báo chuẩn: đỏ (nguy hiểm), vàng (cảnh báo), xanh (thông tin) | P0 |
| UI-04 | Web App quản lý đội xe: live feed, replay log, thống kê | P1 |
| UI-05 | Giao diện thân thiện với mobile | P2 |

**Cloud & API (bổ sung)**

| ID | Yêu cầu | Ưu tiên |
|---|---|---|
| API-01 | REST API gzip + AES-256 encryption | P0 |
| API-02 | Retry on fail, batch upload khi offline | P0 |
| API-03 | Tuân thủ format Thông tư 73/2024/TT-BCA | — |
| API-04 | GPS tracking + NTP sync mỗi 1 phút | P0 |
| API-05 | Certificate pinning và bảo mật API | P1 |

### 6.2. Tính năng nâng cao

| Tính năng | Mô tả | Roadmap |
|---|---|---|
| Cá nhân hóa Drowsiness Threshold | Điều chỉnh baseline EAR/PERCLOS theo từng tài xế | Phase 2 |
| Phân tích hành vi lái xe theo thời gian | Thống kê mức độ mệt mỏi theo ca lái | Phase 2 |
| Camera 360 Perception | Camera trước, sau, trái và phải quan sát môi trường xe | Phase 3 |
| Lane Detection | Xác định làn hiện tại và làn bên phải | Phase 3 |
| Vehicle Surrounding Detection | Phát hiện xe trước, sau và điểm mù | Phase 3 |
| Emergency Lane Keeping | Duy trì phương tiện ổn định trong làn khi tài xế không phản hồi | Phase 4 |
| Automatic Deceleration | Giảm tốc có kiểm soát | Phase 4 |
| Emergency Lane Change | Chuyển sang làn bên phải khi đảm bảo điều kiện an toàn | Phase 5 |
| Minimal Risk Stop | Đưa xe về trạng thái dừng an toàn và kích hoạt phanh đỗ | Phase 5 |

---

## 7. User Stories và Acceptance Criteria

### US1: Tài xế

**US1.1: Cảnh báo khi tôi bắt đầu có dấu hiệu buồn ngủ**
- AC1: Hệ thống phát hiện khuôn mặt tài xế trong vòng 100ms.
- AC2: EAR được tính liên tục.
- AC3: Hệ thống phân biệt blink bình thường và mắt nhắm kéo dài.
- AC4: PERCLOS được tính trên rolling window.
- AC5: Phát hiện được ngáp.
- AC6: Phát hiện được gục đầu.
- AC7: Không tạo cảnh báo chỉ từ một frame.
- AC8: False positive < 5%.

**US1.2: Cảnh báo theo mức độ tăng dần**
- AC1: NORMAL không phát cảnh báo.
- AC2: FATIGUE phát cảnh báo cấp 1.
- AC3: DROWSY phát cảnh báo cấp 2.
- AC4: MICROSLEEP phát cảnh báo cấp 3.
- AC5: Không phục hồi sau cảnh báo phát cảnh báo cấp 4.
- AC6: Mức cảnh báo giảm khi tài xế phục hồi.
- AC7: Mỗi thay đổi trạng thái được ghi Event Log.

**US1.3: Tôi muốn hệ thống nhận biết khi tôi ngáp hoặc gục đầu để cảnh báo sớm**
- AC1: Phát hiện hành vi ngáp.
- AC2: Theo dõi số lần ngáp.
- AC3: Phát hiện Pitch Head Pose bất thường.
- AC4: Phát hiện gục đầu liên tục.
- AC5: Kết hợp kết quả với EAR và PERCLOS.

**US1.4: Tôi muốn hệ thống cảnh báo mạnh hơn nếu tôi tiếp tục ngủ gật**
- AC1: Cấp độ cảnh báo tăng theo Drowsiness Score.
- AC2: Cảnh báo cấp cao sử dụng âm thanh lớn và lặp lại.
- AC3: Cảnh báo hiển thị trên HUD.
- AC4: Cảnh báo nghiêm trọng được gửi lên Dashboard.
- AC5: Event phải lưu snapshot và GPS.

**US1.5: Tôi muốn hệ thống nhận biết khi tôi đã tỉnh lại**
- AC1: Phát hiện mắt mở trở lại.
- AC2: Head Pose trở về trạng thái bình thường.
- AC3: PERCLOS giảm dưới ngưỡng cảnh báo.
- AC4: Không còn gục đầu.
- AC5: Hệ thống giảm mức cảnh báo sau khoảng thời gian phục hồi ổn định.

**US1.6: Tôi muốn cảnh báo không bị kích hoạt chỉ vì tôi chớp mắt bình thường**
- AC1: Blink ngắn không kích hoạt cảnh báo.
- AC2: Một frame mắt nhắm không được coi là ngủ.
- AC3: Quyết định phải dựa trên chuỗi thời gian.
- AC4: False positive < 5%.

**US1.7: Tôi muốn hệ thống vẫn hoạt động vào ban đêm**
- AC1: Camera IR hoạt động trong cabin tối.
- AC2: Phát hiện khuôn mặt trong điều kiện ánh sáng thấp.
- AC3: EAR và PERCLOS vẫn được tính.
- AC4: Test trên dữ liệu ngày và đêm.

**US1.8: Tôi muốn xem lịch sử cảnh báo từ xa để quản lý đội xe**
- AC1: Web App hiển thị cảnh báo theo tài xế.
- AC2: Replay log theo xe/tài xế/ngày.
- AC3: Hiển thị mức Drowsiness.
- AC4: Hiển thị ảnh snapshot.
- AC5: Hiển thị GPS.

**US1.9: Tôi muốn dữ liệu tài xế được bảo mật**
- AC1: API Authentication.
- AC2: Data encryption.
- AC3: Phân quyền người dùng.
- AC4: Log truy cập.
- AC5: GPS tracking + NTP sync.

---

## 8. Luồng nghiệp vụ chính

**Luồng 1: Khởi động và xác định tài xế**
Camera Cabin → Khởi tạo DMS → Phát hiện khuôn mặt → Xác định ROI khuôn mặt/Mắt → Khởi tạo Baseline → Bắt đầu giám sát.

**Luồng 2: Giám sát tài xế (Real-Time)**
Camera → Face Detect → Face Landmark → EAR/Eye State → PERCLOS → Yawn Detection → Head Pose → Drowsiness Score → Driver State.

**Luồng 3: Cảnh báo buồn ngủ**
NORMAL → FATIGUE → Cảnh báo cấp 1 → DROWSY → Cảnh báo cấp 2 → MICROSLEEP → Cảnh báo cấp 3 → Không phục hồi → CRITICAL → Cảnh báo cấp 4.

**Luồng 4: Phục hồi trạng thái**
Cảnh báo → Tài xế mở mắt → Head Pose bình thường → EAR phục hồi → PERCLOS giảm → Theo dõi ổn định → RECOVERED → NORMAL.

**Luồng 5: End-to-End Pipeline**
Camera IR → Preprocessing → Face Detection → Landmark Detection → Feature Extraction → Temporal Analysis → Drowsiness Scoring → Alert Manager → Event Logger → Dashboard / Cloud.

**Luồng 6: Quản lý từ xa báo cáo**
Event Logger → API → Cloud Database → Web Dashboard → Filter theo xe/tài xế/ngày → Replay Event → Thống kê cảnh báo.

---

## 9. Yêu cầu phi chức năng

### 9.1. Hiệu năng (Performance)

| Yêu cầu | Chỉ tiêu | Ưu tiên |
|---|---|---|
| FPS | 15–30 FPS ổn định trong mọi điều kiện | P0 |
| End-to-end latency | < 200ms cho DMS | P0 |
| Face detection | < 100ms | P0 |
| Alert response | < 1s sau khi đủ điều kiện | P0 |
| Boot time | < 30s | P1 |
| Model load time | < 5s | P2 |

### 9.2. Độ chính xác (Accuracy)

| Yêu cầu | Chỉ tiêu | Ưu tiên |
|---|---|---|
| DMS - Face Detection | > 95% Precision | P0 |
| Eye State | > 90% Accuracy | P0 |
| Drowsiness Recall | > 90% | P0 |
| PERCLOS False positive | < 5% | P0 |
| Yawn Detection Recall | > 85% | P1 |
| Head Nod Detection Recall | > 85% | P1 |
| Critical Alert | False negative được ưu tiên giảm tối đa | P0 |

### 9.3. Độ tin cậy / ổn định (Reliability)

| Yêu cầu | Chỉ tiêu | Ưu tiên |
|---|---|---|
| System uptime | > 99% trong 2 giờ | P0 |
| Watchdog | Tự động restart khi module treo | P1 |
| Recovery | Khôi phục mà không mất Event Log | P1 |
| Memory leak | Không có memory leak sau 2 giờ | P0 |
| Temperature | < 75°C | P0 |
| Camera failure | Phải phát hiện mất camera | P0 |

### 9.4. Bảo mật (Security)

| Yêu cầu | Chỉ tiêu | Ưu tiên |
|---|---|---|
| Data encryption | Mã hóa dữ liệu truyền và lưu trữ | P0 |
| API security | Authentication | P0 |
| User authorization | Phân quyền Dashboard | P1 |
| Event integrity | Phát hiện chỉnh sửa Event Log | P1 |
| Secure boot | Khởi động an toàn | P2 |

### 9.5. Khả năng mở rộng (Scalability)

| Yêu cầu | Chỉ tiêu | Ưu tiên |
|---|---|---|
| Number of vehicles | Hỗ trợ tối thiểu 100 xe trên Cloud | P1 |
| Concurrent users | 50+ người dùng | P1 |
| Data retention | Log local và Cloud theo chính sách | P1 |
| Model update | OTA | P2 |
| Future sensor integration | Hỗ trợ mở rộng camera 360 và sensor fusion | P2 |

### 9.6. Khả năng sử dụng (Usability)

| Yêu cầu | Chỉ tiêu | Ưu tiên |
|---|---|---|
| HUD readability | Font ≥ 24pt, dark mode | P0 |
| Alert clarity | Các cấp cảnh báo dễ phân biệt | P0 |
| Web App UX | Load < 2s | P1 |
| Installation | Lắp đặt và hiệu chỉnh đơn giản | P1 |

### 9.7. Bảo trì và khả năng phục vụ (Maintainability)

| Yêu cầu | Chỉ tiêu | Ưu tiên |
|---|---|---|
| Logging | Log có cấu trúc | P0 |
| Monitoring | CPU, RAM, GPU, nhiệt độ | P0 |
| Update mechanism | OTA | P2 |
| Documentation | Architecture, API, Installation, Troubleshooting | P0 |

### 9.8. Tuân thủ (Compliance)

| Yêu cầu | Chỉ tiêu | Ưu tiên |
|---|---|---|
| Camera privacy | Tuân thủ yêu cầu bảo mật hình ảnh tài xế | P0 |
| Device regulation | Tuân thủ quy định thiết bị giám sát liên quan | P0 |
| Data protection | Giới hạn truy cập và thời gian lưu dữ liệu | P0 |
| Future vehicle control | Phải đánh giá tiêu chuẩn an toàn trước khi tích hợp điều khiển phương tiện | P0 |

---

## 10. Yêu cầu kỹ thuật tham chiếu

### 10.1. Phần cứng

| Thành phần | Thông số |
|---|---|
| Edge Device | NVIDIA Jetson Nano / TX2 / Orin Nano |
| GPU | CUDA, cuDNN, TensorRT |
| Python | Python 3.10 |
| Camera cabin | Camera hồng ngoại IR |
| Speaker | Cảnh báo âm thanh |
| GPS | U-blox NEO-6M hoặc SIM7600 |
| Tản nhiệt | Heatsink + quạt, giới hạn TDP mode |
| Display | HUD hoặc màn hình cabin |

### 10.2. Mô hình AI

| Module | Mô hình | Framework | Output |
|---|---|---|---|
| DMS - Face | YOLO11n/s hoặc Face Detector | Ultralytics | Face Detection |
| Face Landmark | MediaPipe Face Mesh / tương đương | MediaPipe | Facial Landmarks |
| DMS - EAR | Custom | OpenCV + NumPy | EAR |
| DMS - PERCLOS | Temporal Rule | Python | PERCLOS |
| DMS - Yawn | MAR / Classifier | OpenCV / PyTorch | Yawn State |
| DMS - Head Pose | PnP / Landmark | OpenCV | Pitch / Yaw / Roll |
| DMS - Head Nod | Temporal Analysis | Python / PyTorch | Head Nod Event |
| Drowsiness Score | Rule Engine / Temporal Model | Python / PyTorch | Driver State |

### 10.3. Pipeline và Deployment

| Thành phần | Công nghệ |
|---|---|
| Pipeline | Threading / Multiprocessing |
| Inference | ONNX → TensorRT INT8 |
| Camera | GStreamer pipeline |
| Image Processing | OpenCV |
| Database | SQLite local, Cloud DB |
| API | FastAPI |
| UI | React |
| Deployment | Docker / systemd |
| Version | Git tagging |
| Auto-start | systemd + watchdog |

### 10.4. Dataset

| Dataset | Số lượng | Điều kiện | Annotation |
|---|---|---|---|
| DMS Face/Eye | 2.000+ ảnh/clip | Ngày/đêm | Face, eye |
| Eye State | TBD | Open/Closed | Eye state |
| Yawn | TBD | Normal/Yawn | Mouth state |
| Head Pose | TBD | Đa góc đầu | Pitch/Yaw/Roll |
| Drowsiness Video | TBD | Alert/Fatigue/Drowsy | Temporal labels |
| Calibration INT8 | 500 ảnh | Đa dạng | Representative |
| Test DMS | 1.000 clip | Đa điều kiện | Validation |

### 10.5. Tối ưu hiệu năng

| Chỉ tiêu | Target | Phương pháp |
|---|---|---|
| FPS | 15-30 FPS | TensorRT INT8 |
| Latency | < 200ms | Async processing |
| Memory | < 4GB RAM | Model pruning, INT8 |
| CPU | < 80% | Thread optimization |
| GPU | < 90% | TensorRT |
| Temperature | < 75°C | Heatsink, fan, TDP limit |

---

## 11. Giả định và ràng buộc

### 11.1. Giả định

| # | Giả định | Rủi ro nếu sai |
|---|---|---|
| A1 | Camera IR quan sát rõ khuôn mặt ngày và đêm | EAR/PERCLOS không chính xác |
| A2 | Camera được lắp đúng góc | Landmark bị mất |
| A3 | Kết nối mạng có thể gián đoạn | Cần lưu local |
| A4 | Nguồn xe ổn định | Hệ thống có thể restart |
| A5 | Dataset đủ đa dạng | Mô hình có thể overfit |
| A6 | Tài xế không cố tình che camera | Cần camera obstruction detection |
| A7 | Một camera cabin đủ cho MVP | Có thể cần camera bổ sung ở Phase sau |

### 11.2. Ràng buộc

| # | Ràng buộc | Mô tả |
|---|---|---|
| C1 | Phần cứng Edge giới hạn | Phải tối ưu model |
| C2 | Nhiệt độ | Bắt buộc tản nhiệt |
| C3 | Góc camera | Có ảnh hưởng lớn đến EAR |
| C4 | Kính mắt | Có thể làm giảm độ chính xác Eye Detection |
| C5 | Ánh sáng | Cần IR hoặc low-light camera |
| C6 | Thời gian MVP | Ưu tiên P0 |
| C7 | Điều khiển phương tiện | Không thuộc MVP |

### 11.3. Ràng buộc dữ liệu

| # | Ràng buộc | Mô tả |
|---|---|---|
| D1 | Dataset phải có người tỉnh táo và buồn ngủ | Đảm bảo đủ positive/negative samples |
| D2 | Phải có dữ liệu ngày và đêm | Đánh giá camera IR |
| D3 | Phải có nhiều người khác nhau | Hạn chế overfit khuôn mặt |
| D4 | Phải có blink bình thường | Giảm false positive |
| D5 | Phải có ngáp và gục đầu | Đánh giá multi-feature |
| D6 | Test set không trùng người với train nếu có thể | Đánh giá generalization |

---

## 12. Rủi ro và giải pháp tham chiếu

### 12.1. Rủi ro kỹ thuật

| ID | Rủi ro | Xác suất | Tác động | Giải pháp |
|---|---|---|---|---|
| R1 | Jetson không đạt FPS | Medium | High | TensorRT INT8 |
| R2 | Camera IR không rõ mắt | Medium | High | Hiệu chỉnh camera |
| R3 | Kính mắt gây sai EAR | High | Medium | Kết hợp nhiều feature |
| R4 | Một frame lỗi gây cảnh báo | Medium | High | Temporal smoothing |
| R5 | Pipeline crash | Low | High | Watchdog |
| R6 | Camera bị che | Medium | Medium | Obstruction Detection |
| R7 | Cảnh báo quá nhiều | Medium | High | Multi-level alert + temporal filtering |

### 12.2. Rủi ro dữ liệu

| ID | Rủi ro | Xác suất | Tác động | Giải pháp |
|---|---|---|---|---|
| R8 | Không đủ dữ liệu buồn ngủ thật | High | High | Dataset public + dữ liệu mô phỏng có kiểm soát |
| R9 | Dataset lệch điều kiện ánh sáng | Medium | High | Bổ sung IR/night data |
| R10 | Annotation không đồng nhất | Medium | Medium | Annotation guideline |
| R11 | Overfitting | Medium | High | Cross-person validation |
| R12 | Blink bị nhầm với ngủ | Medium | High | Temporal feature + PERCLOS |

### 12.3. Rủi ro pháp lý và tuân thủ

| ID | Rủi ro | Xác suất | Tác động | Giải pháp |
|---|---|---|---|---|
| R13 | Vi phạm quyền riêng tư tài xế | Medium | High | Phân quyền và mã hóa |
| R14 | Lưu video quá lâu | Medium | Medium | Chính sách retention |
| R15 | Tính năng tự lái tương lai chưa đáp ứng quy định | High | High | Chỉ phát triển sau đánh giá pháp lý và an toàn |

### 12.4. Rủi ro dự án và vận hành

| ID | Rủi ro | Xác suất | Tác động | Giải pháp |
|---|---|---|---|---|
| R16 | Chậm tiến độ | Medium | High | Chỉ tập trung DMS MVP |
| R17 | Thiếu kinh nghiệm TensorRT | Medium | Medium | Prototype sớm |
| R18 | Camera lắp sai vị trí | Medium | High | Calibration guideline |
| R19 | Tài xế che camera | Medium | Medium | Camera obstruction alert |

### 12.5. Rủi ro kinh doanh

| ID | Rủi ro | Xác suất | Tác động | Giải pháp |
|---|---|---|---|---|
| R20 | Cảnh báo sai khiến tài xế khó chịu | Medium | High | Tune threshold và cảnh báo tăng dần |
| R21 | Khách hàng không thấy giá trị | Medium | High | Dashboard và báo cáo sự kiện |
| R22 | Đối thủ có DMS tương tự | Medium | Medium | Tập trung khả năng mở rộng Emergency Assist |

---

## 13. Release plan - Roadmap

### Phase 0: Nghiên cứu và thiết kế

| Tuần | Milestone | Deliverable |
|---|---|---|
| 1 | Khảo sát bài toán buồn ngủ | PRD hoàn chỉnh |
| 2 | Thiết kế kiến trúc DMS | System Architecture |
| 3 | Thiết lập Edge Device | Jetson + Python + PyTorch |
| 4 | Chuẩn bị dataset | DMS Dataset |

### Phase 1: Phát triển Module

| Tuần | Module | Milestone |
|---|---|---|
| 5 | Face/Eye | Face Detection + Landmark |
| 5 | EAR | Eye State + Blink Detection |
| 5 | PERCLOS | Temporal PERCLOS |
| 6 | Yawn | Yawn Detection |
| 6 | Head Pose | Head Pose + Head Nod |
| 6 | Drowsiness | Drowsiness Scoring Engine |

### Phase 2: Tích hợp và tối ưu

| Tuần | Milestone | Deliverable |
|---|---|---|
| 7 | Ghép DMS pipeline | End-to-End Pipeline |
| 7 | Alert Manager | Cảnh báo nhiều cấp |
| 7 | TensorRT | 15-30 FPS |
| 7 | UI Dashboard | HUD + Web |
| 7 | Event Logger | Log cảnh báo |

### Phase 3: Kiểm thử

| Tuần | Milestone | Deliverable |
|---|---|---|
| 8 | Test Eye State | Drowsiness window |
| 8 | Test PERCLOS | Drowsiness window |
| 8 | Test Yawn | Normal / Yawn |
| 8 | Test Head Nod | Normal / Nod |
| 8 | Test End-to-End | Drowsiness → Alert |
| 8 | False Positive Test | Tune thresholds |

### Phase 4: Triển khai

| Tuần | Milestone | Deliverable |
|---|---|---|
| 9 | Đóng gói phần mềm | MVP v1.0 |
| 9 | Lắp đặt 2-3 xe | Test thực tế |
| 9 | Hiệu chỉnh | Camera + threshold |
| 9 | Bàn giao | Technical Documentation |

### Phase 5: Future - 360 Perception

| Milestone | Deliverable |
|---|---|
| Camera 360 | Camera trước / sau / trái / phải |
| Lane Detection | Nhận diện làn đường |
| Object Detection | Xe, xe máy, người đi bộ |
| Blind Spot Detection | Kiểm tra điểm mù |
| Sensor Fusion | Camera + Radar + GPS + IMU |
| Environment Model | Nhận biết môi trường xung quanh xe |

### Phase 6: Future - Driver Emergency Assistance

| Milestone | Deliverable |
|---|---|
| Driver No-response Detection | Xác nhận tài xế không còn phản hồi |
| Lane Keeping | Duy trì phương tiện trong làn hiện tại |
| Controlled Deceleration | Giảm tốc có kiểm soát |
| Hazard Light | Bật đèn cảnh báo |
| Emergency Notification | Gửi GPS và cảnh báo |

### Phase 7: Future - Minimal Risk Maneuver

| Milestone | Deliverable |
|---|---|
| Safe Lane Assessment | Đánh giá làn bên phải |
| Blind Spot Verification | Kiểm tra xe phía sau và điểm mù |
| Controlled Lane Change | Chuyển làn khi đủ điều kiện an toàn |
| Safe Stop | Dừng phương tiện có kiểm soát |
| Parking Brake | Kích hoạt phanh đỗ |
| Emergency Report | Gửi vị trí và trạng thái xe |

---

## 14. Phụ lục - Thuật ngữ (Glossary)

| Thuật ngữ | Viết tắt | Giải thích |
|---|---|---|
| DMS | Driver Monitoring System | Hệ thống giám sát tài xế |
| EAR | Eye Aspect Ratio | Chỉ số hình học hỗ trợ xác định trạng thái mắt |
| PERCLOS | Percentage of Eye Closure | Tỷ lệ thời gian mắt ở trạng thái đóng trong một cửa sổ thời gian |
| MAR | Mouth Aspect Ratio | Chỉ số hình học hỗ trợ phát hiện há miệng/ngáp |
| Microsleep | - | Trạng thái ngủ rất ngắn trong khi tài xế mất tỉnh táo |
| Head Pose | - | Hướng và góc đầu của tài xế |
| Pitch | - | Góc cúi/ngẩng đầu |
| Yaw | - | Góc quay đầu trái/phải |
| Roll | - | Góc nghiêng đầu |
| FPS | Frames Per Second | Số khung hình mỗi giây |
| IR | Infrared | Camera hồng ngoại |
| GPS | Global Positioning System | Hệ thống định vị |
| HUD | Head-Up Display | Cảnh báo lệch làn đường |
| API | Application Programming Interface | Giao diện lập trình ứng dụng |
| CUDA | Compute Unified Device Architecture | Nền tảng tính toán NVIDIA |
| TensorRT | - | SDK tối ưu inference NVIDIA |
| ONNX | Open Neural Network Exchange | Định dạng trao đổi mô hình |
| MVP | Minimum Viable Product | Sản phẩm tối thiểu khả dụng |
| NTP | Network Time Protocol | Đồng bộ thời gian |
| OTA | Over-the-Air | Cập nhật phần mềm/mô hình từ xa |
| ADAS | Advanced Driver Assistance Systems | Hệ thống hỗ trợ lái nâng cao |
| LKA | Lane Keeping Assist | Hỗ trợ duy trì làn |
| Sensor Fusion | - | Kết hợp dữ liệu từ nhiều cảm biến |
| Drowsiness Score | - | Điểm tổng hợp đánh giá mức độ buồn ngủ |
| Minimal Risk Maneuver | MRM | Quy trình đưa phương tiện về trạng thái rủi ro tối thiểu khi tài xế không thể tiếp tục điều khiển |
