# Task Plan — Chiến (Frontend, UI/UX & Dashboard)

**Project:** GuardianPilot360 MVP v1.0  
**Focus:** HUD Dashboard, Web App quản lý đội xe, UI overhaul, Alert UX  
**Timeline:** Phase 1-3 (4-6 weeks)

---

## Context từ PRD

Theo PRD_GuardianPilot360.md, UI/UX cần:
- **HUD Dashboard trên xe**: dark mode, font ≥24pt, 4 màu cảnh báo (UI-01 → UI-04)
- **Alert 5 cấp**: NORMAL/FATIGUE/DROWSY/MICROSLEEP/CRITICAL với màu sắc và âm thanh
- **Web App quản lý đội xe**: live status, replay log, thống kê theo xe/tài xế/ngày (UI-05, UI-06)
- **Yawn alarm** hiển thị trên UI (hiện tại backend có nhưng frontend chưa show)
- **PERCLOS gauge** trên dashboard
- **Mobile responsive** (UI-05, giao diện mobile friendly)

---

## Gap Analysis — Những gì Frontend đang thiếu so với PRD

| PRD Requirement | Trạng thái | Priority |
|---|---|---|
| 5 driver states (NORMAL/FATIGUE/DROWSY/MICROSLEEP/CRITICAL) | ❌ Thiếu — chỉ có NORMAL/DROWSY | P0 |
| PERCLOS gauge display | ❌ Thiếu hoàn toàn | P0 |
| Yawn alarm banner (UI) | ⚠ Backend có, frontend chưa hiển thị rõ | P0 |
| 4-level color coding (xanh/vàng/cam/đỏ) | ⚠ Chỉ có xanh/đỏ | P0 |
| Audio alert (4 cấp) | ❌ Thiếu hoàn toàn | P0 |
| HUD font ≥ 24pt rõ ràng | ⚠ Font hiện tại nhỏ | P0 |
| Web App quản lý đội xe | ❌ Thiếu hoàn toàn | P1 |
| Event Log replay page | ❌ Thiếu hoàn toàn | P1 |
| Thống kê theo xe/tài xế/ngày | ❌ Thiếu hoàn toàn | P1 |
| Mobile responsive UI | ⚠ Chỉ có media query cơ bản | P1 |
| PERCLOS timeline chart | ❌ Thiếu hoàn toàn | P2 |
| Camera obstruction warning | ❌ Thiếu hoàn toàn | P2 |

---

## Task List

### 🔴 P0 — CRITICAL (MVP blocker)

#### **Task 1: 5-Level Alert State Display**
**Priority:** P0  
**Effort:** 2-3 days  
**Files:** `web/templates/index.html`, `web/static/css/style.css`, `web/static/js/app.js`

**Current state:**  
- Alert card chỉ có 2 trạng thái: NORMAL (xanh) và DROWSY (đỏ)
- PRD yêu cầu 5 states với màu sắc khác nhau

**Requirements:**
1. 5 trạng thái + màu chuẩn PRD:
   | State | Màu | Hex |
   |---|---|---|
   | NORMAL | Xanh lá | `#5c9e6e` |
   | FATIGUE | Vàng | `#c49a3c` |
   | DROWSY | Cam | `#d97757` |
   | MICROSLEEP | Đỏ | `#c04545` |
   | CRITICAL | Đỏ pulse + flash | `#c04545` + animation |

2. CSS variables cần thêm:
   ```css
   --fatigue:    #c49a3c;
   --drowsy:     #d97757;
   --microsleep: #c04545;
   --critical:   #c04545;
   ```

3. Update `alertCard` CSS `data-state` attribute:
   - `data-state="normal"`, `"fatigue"`, `"drowsy"`, `"microsleep"`, `"critical"`
   - Thay `data-alarm="on/off"` thành `data-state=<string>`

4. Update `applyResult()` trong `app.js`:
   ```js
   function getDriverState(data) {
       // Backend Task 1 (Long) sẽ trả về data.driver_state
       // Fallback mapping nếu backend chưa có:
       if (data.alarm_on) {
           if (data.ema_prob >= 0.85) return "microsleep";
           if (data.ema_prob >= 0.65) return "drowsy";
           return "fatigue";
       }
       return "normal";
   }
   ```

5. Alert label font size tăng lên 2rem (PRD: font ≥ 24pt):
   ```css
   .alert-label { font-size: 2rem; font-weight: 800; letter-spacing: 0.1em; }
   ```

**Acceptance Criteria:**
- ✅ 5 màu hiển thị đúng theo state
- ✅ CRITICAL có animation flash (đỏ nhấp nháy nhanh hơn)
- ✅ Alert label ≥ 24pt (2rem)
- ✅ Transition mượt giữa các state

**Dependencies:** None (backend fallback đã handle)  
**Blocks:** Task 2 (Audio), Task 3 (PERCLOS)

---

#### **Task 2: Audio Alert System (4 cấp)**
**Priority:** P0  
**Effort:** 1-2 days  
**Files:** `web/static/js/app.js`, `web/static/audio/` (new folder)

**Current state:**  
- Không có âm thanh cảnh báo
- PRD yêu cầu cảnh báo âm thanh nhiều mức (SYS-02)

**Requirements:**
1. Audio files (generated bằng Web Audio API — không cần file tải về):
   ```
   Level 1 (FATIGUE):    1 beep nhẹ, 440Hz, 0.3s
   Level 2 (DROWSY):     2 beep liên tiếp, 660Hz, 0.2s×2
   Level 3 (MICROSLEEP): 3 beep nhanh, 880Hz, 0.15s×3
   Level 4 (CRITICAL):   Alarm liên tục, 1100Hz, 0.1s interval
   ```

2. Implement `AudioAlertManager`:
   ```js
   class AudioAlertManager {
       constructor() {
           this.ctx = new AudioContext();
           this.currentLevel = 0;
           this.criticalTimer = null;
       }
       
       setLevel(state) {
           const levelMap = {
               "normal": 0, "fatigue": 1,
               "drowsy": 2, "microsleep": 3, "critical": 4
           };
           const level = levelMap[state] || 0;
           if (level === this.currentLevel) return;
           this.currentLevel = level;
           this._playForLevel(level);
       }
       
       _beep(freq, duration, delay=0) {
           // Tạo oscillator + gain envelope
       }
       
       _playForLevel(level) {
           // Level 0: silence
           // Level 1: 1 beep
           // Level 2: 2 beeps
           // Level 3: 3 beeps nhanh
           // Level 4: continuous alarm loop
       }
   }
   ```

3. Mute button:
   ```html
   <button id="btnMute" class="btn btn-sm" title="Tắt cảnh báo âm thanh">🔔</button>
   ```

4. Browser permission:
   - AudioContext cần user gesture để unlock
   - Unlock khi user click "Bật webcam" hoặc "Khởi tạo"

**Acceptance Criteria:**
- ✅ 4 cấp âm thanh khác nhau rõ ràng
- ✅ CRITICAL loop alarm cho đến khi state về bình thường
- ✅ Mute button hoạt động
- ✅ Không autoplay khi chưa có user interaction

**Dependencies:** Task 1 (cần driver state)  
**Blocks:** None

---

#### **Task 3: PERCLOS Gauge Display**
**Priority:** P0  
**Effort:** 1-2 days  
**Files:** `web/templates/index.html`, `web/static/css/style.css`, `web/static/js/app.js`

**Current state:**  
- Không có PERCLOS display trên UI
- Backend sẽ có sau khi Long hoàn thành Task 5

**Requirements:**
1. PERCLOS gauge trong result panel:
   ```html
   <div class="perclos-gauge">
       <div class="gauge-label">PERCLOS (30s)</div>
       <div class="gauge-bar">
           <div class="gauge-fill" id="perclosBar"></div>
       </div>
       <div class="gauge-value" id="perclosValue">—</div>
   </div>
   ```

2. Color thresholds:
   - 0-30%: xanh (`--normal`)
   - 30-50%: vàng (`--fatigue`)
   - 50-70%: cam (`--drowsy`)
   - >70%: đỏ (`--microsleep`)

3. CSS gauge:
   ```css
   .perclos-gauge { margin-top: 1rem; }
   .gauge-bar { height: 8px; background: var(--surface-3); border-radius: 4px; overflow: hidden; }
   .gauge-fill { height: 100%; transition: width 0.5s ease, background 0.3s ease; }
   ```

4. Fallback khi backend chưa có:
   - Nếu `data.perclos` undefined → ước tính từ `eye_closed_streak_ms` / 30000

**Acceptance Criteria:**
- ✅ PERCLOS bar fill đúng màu theo ngưỡng
- ✅ Update smooth (transition 0.5s)
- ✅ Hiển thị % (e.g. "42%")
- ✅ Fallback hoạt động khi backend chưa return perclos

**Dependencies:** None  
**Blocks:** None

---

#### **Task 4: Yawn Banner + Full Alarm Panel**
**Priority:** P0  
**Effort:** 1 day  
**Files:** `web/templates/index.html`, `web/static/css/style.css`, `web/static/js/app.js`

**Current state:**  
- Backend `fusion.py` đã có `yawn_alarm: true/false`
- `app.js` có `data.yawn_alarm` nhưng **không có banner trên UI** (`eyeBanner` và `neckBanner` có, `yawnBanner` không có!)
- Check: tìm trong `index.html` → không có `yawnAlarmBanner` element

**Requirements:**
1. Thêm Yawn banner vào `index.html`:
   ```html
   <div id="yawnAlarmBanner" class="alarm-banner yawn-banner hidden">
       <span class="banner-icon">😮</span>
       Yawn detected — ngáp phát hiện
   </div>
   ```

2. CSS yawn banner:
   ```css
   .yawn-banner {
       background: rgba(100, 180, 255, 0.15);
       border-color: rgba(100, 180, 255, 0.4);
       color: #64b4ff;
   }
   ```

3. Update `applyResult()` trong `app.js`:
   ```js
   const yawnBanner = document.getElementById('yawnAlarmBanner');
   if (data.yawn_alarm) yawnBanner.classList.remove('hidden');
   else yawnBanner.classList.add('hidden');
   ```

4. Unify alarm banners thành `AlarmPanel` component:
   ```html
   <div class="alarm-panel">
       <div id="neckAlarmBanner" class="alarm-banner neck-banner hidden">
           <span class="banner-icon">↕</span> Neck-tilt alarm
       </div>
       <div id="eyeAlarmBanner" class="alarm-banner eye-banner hidden">
           <span class="banner-icon">👁</span> Eye-closure alarm
       </div>
       <div id="yawnAlarmBanner" class="alarm-banner yawn-banner hidden">
           <span class="banner-icon">😮</span> Yawn detected
       </div>
   </div>
   ```

**Acceptance Criteria:**
- ✅ Yawn banner hiển thị khi `yawn_alarm=true`
- ✅ 3 banners (neck, eye, yawn) được group lại đẹp
- ✅ Icons giúp phân biệt nhanh
- ✅ Tất cả banners hideable/showable độc lập

**Dependencies:** None  
**Blocks:** None

---

#### **Task 5: HUD Mode (In-Car Display)**
**Priority:** P0  
**Effort:** 2-3 days  
**Files:** `web/templates/index.html`, `web/static/css/style.css`, `web/static/js/app.js`

**Current state:**  
- UI hiện tại là developer UI, không phù hợp hiển thị trên màn hình trong xe
- PRD yêu cầu HUD: dark mode, font lớn ≥24pt, dễ đọc khi lái xe

**Requirements:**
1. HUD mode toggle button:
   ```html
   <button id="btnHudMode" class="btn btn-sm" title="Chế độ HUD">⬛ HUD</button>
   ```

2. HUD layout (`body.hud-mode`):
   - Chỉ hiển thị: trạng thái driver, prob, PERCLOS, 3 banners
   - Font rất lớn (≥ 2.5rem cho status, 1.5rem cho số liệu)
   - Màu nền tối hoàn toàn `#0a0a0a`
   - Không hiển thị: feature grid, buttons, tabs, timeline

   ```css
   body.hud-mode .topbar { display: none; }
   body.hud-mode .input-panel { display: none; }
   body.hud-mode .layout { display: block; max-width: 100%; padding: 0; }
   body.hud-mode .result-panel {
       position: fixed; inset: 0;
       display: flex; flex-direction: column;
       justify-content: center; align-items: center;
       background: #0a0a0a;
       border: none; border-radius: 0;
   }
   body.hud-mode .alert-label { font-size: 4rem; }
   body.hud-mode .alert-prob  { font-size: 2rem; }
   body.hud-mode .section-title { font-size: 1rem; }
   body.hud-mode .feature-grid { display: none; }
   body.hud-mode .alarm-panel { width: 100%; max-width: 600px; }
   body.hud-mode .alarm-banner { font-size: 1.5rem; padding: 1rem 1.5rem; }
   body.hud-mode .perclos-gauge { width: 100%; max-width: 600px; font-size: 1.2rem; }
   
   /* Exit HUD button */
   body.hud-mode #hudExitBtn { display: flex; }
   #hudExitBtn { display: none; }
   ```

3. HUD exit button (fixed top-right):
   ```html
   <button id="hudExitBtn" class="btn btn-sm" 
           style="position:fixed;top:1rem;right:1rem;z-index:999">✕ Exit HUD</button>
   ```

4. Persist HUD preference in `localStorage`:
   ```js
   const HUD_KEY = 'guardian_hud_mode';
   function toggleHudMode() {
       document.body.classList.toggle('hud-mode');
       localStorage.setItem(HUD_KEY, document.body.classList.contains('hud-mode') ? '1' : '0');
   }
   // On load:
   if (localStorage.getItem(HUD_KEY) === '1') document.body.classList.add('hud-mode');
   ```

**Acceptance Criteria:**
- ✅ HUD mode: chỉ thấy trạng thái, font rất lớn, background cực tối
- ✅ HUD mode toggle hoạt động mượt
- ✅ Exit HUD nhanh (1 click)
- ✅ Font status ≥ 4rem, dễ đọc từ xa
- ✅ Preference được lưu (F5 không mất HUD mode)

**Dependencies:** Task 1 (5 states), Task 3 (PERCLOS), Task 4 (Yawn banner)  
**Blocks:** None

---

### 🟡 P1 — IMPORTANT (Post-MVP priority)

#### **Task 6: Web App Quản Lý Đội Xe (Fleet Dashboard)**
**Priority:** P1  
**Effort:** 5-7 days  
**Files:** NEW `web/templates/dashboard.html`, `web/static/js/dashboard.js`, `web/static/css/dashboard.css`

**Current state:**  
- Không có trang quản lý đội xe
- PRD UI-05: "Web App quản lý đội xe: live feed, replay log, thống kê"

**Requirements:**
1. New route `/dashboard`:
   ```python
   # app.py
   @app.route("/dashboard")
   def dashboard():
       return render_template("dashboard.html")
   ```

2. Dashboard layout (`dashboard.html`):
   ```
   ┌─────────────────────────────────────────────┐
   │  GuardianPilot360  [Fleet Dashboard]   🚗 12 │
   ├─────────────────────────────────────────────┤
   │  Active Alerts  │  Today's Stats             │
   │  🔴 Xe 001 - DROWSY   │  Total alerts: 23   │
   │  🟡 Xe 003 - FATIGUE  │  CRITICAL events: 2 │
   │                       │  Avg PERCLOS: 18%    │
   ├─────────────────────────────────────────────┤
   │  Event Log                                   │
   │  [Filter: Xe ▾] [Driver ▾] [Date ▾] [State ▾]│
   │  ┌──────────────────────────────────────────┐│
   │  │ Time     │ Xe  │ State    │ PERCLOS │ GPS ││
   │  │ 10:32:15 │ 001 │ DROWSY   │  72%    │ 📍  ││
   │  │ 10:28:03 │ 003 │ FATIGUE  │  45%    │ 📍  ││
   │  └──────────────────────────────────────────┘│
   └─────────────────────────────────────────────┘
   ```

3. API endpoints cần (Long sẽ làm, Chiến consumes):
   - `GET /api/events?since=<ts>&state=<level>&limit=100`
   - `GET /api/stats?date=<YYYY-MM-DD>`

4. JavaScript:
   ```js
   // dashboard.js
   async function loadEvents(filters) {
       const params = new URLSearchParams(filters);
       const res = await fetch(`/api/events?${params}`);
       const data = await res.json();
       renderEventTable(data.events);
   }
   
   function renderEventTable(events) {
       // Render table với snapshot preview on hover
   }
   
   // Auto-refresh mỗi 10s
   setInterval(() => loadEvents(currentFilters), 10000);
   ```

5. Style: dark mode, consistent với main UI

**Acceptance Criteria:**
- ✅ Event table load <2s
- ✅ Filter theo xe, tài xế, ngày, state hoạt động
- ✅ Auto-refresh mỗi 10s
- ✅ Snapshot preview on row hover
- ✅ GPS link (click → mở Google Maps)

**Dependencies:** Long Task 2 (EventLogger + GET /api/events API)  
**Blocks:** Task 7 (Event Replay)

---

#### **Task 7: Event Replay Player**
**Priority:** P1  
**Effort:** 3-4 days  
**Files:** `web/templates/dashboard.html`, `web/static/js/dashboard.js`

**Current state:**  
- PRD UI-06: "Replay Event Log"
- Không có replay functionality

**Requirements:**
1. Event detail modal khi click vào row:
   ```html
   <div class="event-modal" id="eventModal">
       <div class="modal-header">
           <h3>Event Detail — 10:32:15 — Xe 001</h3>
           <button class="close-btn">✕</button>
       </div>
       <div class="modal-body">
           <div class="snapshot-view">
               <img id="snapshotImg" src="" alt="Snapshot" />
               <div class="snapshot-overlay" id="snapshotOverlay">
                   <!-- overlay info: EAR, MAR, state... -->
               </div>
           </div>
           <div class="event-stats">
               <!-- feature values at time of event -->
           </div>
       </div>
   </div>
   ```

2. Replay timeline (nếu có nhiều events liên tiếp):
   - Slider để seek qua sequence of events
   - Play/Pause button

3. Snapshot display:
   - Load snapshot từ server: `GET /api/snapshot/<event_id>`
   - Fallback: hiển thị placeholder nếu snapshot không có

**Acceptance Criteria:**
- ✅ Modal mở khi click event row
- ✅ Snapshot hiển thị nếu có
- ✅ Feature values hiển thị đúng (EAR, PERCLOS, state...)
- ✅ Close modal khi click outside hoặc ESC

**Dependencies:** Task 6, Long Task 2 (snapshot API)  
**Blocks:** None

---

#### **Task 8: Stats & Analytics Page**
**Priority:** P1  
**Effort:** 3-4 days  
**Files:** `web/templates/dashboard.html`, `web/static/js/dashboard.js`

**Requirements:**
1. Daily stats card:
   ```
   Today (Aug 7):
   - Total drives: 5
   - Total alerts: 12
   - Critical events: 1
   - Avg PERCLOS: 22%
   - Most common state: FATIGUE (60%)
   ```

2. Simple bar chart (dùng Canvas API, không cần thư viện):
   - Alerts per hour (0-23h)
   - State distribution (pie chart)

3. Per-driver stats (nếu có multi-driver):
   - Top 3 alerting drivers
   - Average fatigue score per driver

**Acceptance Criteria:**
- ✅ Stats hiển thị ngay khi load page
- ✅ Charts đơn giản, dễ đọc, dùng Canvas native
- ✅ Responsive trên mobile
- ✅ Không cần thư viện Chart.js/D3 (keep it simple)

**Dependencies:** Long Task 2 (GET /api/stats endpoint)  
**Blocks:** None

---

### 🟢 P2 — NICE TO HAVE

#### **Task 9: Mobile-Responsive Improvements**
**Priority:** P2  
**Effort:** 1-2 days  
**Files:** `web/static/css/style.css`

**Requirements:**
1. Better mobile breakpoints:
   - `≤ 600px`: single column, stacked layout
   - `≤ 900px`: current 2-col collapse
   - Touch-friendly buttons (min 44px touch target)

2. Mobile-specific HUD improvements:
   - Full-screen mode on mobile (landscape)
   - Swipe to toggle HUD mode

**Acceptance Criteria:**
- ✅ Usable trên iOS/Android browser
- ✅ Buttons đủ lớn để tap
- ✅ HUD mode works trên mobile landscape

---

#### **Task 10: Camera Obstruction Warning**
**Priority:** P2  
**Effort:** 1 day  
**Files:** `web/static/js/app.js`

**Current state:**  
- PRD R6: camera bị che → phát hiện và cảnh báo

**Requirements:**
1. Frontend detection:
   - Khi `face_found=false` liên tục >10s → hiển thị "Camera bị che hoặc mất kết nối"
   - Banner với màu vàng (warning, không phải drowsiness alarm)

2. Backend support (phối hợp với Long):
   - Backend trả về `camera_status: "ok" | "obstructed" | "low_light"`

**Acceptance Criteria:**
- ✅ Banner hiển thị sau 10s không detect face
- ✅ Banner tự ẩn khi face detected lại
- ✅ Phân biệt "camera bị che" vs "không có người"

---

## Timeline & Milestones

**Week 1:**
- Task 1: 5-level alert display
- Task 2: Audio alerts
- Task 4: Yawn banner fix

**Week 2:**
- Task 3: PERCLOS gauge
- Task 5: HUD mode

**Week 3-4:**
- Task 6: Fleet dashboard
- Task 7: Event replay

**Week 5-6:**
- Task 8: Stats & analytics
- Task 9: Mobile responsive
- Task 10: Camera obstruction

**Milestones:**
- **M1 (End Week 1):** 5 states + audio + yawn UI fix done
- **M2 (End Week 2):** PERCLOS + HUD mode done
- **M3 (End Week 4):** Fleet dashboard + event replay done
- **M4 (End Week 6):** Full MVP UI complete

---

## API Contract với Long

Chiến cần Long implement các endpoints này (theo thứ tự ưu tiên):

| Endpoint | Priority | Cần cho Task |
|---|---|---|
| `data.driver_state` trong `/api/analyze` response | P0 | Task 1, 2 |
| `data.perclos` trong `/api/analyze` response | P0 | Task 3 |
| `GET /api/events?since&state&limit` | P1 | Task 6 |
| `GET /api/snapshot/<event_id>` | P1 | Task 7 |
| `GET /api/stats?date` | P1 | Task 8 |

---

## Design Guidelines

### Màu sắc theo PRD
```css
:root {
    /* Cảnh báo chuẩn ADAS */
    --state-normal:     #5c9e6e;  /* Xanh lá — an toàn */
    --state-fatigue:    #c49a3c;  /* Vàng    — mệt mỏi */
    --state-drowsy:     #d97757;  /* Cam     — buồn ngủ */
    --state-microsleep: #c04545;  /* Đỏ      — ngủ gật */
    --state-critical:   #c04545;  /* Đỏ + flash — nguy hiểm */
}
```

### Font cho HUD
- Label chính (STATUS): ≥ 4rem, monospace, bold
- Sub-label (p=0.72): ≥ 2rem
- Số liệu (EAR, PERCLOS): ≥ 1.5rem

### Dark Mode
- Background: `#0a0a0a` (HUD), `#1e1c1a` (UI thường)
- Text: `#f0ebe4`
- Không dùng màu sáng trên nền tối

---

## Questions & Blockers

1. **Amplitude audio**: Web Audio API đủ hay cần file `.wav` thật?
2. **Multi-driver**: Dashboard có cần phân biệt nhiều tài xế khác nhau không?
3. **GPS Maps**: Click GPS link → mở Google Maps hay có bản đồ riêng?
4. **Snapshot size**: Mỗi snapshot bao nhiêu KB? Cần lazy load?

---

_Last updated: 2026-08-07_
