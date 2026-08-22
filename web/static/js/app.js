'use strict';

// ── Fix: Worker origin is null → absolute fetch fails CORS. ────────────────
// Empty string = same-origin relative URL works correctly inside a Worker.
const WORKER_API_BASE = '';

// ── Session ID (H2) ────────────────────────────────────────────────────────
// Server giữ FusionState/AlertManager/TripMemory riêng cho từng session_id.
// Trước đây mọi tab dùng chung một state toàn cục: mở hai tab là hai luồng
// nhận diện trộn vào nhau, và "Đặt lại" ở tab này xoá state của tab kia.
//
// Lưu trong sessionStorage → mỗi tab một id, F5 vẫn giữ nguyên phiên.
const SESSION_ID = (() => {
  const KEY = 'guardian_session_id';
  // sessionStorage ném SecurityError khi trình duyệt chặn site-data (private
  // mode, iframe sandbox, cookie bị chặn). Đây là code top-level: một exception
  // ở đây giết cả file app.js → UI trắng. Luôn bọc try/catch.
  let id = null;
  try { id = sessionStorage.getItem(KEY); } catch (_) { /* storage bị chặn */ }
  if (!id) {
    // crypto.randomUUID chỉ có trong secure context (https hoặc localhost).
    // Truy cập qua http://<LAN-IP>:5000 thì nó undefined → dùng fallback.
    id = ((self.crypto && crypto.randomUUID)
      ? crypto.randomUUID()
      : 'sess-' + Date.now() + '-' + Math.random().toString(36).slice(2, 10)
    ).replace(/[^a-zA-Z0-9_.-]/g, '');
    try { sessionStorage.setItem(KEY, id); } catch (_) { /* chạy tiếp, id chỉ sống trong tab */ }
  }
  return id;
})();

/** Header chuẩn cho mọi request tới API (JSON + session). */
function sessionHeaders(extra) {
  return Object.assign(
    { 'Content-Type': 'application/json', 'X-Session-Id': SESSION_ID },
    extra || {},
  );
}

// ── DOM refs ──────────────────────────────────────────────────────────────
const btnInit          = document.getElementById('btnInit');
const systemBadge      = document.getElementById('systemBadge');
const statusBar        = document.getElementById('statusBar');

const btnMute          = document.getElementById('btnMute');
const btnHudMode       = document.getElementById('btnHudMode');
const hudExitBtn       = document.getElementById('hudExitBtn');

const tabButtons       = document.querySelectorAll('.tab');
const tabContents      = document.querySelectorAll('.tab-content');

const webcamEl         = document.getElementById('webcam');
const captureCanvas    = document.getElementById('captureCanvas');
const annotatedCanvas  = document.getElementById('annotatedCanvas');
const videoOverlay     = document.getElementById('videoOverlay');
const overlayStatus    = document.getElementById('overlayStatus');
const overlayProb      = document.getElementById('overlayProb');
const btnStartCam      = document.getElementById('btnStartCam');
const btnAnalyzeLive   = document.getElementById('btnAnalyzeLive');
const btnStopLive      = document.getElementById('btnStopLive');

const dropZone         = document.getElementById('dropZone');
const fileInput        = document.getElementById('fileInput');
const btnPickFile      = document.getElementById('btnPickFile');
const previewImg       = document.getElementById('previewImg');
const btnAnalyzeUpload = document.getElementById('btnAnalyzeUpload');

const videoDropZone    = document.getElementById('videoDropZone');
const videoInput       = document.getElementById('videoInput');
const btnPickVideo     = document.getElementById('btnPickVideo');
const videoFileMeta    = document.getElementById('videoFileMeta');
const videoFileName    = document.getElementById('videoFileName');
const videoFileInfo    = document.getElementById('videoFileInfo');
const videoWorkspace   = document.getElementById('videoWorkspace');
const fileVideo        = document.getElementById('fileVideo');
const videoAnnotatedPreview = document.getElementById('videoAnnotatedPreview');
const videoResultPlaceholder = document.getElementById('videoResultPlaceholder');
const videoLiveBadge   = document.getElementById('videoLiveBadge');
const btnAnalyzeVideo  = document.getElementById('btnAnalyzeVideo');
const btnStopVideo     = document.getElementById('btnStopVideo');
const videoFpsSlider   = document.getElementById('videoFpsSlider');
const videoFpsValue    = document.getElementById('videoFpsValue');
const videoProgressWrap= document.getElementById('videoProgressWrap');
const videoProgressBar = document.getElementById('videoProgressBar');
const videoProgressText= document.getElementById('videoProgressText');
const videoProcessingMeta = document.getElementById('videoProcessingMeta');
const videoFrameCount  = document.getElementById('videoFrameCount');
const videoElapsed     = document.getElementById('videoElapsed');
const videoProcessingState = document.getElementById('videoProcessingState');
const videoTimeline    = document.getElementById('videoTimeline');
const videoDownloadPanel = document.getElementById('videoDownloadPanel');
const videoOutputName  = document.getElementById('videoOutputName');
const btnDownloadVideo = document.getElementById('btnDownloadVideo');

const resultPanel      = document.getElementById('resultPanel');
const alertCard        = document.getElementById('alertCard');
const alertLabel       = document.getElementById('alertLabel');
const alertProb        = document.getElementById('alertProb');
const mlpProb          = document.getElementById('mlpProb');
const lstmProb         = document.getElementById('lstmProb');
const emaProb          = document.getElementById('emaProb');

const perclosBar       = document.getElementById('perclosBar');
const perclosValue     = document.getElementById('perclosValue');

const featEarLeft      = document.getElementById('featEarLeft');
const featEarRight     = document.getElementById('featEarRight');
const featEarAvg       = document.getElementById('featEarAvg');
const featMar          = document.getElementById('featMar');
const featPitch        = document.getElementById('featPitch');
const featYaw          = document.getElementById('featYaw');
const featRoll         = document.getElementById('featRoll');
const featNeck         = document.getElementById('featNeck');

const neckBanner       = document.getElementById('neckAlarmBanner');
const eyeBanner        = document.getElementById('eyeAlarmBanner');
const yawnBanner       = document.getElementById('yawnAlarmBanner');
const obsBanner        = document.getElementById('cameraObsBanner');
const phoneBanner      = document.getElementById('phoneBanner');
const lookAwayBanner   = document.getElementById('lookAwayBanner');
const timeline         = document.getElementById('timeline');

// ── New UI elements (redesigned) ──────────────────────────────────────────
const alertMessage     = document.getElementById('alertMessage');
const channelSound     = document.getElementById('channelSound');
const channelVibration = document.getElementById('channelVibration');
const channelBreak     = document.getElementById('channelBreak');
const earBar           = document.getElementById('earBar');
const earVal           = document.getElementById('earVal');
const eyeStreakBar     = document.getElementById('eyeStreakBar');
const eyeStreakVal     = document.getElementById('eyeStreakVal');
const footerDot        = document.getElementById('footerDot');
const sessionIdEl      = document.getElementById('sessionId');
const ruleOnlyBadge    = document.getElementById('ruleOnlyBadge');
const videoPlaceholder = document.getElementById('videoPlaceholder');
const hudDot           = document.getElementById('hudDot');
const hudEar           = document.getElementById('hudEar');
const hudPerclos       = document.getElementById('hudPerclos');
const hudFps           = document.getElementById('hudFps');
const hudFpsRef        = document.getElementById('hudFps');
// Level pips
const levelPips        = [0,1,2,3,4].map(i => document.getElementById('pip' + i));

// Display session ID on load
if (sessionIdEl) sessionIdEl.textContent = SESSION_ID.slice(0, 8) + '…';

// ── Audio Alert Manager (Web Audio API - 4 Levels) ─────────────────────────
class AudioAlertManager {
  constructor() {
    this.ctx = null;
    this.muted = false;
    this.currentLevel = 0;
    this.loopTimer = null;
  }

  unlock() {
    if (!this.ctx) {
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      if (AudioCtx) this.ctx = new AudioCtx();
    }
    if (this.ctx && this.ctx.state === 'suspended') {
      this.ctx.resume();
    }
  }

  toggleMute() {
    this.muted = !this.muted;
    if (this.muted) {
      this.stop();
      if (btnMute) btnMute.textContent = '🔇';
    } else {
      if (btnMute) btnMute.textContent = '🔔';
      this._playLevel(this.currentLevel);
    }
  }

  setLevel(level) {
    if (level === this.currentLevel) return;
    this.currentLevel = level;
    if (this.muted) return;
    this._playLevel(level);
  }

  _beep(freq, durationMs, type = 'sine') {
    if (!this.ctx || this.muted) return;
    try {
      const osc = this.ctx.createOscillator();
      const gain = this.ctx.createGain();
      osc.type = type;
      osc.frequency.setValueAtTime(freq, this.ctx.currentTime);
      gain.gain.setValueAtTime(0.15, this.ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.001, this.ctx.currentTime + (durationMs / 1000.0));
      osc.connect(gain);
      gain.connect(this.ctx.destination);
      osc.start();
      osc.stop(this.ctx.currentTime + (durationMs / 1000.0));
    } catch (e) {
      console.warn('[AudioAlert] Beep failed:', e);
    }
  }

  _playLevel(level) {
    this.stop();
    if (level === 0 || this.muted) return;

    if (level === 1) {
      // Level 1 (FATIGUE): 1 beep nhẹ, 440Hz, 0.3s
      this._beep(440, 300);
    } else if (level === 2) {
      // Level 2 (DROWSY): 2 beeps liên tiếp, 660Hz
      this._beep(660, 200);
      setTimeout(() => this._beep(660, 200), 250);
    } else if (level === 3) {
      // Level 3 (MICROSLEEP): 3 beeps nhanh, 880Hz
      this._beep(880, 150);
      setTimeout(() => this._beep(880, 150), 200);
      setTimeout(() => this._beep(880, 150), 400);
    } else if (level >= 4) {
      // Level 4 (CRITICAL): Còi báo động lặp lại 1100Hz
      const playLoop = () => {
        if (this.currentLevel < 4 || this.muted) return;
        this._beep(1100, 120, 'sawtooth');
      };
      playLoop();
      this.loopTimer = setInterval(playLoop, 250);
    }
  }

  stop() {
    if (this.loopTimer) {
      clearInterval(this.loopTimer);
      this.loopTimer = null;
    }
  }
}

const audioAlerts = new AudioAlertManager();

if (btnMute) {
  btnMute.addEventListener('click', () => {
    audioAlerts.unlock();
    audioAlerts.toggleMute();
  });
}

// ── HUD Mode Logic ────────────────────────────────────────────────────────
const HUD_STORAGE_KEY = 'guardian_hud_mode';

function toggleHudMode(enable) {
  const isHud = enable !== undefined ? enable : !document.body.classList.contains('hud-mode');
  if (isHud) {
    document.body.classList.add('hud-mode');
    if (hudExitBtn) hudExitBtn.classList.remove('hidden');
    localStorage.setItem(HUD_STORAGE_KEY, '1');
  } else {
    document.body.classList.remove('hud-mode');
    if (hudExitBtn) hudExitBtn.classList.add('hidden');
    localStorage.setItem(HUD_STORAGE_KEY, '0');
  }
}

if (btnHudMode) btnHudMode.addEventListener('click', () => toggleHudMode());
if (hudExitBtn) hudExitBtn.addEventListener('click', () => toggleHudMode(false));

// On load restore HUD preference
if (localStorage.getItem(HUD_STORAGE_KEY) === '1') {
  toggleHudMode(true);
}

// Bật debug logging cục bộ để chẩn đoán DROWSY-alarm-stuck
const DEBUG_FUSION = true;   // đổi thành false khi đã ổn định
const latencyInfo      = document.getElementById('latencyInfo');

// ── NEW: Performance metrics DOM refs ─────────────────────────────────────
const displayFpsEl     = document.getElementById('displayFps');
const inferenceFpsEl   = document.getElementById('inferenceFps');
const alertLatencyEl   = document.getElementById('alertLatency');

// ── State ─────────────────────────────────────────────────────────────────
let initialized  = false;
let camStream    = null;
let videoRunning = false;
let videoAbort   = false;
let selectedVideoFile = null;
let selectedVideoUrl  = null;
let activeVideoOutputId = null;

// ── Runtime profile (dev vs automotive edge) ─────────────────────────────
let runtimeProfile = {
  inference_interval_ms: 100,
  inference_width: 320,
  inference_height: 240,
  display_fps_cap: 30,
};

async function loadRuntimeProfile() {
  try {
    const res = await fetch('/api/runtime-profile', { headers: sessionHeaders() });
    const data = await res.json();
    if (data.ok) {
      runtimeProfile = data;
      // H5: ngưỡng "mắt nhắm" lấy từ server thay vì hằng số cứng trong file này.
      // Trước đây UI dùng 0.20 còn backend dùng 0.16 → hai bên báo khác nhau.
      if (typeof data.eye_closed_thresh === 'number') {
        earClosedThreshold = data.eye_closed_thresh;
      }
      console.log('[Profile]', data.profile, data.description,
                  'EAR<' + earClosedThreshold);
    }
  } catch (_) { /* giữ default */ }
}
let liveActive      = false;
let rafHandle       = null;      // requestAnimationFrame handle
let inferenceWorker = null;      // Web Worker

// Last inference result (shared between display loop & worker callback)
let lastResult = null;

// FPS counters
let displayFrameCount  = 0;
let inferenceFrameCount = 0;
let fpsLastTime        = performance.now();

// Alert latency tracking
let eyeClosedAt        = null;   // timestamp (ms) khi EAR tụt xuống dưới ngưỡng
let alertLatencyMs     = null;

// EAR threshold để phát hiện mắt nhắm.
// H5: chỉ là giá trị khởi tạo — loadRuntimeProfile() ghi đè bằng giá trị thật
// của server (/api/runtime-profile → eye_closed_thresh) để UI và backend luôn
// nói cùng một con số.
let earClosedThreshold = 0.16;

// Canvas 2D context cache
let annotatedCtx = null;

// Overlay drawing state — dùng để vẽ EAR/status trực tiếp lên canvas
const FONT_SCALE = 14; // px

// ── Tabs ──────────────────────────────────────────────────────────────────
tabButtons.forEach(btn => {
  btn.addEventListener('click', () => {
    tabButtons.forEach(b => b.classList.remove('active'));
    tabContents.forEach(c => c.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById('tab' + capitalize(btn.dataset.tab))
            .classList.add('active');
  });
});
function capitalize(s) { return s.charAt(0).toUpperCase() + s.slice(1); }

// ── Init ──────────────────────────────────────────────────────────────────
btnInit.addEventListener('click', async () => {
  audioAlerts.unlock();
  setBadge('loading', 'Đang khởi tạo…');
  setStatus('Đang load model (có thể mất 30-60s)…');
  btnInit.disabled = true;
  await loadRuntimeProfile();
  try {
    const res = await fetch('/api/init', {
      method: 'POST',
      headers: sessionHeaders(),
      body: JSON.stringify({}),
    });
    const data = await res.json();
    if (data.ok) {
      initialized = true;
      if (data.rule_only_mode) {
        setBadge('warn', 'Rule-only');
        setStatus('⚠ Đang chạy rule-only (model ML chưa sẵn sàng). Chạy tools/convert_models.py');
        if (ruleOnlyBadge) ruleOnlyBadge.classList.remove('hidden');
      } else {
        setBadge('ready', 'Sẵn sàng');
        const mode = data.load_mode ? ` · ${data.load_mode}` : '';
        setStatus('✓ Hệ thống sẵn sàng' + mode + '. Có thể bắt đầu phân tích.');
        if (ruleOnlyBadge) ruleOnlyBadge.classList.add('hidden');
      }
      if (footerDot) footerDot.classList.remove('error');
      btnAnalyzeLive.disabled   = camStream === null;
      btnAnalyzeUpload.disabled = previewImg.classList.contains('hidden');
      btnAnalyzeVideo.disabled  = !selectedVideoFile;
    } else {
      setBadge('error', 'Lỗi');
      if (footerDot) footerDot.classList.add('error');
      setStatus('✗ Lỗi khởi tạo: ' + data.error);
    }
  } catch (e) {
    setBadge('error', 'Lỗi');
    setStatus('Không kết nối được server.');
  } finally {
    btnInit.disabled = false;
  }
});

// ── FPS sliders ───────────────────────────────────────────────────────────
videoFpsSlider.addEventListener('input', () => { videoFpsValue.textContent = videoFpsSlider.value; });

// ── Webcam ────────────────────────────────────────────────────────────────
btnStartCam.addEventListener('click', async () => {
  audioAlerts.unlock();
    // Request HD resolution from camera
    camStream = await navigator.mediaDevices.getUserMedia({
      video: {
        width:     { ideal: 1280, min: 640 },
        height:    { ideal: 720,  min: 480 },
        frameRate: { ideal: 30, max: 60 },
      }
    });
    webcamEl.srcObject = camStream;

    // Log actual camera capabilities
    const track = camStream.getVideoTracks()[0];
    const settings = track.getSettings();
    console.log('[Camera] Actual settings:', settings);
    setStatus(`Webcam: ${settings.width}×${settings.height} @ ${settings.frameRate}fps`);

    if (initialized) btnAnalyzeLive.disabled = false;
  } catch (e) {
    setStatus('Không mở được webcam: ' + e.message);
  }
});

// ── Live analysis — NEW ARCHITECTURE ─────────────────────────────────────
btnAnalyzeLive.addEventListener('click', startLive);
btnStopLive.addEventListener('click', stopLive);

function startLive() {
  if (liveActive) return;
  if (!initialized || !camStream) return;

  liveActive = true;
  lastResult = null;
  eyeClosedAt = null;
  alertLatencyMs = null;
  displayFrameCount = 0;
  inferenceFrameCount = 0;
  fpsLastTime = performance.now();

  btnAnalyzeLive.classList.add('live-active');
  btnAnalyzeLive.disabled = true;
  btnStopLive.disabled    = false;

  // Show canvas (live annotated), hide placeholder
  if (videoPlaceholder) videoPlaceholder.style.display = 'none';
  annotatedCanvas.classList.remove('hidden');
  videoOverlay.classList.remove('hidden');
  setStatus('▶ Live phân tích đang chạy…');

  // Cache canvas context
  annotatedCtx = annotatedCanvas.getContext('2d');

  // ── Start Web Worker ──
  inferenceWorker = new Worker('/static/js/worker.js');
  inferenceWorker.onmessage = onWorkerMessage;
  inferenceWorker.onerror   = (e) => console.error('[Worker error]', e);
  inferenceWorker.postMessage({
    type: 'start',
    profile: runtimeProfile,
    sessionId: SESSION_ID,   // H2: worker phải gửi cùng session với main thread
  });

  // ── Start Display Loop via requestAnimationFrame ──
  startDisplayLoop();
}

function stopLive() {
  if (!liveActive) return;
  liveActive = false;

  // Stop RAF
  if (rafHandle) { cancelAnimationFrame(rafHandle); rafHandle = null; }

  // Stop Worker
  if (inferenceWorker) {
    inferenceWorker.postMessage({ type: 'stop' });
    inferenceWorker.terminate();
    inferenceWorker = null;
  }

  btnAnalyzeLive.classList.remove('live-active');
  btnAnalyzeLive.disabled = !initialized || camStream === null;
  btnStopLive.disabled    = true;
  annotatedCanvas.classList.add('hidden');
  videoOverlay.classList.add('hidden');
  if (videoPlaceholder) videoPlaceholder.style.display = '';

  if (annotatedCtx) {
    annotatedCtx.clearRect(0, 0, annotatedCanvas.width, annotatedCanvas.height);
  }

  // Clear metrics
  if (displayFpsEl)   displayFpsEl.textContent   = '—';
  if (inferenceFpsEl) inferenceFpsEl.textContent = '—';
  if (alertLatencyEl) alertLatencyEl.textContent  = '—';
  if (hudEar)    hudEar.textContent    = '—';
  if (hudPerclos)hudPerclos.textContent= '0%';
  if (hudFps)    hudFps.textContent    = '—';

  setStatus('Đã dừng live analysis.');
}

// ── Display Loop — runs at full camera FPS via requestAnimationFrame ──────
// This NEVER waits for inference. It just draws whatever is available.
let lastFrameSentTs = 0;

function getSendIntervalMs() {
  return runtimeProfile.inference_interval_ms || 100;
}

function startDisplayLoop() {
  function loop() {
    if (!liveActive) return;
    rafHandle = requestAnimationFrame(loop);

    const now = performance.now();

    // ── Draw current webcam frame to canvas ──
    if (webcamEl.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
      const vw = webcamEl.videoWidth  || 640;
      const vh = webcamEl.videoHeight || 480;

      if (annotatedCanvas.width !== vw || annotatedCanvas.height !== vh) {
        annotatedCanvas.width  = vw;
        annotatedCanvas.height = vh;
      }

      // Draw raw webcam frame (this is the "display" layer — always at camera FPS)
      annotatedCtx.drawImage(webcamEl, 0, 0, vw, vh);

      // Overlay inference result (from last worker callback — may be 100ms stale, that's OK)
      if (lastResult) {
        drawInferenceOverlay(annotatedCtx, vw, vh, lastResult);
      }

      // Update display FPS counter
      displayFrameCount++;
    }

    // ── Send frame to worker (rate-limited theo EDGE_PROFILE) ──
    if (now - lastFrameSentTs >= getSendIntervalMs()) {
      lastFrameSentTs = now;
      sendFrameToWorker(now);
    }

    // ── Update FPS display every second ──
    const elapsed = now - fpsLastTime;
    if (elapsed >= 1000) {
      const dispFps = Math.round(displayFrameCount   / (elapsed / 1000));
      const infFps  = Math.round(inferenceFrameCount / (elapsed / 1000));
      displayFrameCount   = 0;
      inferenceFrameCount = 0;
      fpsLastTime = now;

      if (displayFpsEl)   displayFpsEl.textContent   = dispFps;
      if (inferenceFpsEl) inferenceFpsEl.textContent = infFps;
      if (alertLatencyEl && alertLatencyMs != null)
        alertLatencyEl.textContent = alertLatencyMs + 'ms';
      if (hudFps) hudFps.textContent = infFps;
    }
  }

  rafHandle = requestAnimationFrame(loop);
}

// Inference canvas size — theo runtime profile (edge: 256×192)
function getInferenceSize() {
  return {
    w: runtimeProfile.inference_width  || 320,
    h: runtimeProfile.inference_height || 240,
  };
}

// Reuse small canvas for encoding (resize khi profile đổi)
const _smallCanvas  = document.createElement('canvas');
const _smallCtx     = _smallCanvas.getContext('2d');

function sendFrameToWorker(timestamp) {
  if (!inferenceWorker || !liveActive) return;
  if (webcamEl.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) return;

  const { w: inferW, h: inferH } = getInferenceSize();
  if (_smallCanvas.width !== inferW || _smallCanvas.height !== inferH) {
    _smallCanvas.width  = inferW;
    _smallCanvas.height = inferH;
  }
  _smallCtx.drawImage(webcamEl, 0, 0, inferW, inferH);
  const dataUrl = _smallCanvas.toDataURL('image/jpeg', 0.88);

  inferenceWorker.postMessage({ type: 'frame', dataUrl, timestamp });
}

// ── Worker message handler ────────────────────────────────────────────────
function onWorkerMessage(e) {
  const msg = e.data;

  if (msg.type === 'result') {
    const data = msg.data;
    if (!data.ok) return;

    lastResult = data;
    inferenceFrameCount++;

    // Update UI panels
    applyResult(data);
    addTimelineSegment(data.alarm_on, data.drowsiness_state);
    if (latencyInfo) latencyInfo.textContent = msg.inferenceMs + ' ms';

    // ── Alert latency tracking ──
    // Track how long from eye closure to alarm trigger
    const ear = data.features?.ear_avg;
    if (ear != null) {
      if (ear < earClosedThreshold) {
        // Eyes closed
        if (eyeClosedAt === null) eyeClosedAt = msg.timestamp;
      } else {
        // Eyes open
        eyeClosedAt = null;
        alertLatencyMs = null;
      }

      if (data.alarm_on && eyeClosedAt !== null && alertLatencyMs === null) {
        alertLatencyMs = Math.round(performance.now() - eyeClosedAt);
      }
    }

  } else if (msg.type === 'log') {
    console.log('[Worker]', msg.message);
  } else if (msg.type === 'error') {
    console.warn('[Worker error]', msg.message);
  }
}

// ── Landmark index groups (mirrors app.py) ────────────────────────────────
const IDX_EYE_L  = [362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398];
const IDX_EYE_R  = [33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246];
const IDX_MOUTH  = [61,185,40,39,37,0,267,269,270,409,291,375,321,405,314,17,84,181,91,146];

// Face mesh tessellation — subset of connections for a visible mesh grid
// We draw all 468 tiny dots + key contour lines
function _lmXY(face_lm, idx, w, h) {
  const i = idx * 2;
  if (i + 1 >= face_lm.length) return null;
  return [face_lm[i] * w, face_lm[i + 1] * h];
}

function _drawContour(ctx, face_lm, indices, w, h, color, lineWidth = 1) {
  const pts = indices.map(i => _lmXY(face_lm, i, w, h)).filter(Boolean);
  if (pts.length < 2) return;
  ctx.beginPath();
  ctx.moveTo(pts[0][0], pts[0][1]);
  for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]);
  ctx.closePath();
  ctx.strokeStyle = color;
  ctx.lineWidth = lineWidth;
  ctx.stroke();
}

// ── Draw inference overlay DIRECTLY on display canvas ─────────────────────
function drawInferenceOverlay(ctx, w, h, data) {
  const alarmOn   = data.alarm_on;
  const emaProb   = data.ema_prob   || 0;
  const neckAlarm = data.neck_alarm || false;
  const eyeAlarm  = data.eye_alarm  || false;
  const feat      = data.features   || {};
  const face_lm   = data.face_lm;   // flat [x0,y0,x1,y1,...] normalized
  const pose_lm   = data.pose_lm;   // {nose,l_eye,r_eye,l_ear,r_ear,l_sh,r_sh,...}

  // Tỉ lệ: cam real-time vẽ trên canvas đã scale về display size
  // nhưng landmark là normalized (0-1) → nhân với w,h hiện tại

  // ════════════════════════════════════════════════════
  // 1. FACE MESH LANDMARKS (đẹp hơn, dots to + iris)
  // ════════════════════════════════════════════════════
  if (face_lm && face_lm.length >= 2) {
    const dotColor     = eyeAlarm ? 'rgba(255,150,60,0.85)'
                          : alarmOn ? 'rgba(220,80,80,0.75)'
                          :           'rgba(0,220,100,0.65)';
    const contourColor = eyeAlarm ? 'rgba(255,160,70,0.95)'
                          : alarmOn ? 'rgba(220,80,80,0.95)'
                          :           'rgba(0,230,110,0.95)';
    const irisColor    = 'rgba(180,140,255,0.95)';   // iris tím nổi bật

    // All face dots (to hơn, 1.6px → rõ landmark)
    ctx.fillStyle = dotColor;
    const n = face_lm.length / 2;
    for (let i = 0; i < n; i++) {
      const x = face_lm[i * 2]     * w;
      const y = face_lm[i * 2 + 1] * h;
      ctx.beginPath();
      ctx.arc(x, y, 1.6, 0, Math.PI * 2);
      ctx.fill();
    }

    // Iris landmarks (468-477, 10 điểm quanh mống mắt)
    ctx.fillStyle = irisColor;
    for (let i = 468; i <= 477 && i < n; i++) {
      const x = face_lm[i * 2]     * w;
      const y = face_lm[i * 2 + 1] * h;
      ctx.beginPath();
      ctx.arc(x, y, 2.5, 0, Math.PI * 2);
      ctx.fill();
    }

    // Eye contours
    _drawContour(ctx, face_lm, IDX_EYE_L, w, h, contourColor, 1.8);
    _drawContour(ctx, face_lm, IDX_EYE_R, w, h, contourColor, 1.8);

    // Mouth contour
    _drawContour(ctx, face_lm, IDX_MOUTH, w, h, contourColor, 1.4);
  }

  // ════════════════════════════════════════════════════
  // 2. POSE SKELETON (đầy đủ: nose, eyes, ears, shoulders, elbows)
  // ════════════════════════════════════════════════════
  if (pose_lm) {
    const skeletonColor = neckAlarm ? '#ff9800' : '#5599ff';
    const accentColor   = neckAlarm ? '#ffb74d' : '#80aaff';

    // Helper để lấy (x,y) tuyệt đối
    const XY = (p) => p ? [p[0] * w, p[1] * h] : null;

    const nose  = XY(pose_lm.nose);
    const lEye  = XY(pose_lm.l_eye);
    const rEye  = XY(pose_lm.r_eye);
    const lEar  = XY(pose_lm.l_ear);
    const rEar  = XY(pose_lm.r_ear);
    const lSh   = XY(pose_lm.l_sh);
    const rSh   = XY(pose_lm.r_sh);
    const lEl   = XY(pose_lm.l_el);
    const rEl   = XY(pose_lm.r_el);

    ctx.lineWidth = 2.5;

    // ── Shoulder line ──
    if (lSh && rSh) {
      ctx.beginPath();
      ctx.moveTo(lSh[0], lSh[1]);
      ctx.lineTo(rSh[0], rSh[1]);
      ctx.strokeStyle = skeletonColor;
      ctx.stroke();

      const mx = (lSh[0] + rSh[0]) / 2;
      const my = (lSh[1] + rSh[1]) / 2;

      // ── Neck line (mid shoulder → nose) ──
      if (nose) {
        ctx.beginPath();
        ctx.moveTo(mx, my);
        ctx.lineTo(nose[0], nose[1]);
        ctx.strokeStyle = neckAlarm ? '#ff9800' : '#80aaff';
        ctx.stroke();

        // ── Vertical reference line (để so góc) ──
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.moveTo(mx, my);
        ctx.lineTo(mx, nose[1]);
        ctx.strokeStyle = 'rgba(150,150,150,0.6)';
        ctx.lineWidth = 1;
        ctx.stroke();
        ctx.setLineDash([]);

        // ── Neck-tilt arc (cung tròn góc) ở mũi ──
        const dx = nose[0] - mx;
        const dy = my - nose[1];  // +nếu nose ở trên vai
        const neckAngle = Math.atan2(dx, dy);  // góc từ phương thẳng đứng
        const arcR = 30;
        // Vẽ cung tròn từ phương thẳng đứng → vector thực
        ctx.beginPath();
        ctx.arc(mx, nose[1], arcR, -Math.PI/2, -Math.PI/2 + neckAngle, dx > 0);
        ctx.strokeStyle = neckAlarm ? '#ff9800' : 'rgba(120,180,255,0.85)';
        ctx.lineWidth = 2;
        ctx.stroke();

        // ── Hiển thị neck_tilt độ ──
        const ntDeg = feat.neck_tilt;
        if (ntDeg != null && !isNaN(ntDeg)) {
          ctx.fillStyle = neckAlarm ? '#ff9800' : '#90caf9';
          ctx.font = 'bold 12px monospace';
          const label = `∠${ntDeg.toFixed(1)}°`;
          const tx = mx + 12, ty = nose[1] - 8;
          // background
          ctx.fillStyle = 'rgba(0,0,0,0.55)';
          ctx.fillRect(tx - 2, ty - 12, 50, 16);
          ctx.fillStyle = neckAlarm ? '#ff9800' : '#90caf9';
          ctx.fillText(label, tx, ty);
        }
      }
    }

    // ── Eyes line (l_eye ↔ r_eye) ──
    if (lEye && rEye) {
      ctx.beginPath();
      ctx.moveTo(lEye[0], lEye[1]);
      ctx.lineTo(rEye[0], rEye[1]);
      ctx.strokeStyle = accentColor;
      ctx.stroke();
    }

    // ── Eyes → Ears (tai → mắt mỗi bên) ──
    if (lEye && lEar) {
      ctx.beginPath();
      ctx.moveTo(lEye[0], lEye[1]);
      ctx.lineTo(lEar[0], lEar[1]);
      ctx.strokeStyle = accentColor;
      ctx.stroke();
    }
    if (rEye && rEar) {
      ctx.beginPath();
      ctx.moveTo(rEye[0], rEye[1]);
      ctx.lineTo(rEar[0], rEar[1]);
      ctx.strokeStyle = accentColor;
      ctx.stroke();
    }

    // ── Shoulders → Elbows ──
    if (lSh && lEl) {
      ctx.beginPath();
      ctx.moveTo(lSh[0], lSh[1]);
      ctx.lineTo(lEl[0], lEl[1]);
      ctx.strokeStyle = accentColor;
      ctx.stroke();
    }
    if (rSh && rEl) {
      ctx.beginPath();
      ctx.moveTo(rSh[0], rSh[1]);
      ctx.lineTo(rEl[0], rEl[1]);
      ctx.strokeStyle = accentColor;
      ctx.stroke();
    }

    // ── Dots cho tất cả pose landmarks ──
    ctx.fillStyle = skeletonColor;
    [nose, lEye, rEye, lEar, rEar, lSh, rSh, lEl, rEl].forEach(p => {
      if (!p) return;
      ctx.beginPath();
      ctx.arc(p[0], p[1], 4, 0, Math.PI * 2);
      ctx.fill();
    });
  }

  // ════════════════════════════════════════════════════
  // 3. HUD TEXT OVERLAY (dark bar at top)
  // ════════════════════════════════════════════════════
  const nLines = (neckAlarm ? 1 : 0) + (eyeAlarm ? 1 : 0);
  const barH = 70 + 25 * nLines;
  ctx.fillStyle = 'rgba(10, 10, 10, 0.60)';
  ctx.fillRect(0, 0, w, barH);

  // Status label
  const statusColor = alarmOn ? '#ff5555' : '#44dd88';
  const label = alarmOn ? '⚠ DROWSY' : '✓ NORMAL';
  ctx.fillStyle = statusColor;
  ctx.font = 'bold 15px monospace';
  ctx.fillText(`${label}  (p=${emaProb.toFixed(2)})`, 10, 28);

  // MLP / LSTM detail
  const mlp  = data.p_mlp_drowsy;
  const lstm = data.p_lstm_drowsy;
  let detail = '';
  if (mlp  != null) detail += `MLP=${mlp.toFixed(2)}`;
  if (lstm != null) detail += `  LSTM=${lstm.toFixed(2)}`;
  if (detail) {
    ctx.fillStyle = 'rgba(160,160,160,0.9)';
    ctx.font = '11px monospace';
    ctx.fillText(detail, 10, 50);
  }

  // Rule-based alarms
  let yOff = 78;
  if (neckAlarm) {
    ctx.fillStyle = '#ff9800';
    ctx.font = 'bold 13px monospace';
    ctx.fillText('⚠ NECK-TILT ALARM', 10, yOff);
    yOff += 25;
  }
  if (eyeAlarm) {
    ctx.fillStyle = '#ff8a3c';
    ctx.font = 'bold 13px monospace';
    ctx.fillText('⚠ EYE-CLOSED ALARM', 10, yOff);
    yOff += 25;
  }

  // ── Internal debug (eye-closure escape-valve) ────────────────────────────
  if (data.ear_smooth != null) {
    ctx.fillStyle = 'rgba(140,140,140,0.85)';
    ctx.font = '10px monospace';
    const earTxt   = `EAR=${data.ear_smooth.toFixed(3)}`;
    const openTxt  = `open=${(data.eyes_open_streak_ms/1000).toFixed(1)}s`;
    const closeTxt = `close=${(data.eye_closed_streak_ms/1000).toFixed(1)}s`;
    ctx.fillText(earTxt, 10, yOff);
    ctx.fillText(openTxt, 100, yOff);
    ctx.fillText(closeTxt, 180, yOff);
  }

  // EAR / MAR — top right
  const ear = feat.ear_avg;
  const mar = feat.mar;
  ctx.font = '11px monospace';
  ctx.fillStyle = 'rgba(200,200,200,0.85)';
  if (ear != null) ctx.fillText(`EAR ${ear.toFixed(3)}`, w - 115, 22);
  if (mar != null) ctx.fillText(`MAR ${mar.toFixed(3)}`, w - 115, 38);

  // Face size (px + %frame) — giúp user biết mình đang ở khoảng cách nào
  if (face_lm && face_lm.length >= 2) {
    let minX = 1, minY = 1, maxX = 0, maxY = 0;
    const n = face_lm.length / 2;
    for (let i = 0; i < n; i++) {
      const x = face_lm[i * 2];
      const y = face_lm[i * 2 + 1];
      if (x < minX) minX = x;
      if (y < minY) minY = y;
      if (x > maxX) maxX = x;
      if (y > maxY) maxY = y;
    }
    const fw = Math.round((maxX - minX) * w);
    const fh = Math.round((maxY - minY) * h);
    const ratio = ((maxX - minX) * (maxY - minY) * 100).toFixed(1);
    // Màu sắc theo khoảng cách: < 50px cam (xa), 50-90 vàng (vừa), > 90 xanh (gần)
    let distColor = 'rgba(140,140,140,0.85)';
    if (fw >= 90)      distColor = 'rgba(120,220,140,0.95)';   // gần
    else if (fw >= 50) distColor = 'rgba(220,200,80,0.95)';    // vừa
    else               distColor = 'rgba(255,140,80,0.95)';    // xa
    ctx.fillStyle = distColor;
    ctx.fillText(`Face ${fw}×${fh}px (${ratio}%)`, w - 165, 54);
  }

  // Red border flash when drowsy
  if (alarmOn) {
    ctx.strokeStyle = 'rgba(220, 50, 50, 0.80)';
    ctx.lineWidth = 5;
    ctx.strokeRect(2, 2, w - 4, h - 4);
  }
}

// ── Upload image ──────────────────────────────────────────────────────────
btnPickFile.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', () => loadImageFile(fileInput.files[0]));
dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('dragover'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
dropZone.addEventListener('drop', e => {
  e.preventDefault(); dropZone.classList.remove('dragover');
  loadImageFile(e.dataTransfer.files[0]);
});

function loadImageFile(file) {
  if (!file) return;
  const url = URL.createObjectURL(file);
  previewImg.src = url;
  previewImg.classList.remove('hidden');
  btnAnalyzeUpload.disabled = !initialized;
}

btnAnalyzeUpload.addEventListener('click', async () => {
  if (previewImg.classList.contains('hidden')) return;
  const canvas = document.createElement('canvas');
  canvas.width  = previewImg.naturalWidth;
  canvas.height = previewImg.naturalHeight;
  canvas.getContext('2d').drawImage(previewImg, 0, 0);
  await analyzeFrameOnce(canvas.toDataURL('image/jpeg', 0.9), true);
});

// ── Upload video ──────────────────────────────────────────────────────────
btnPickVideo.addEventListener('click', () => videoInput.click());
videoInput.addEventListener('change', () => loadVideoFile(videoInput.files[0]));
videoDropZone.addEventListener('dragover', event => {
  event.preventDefault();
  videoDropZone.classList.add('dragover');
});
videoDropZone.addEventListener('dragleave', () => videoDropZone.classList.remove('dragover'));
videoDropZone.addEventListener('drop', event => {
  event.preventDefault();
  videoDropZone.classList.remove('dragover');
  loadVideoFile(event.dataTransfer.files[0]);
});

function loadVideoFile(file) {
  if (!file || videoRunning) return;
  const supported = file.type.startsWith('video/') || /\.(mp4|webm|mov|m4v|avi|mkv)$/i.test(file.name);
  if (!supported) {
    setStatus('File không hợp lệ. Vui lòng chọn một file video.');
    return;
  }

  if (selectedVideoUrl) URL.revokeObjectURL(selectedVideoUrl);
  selectedVideoFile = file;
  selectedVideoUrl = URL.createObjectURL(file);

  videoFileName.textContent = file.name;
  videoFileInfo.textContent = `${formatBytes(file.size)} · Đang đọc thông tin…`;
  videoFileMeta.classList.remove('hidden');
  videoWorkspace.classList.remove('hidden');
  videoAnnotatedPreview.classList.add('hidden');
  videoResultPlaceholder.classList.remove('hidden');
  videoDownloadPanel.classList.add('hidden');
  videoProcessingMeta.classList.add('hidden');
  videoProgressWrap.classList.add('hidden');
  videoLiveBadge.textContent = 'CHỜ';
  videoLiveBadge.dataset.state = 'idle';
  btnAnalyzeVideo.disabled = true;

  fileVideo.onloadedmetadata = () => {
    const duration = Number.isFinite(fileVideo.duration) ? fileVideo.duration : 0;
    videoFileInfo.textContent = `${formatBytes(file.size)} · ${fileVideo.videoWidth}×${fileVideo.videoHeight} · ${formatTime(duration)}`;
    btnAnalyzeVideo.disabled = !initialized;
    setStatus(initialized ? 'Video đã sẵn sàng để phân tích.' : 'Hãy khởi tạo model trước khi phân tích.');
  };
  fileVideo.onerror = () => {
    btnAnalyzeVideo.disabled = true;
    setStatus('Trình duyệt không đọc được video này. Hãy thử MP4 hoặc WebM.');
  };
  fileVideo.src = selectedVideoUrl;
  fileVideo.load();
}

btnAnalyzeVideo.addEventListener('click', async () => {
  if (videoRunning || !initialized || !selectedVideoFile) return;
  if (!Number.isFinite(fileVideo.duration) || fileVideo.duration <= 0) {
    setStatus('Không đọc được thời lượng video.');
    return;
  }

  const fps = parseInt(videoFpsSlider.value, 10);
  const dimensions = getVideoAnalysisDimensions();
  try {
    const output = await startVideoOutput(dimensions.width, dimensions.height, fps);
    activeVideoOutputId = output.output_id;
  } catch (error) {
    setStatus('Không tạo được video output: ' + error.message);
    return;
  }

  videoRunning = true;
  videoAbort = false;
  fileVideo.pause();
  fileVideo.controls = false;
  btnAnalyzeVideo.disabled = true;
  btnStopVideo.disabled = false;
  videoFpsSlider.disabled = true;
  videoProgressWrap.classList.remove('hidden');
  videoProcessingMeta.classList.remove('hidden');
  videoDownloadPanel.classList.add('hidden');
  videoTimeline.innerHTML = '';
  videoAnnotatedPreview.classList.remove('hidden');
  videoResultPlaceholder.classList.add('hidden');
  videoLiveBadge.textContent = 'LIVE';
  videoLiveBadge.dataset.state = 'running';
  videoProgressBar.style.width = '0%';
  videoProgressText.textContent = '0%';
  setStatus('Model đang phân tích trực tiếp từng khung hình…');

  await runVideoAnalysis(dimensions.width, dimensions.height, fps);
});

btnStopVideo.addEventListener('click', () => {
  if (!videoRunning) return;
  videoAbort = true;
  btnStopVideo.disabled = true;
  videoProcessingState.textContent = 'Đang đóng file output…';
  setStatus('Đang dừng an toàn và hoàn tất video đã xử lý…');
});

window.addEventListener('pagehide', () => {
  if (activeVideoOutputId) {
    navigator.sendBeacon(
      `/api/video-output/${encodeURIComponent(activeVideoOutputId)}/finish`,
      new Blob([], { type: 'application/octet-stream' }),
    );
  }
  if (selectedVideoUrl) URL.revokeObjectURL(selectedVideoUrl);
});

async function runVideoAnalysis(width, height, fps) {
  const duration = fileVideo.duration;
  const totalFrames = Math.max(1, Math.ceil(duration * fps));
  let processedFrames = 0;
  let processingError = null;

  videoFrameCount.textContent = `0 / ${totalFrames} frame`;
  videoElapsed.textContent = `00:00 / ${formatTime(duration)}`;
  videoProcessingState.textContent = 'Đang khởi tạo pipeline…';

  const resetResponse = await fetch('/api/reset', {
    method: 'POST',
    headers: sessionHeaders(),
    body: JSON.stringify({}),
  });
  if (!resetResponse.ok) {
    processingError = new Error('Không reset được trạng thái nhận diện.');
  }

  const snapCanvas = document.createElement('canvas');
  snapCanvas.width = width;
  snapCanvas.height = height;
  const ctx = snapCanvas.getContext('2d', { alpha: false });

  try {
    for (let frameIndex = 0; frameIndex < totalFrames && !videoAbort; frameIndex += 1) {
      if (processingError) throw processingError;
      const mediaTime = Math.min(frameIndex / fps, Math.max(0, duration - 0.001));
      await seekVideo(fileVideo, mediaTime);
      ctx.drawImage(fileVideo, 0, 0, width, height);
      const dataUrl = snapCanvas.toDataURL('image/jpeg', 0.86);

      const startedAt = performance.now();
      const result = await callAnalyze(dataUrl, false, true, {
        outputId: activeVideoOutputId,
        sourceTimestampMs: mediaTime * 1000,
      });
      if (!result) throw new Error('Server không trả về kết quả nhận diện.');

      processedFrames += 1;
      applyResult(result);
      addTimelineSegment(result.alarm_on, result.drowsiness_state);
      addVideoTimelineSegment(result.alarm_on);
      if (latencyInfo) latencyInfo.textContent = Math.round(performance.now() - startedAt) + ' ms';
      if (result.annotated_frame) await drawVideoAnnotatedFrame(result.annotated_frame);

      const pct = Math.min(100, (processedFrames / totalFrames) * 100);
      videoProgressBar.style.width = pct + '%';
      videoProgressText.textContent = Math.round(pct) + '%';
      videoFrameCount.textContent = `${processedFrames} / ${totalFrames} frame`;
      videoElapsed.textContent = `${formatTime(mediaTime)} / ${formatTime(duration)}`;
      videoProcessingState.textContent = result.face_found ? 'Đã phát hiện khuôn mặt' : 'Không thấy khuôn mặt';

      // Yield to the browser so the progress UI is painted between inferences.
      await new Promise(resolve => setTimeout(resolve, 0));
    }
  } catch (error) {
    processingError = error;
  }

  const wasStopped = videoAbort;
  let output = null;
  try {
    output = await finishVideoOutput(activeVideoOutputId);
  } catch (error) {
    processingError = processingError || error;
  }

  videoRunning = false;
  videoAbort = false;
  activeVideoOutputId = null;
  btnAnalyzeVideo.disabled = !initialized || !selectedVideoFile;
  btnStopVideo.disabled = true;
  videoFpsSlider.disabled = false;
  fileVideo.controls = true;

  if (output && output.download_ready) {
    btnDownloadVideo.href = output.download_url;
    btnDownloadVideo.download = output.filename;
    videoOutputName.textContent = `output/${output.filename}`;
    videoDownloadPanel.classList.remove('hidden');
  }

  if (processingError) {
    videoLiveBadge.textContent = 'LỖI';
    videoLiveBadge.dataset.state = 'error';
    videoProcessingState.textContent = 'Phân tích gặp lỗi';
    setStatus(`Lỗi phân tích video: ${processingError.message}`);
  } else if (wasStopped) {
    videoLiveBadge.textContent = 'ĐÃ DỪNG';
    videoLiveBadge.dataset.state = 'done';
    videoProcessingState.textContent = `Đã lưu ${processedFrames} frame`;
    setStatus('Đã dừng. Phần video đã phân tích vẫn có thể tải xuống.');
  } else {
    videoLiveBadge.textContent = 'HOÀN TẤT';
    videoLiveBadge.dataset.state = 'done';
    videoProcessingState.textContent = `Hoàn tất ${processedFrames} frame`;
    videoProgressBar.style.width = '100%';
    videoProgressText.textContent = '100%';
    setStatus('Phân tích hoàn tất. Video đã được lưu trong thư mục output.');
  }
}

function getVideoAnalysisDimensions() {
  const sourceWidth = fileVideo.videoWidth || 640;
  const sourceHeight = fileVideo.videoHeight || 480;
  const scale = Math.min(1, 960 / sourceWidth, 720 / sourceHeight);
  let width = Math.max(16, Math.round(sourceWidth * scale));
  let height = Math.max(16, Math.round(sourceHeight * scale));
  width -= width % 2;
  height -= height % 2;
  return { width, height };
}

async function startVideoOutput(width, height, fps) {
  const response = await fetch('/api/video-output/start', {
    method: 'POST',
    headers: sessionHeaders(),
    body: JSON.stringify({ filename: selectedVideoFile.name, width, height, fps }),
  });
  const data = await response.json();
  if (!response.ok || !data.ok) throw new Error(data.error || 'Không tạo được phiên output.');
  return data;
}

async function finishVideoOutput(outputId) {
  if (!outputId) return null;
  const response = await fetch(`/api/video-output/${encodeURIComponent(outputId)}/finish`, {
    method: 'POST',
    headers: sessionHeaders(),
  });
  const data = await response.json();
  if (!response.ok || !data.ok) throw new Error(data.error || 'Không đóng được video output.');
  return data;
}

function seekVideo(video, timeSec) {
  if (Math.abs(video.currentTime - timeSec) < 0.001 && video.readyState >= 2) {
    return Promise.resolve();
  }
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      cleanup();
      reject(new Error('Hết thời gian chờ giải mã frame video.'));
    }, 8000);
    const cleanup = () => {
      clearTimeout(timer);
      video.removeEventListener('seeked', onSeeked);
      video.removeEventListener('error', onError);
    };
    const onSeeked = () => { cleanup(); resolve(); };
    const onError = () => { cleanup(); reject(new Error('Không giải mã được frame video.')); };
    video.addEventListener('seeked', onSeeked, { once: true });
    video.addEventListener('error', onError, { once: true });
    video.currentTime = timeSec;
  });
}

function drawVideoAnnotatedFrame(dataUrl) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      videoAnnotatedPreview.width = img.naturalWidth;
      videoAnnotatedPreview.height = img.naturalHeight;
      videoAnnotatedPreview.getContext('2d').drawImage(img, 0, 0);
      resolve();
    };
    img.onerror = () => reject(new Error('Không hiển thị được frame kết quả.'));
    img.src = dataUrl;
  });
}

// ── One-shot image analyze (upload tab) ──────────────────────────────────
async function analyzeFrameOnce(dataUrl, resetState = false) {
  if (!initialized) { setStatus('Chưa khởi tạo hệ thống.'); return; }
  const t0 = performance.now();
  const result = await callAnalyze(dataUrl, resetState, true);
  if (result) {
    applyResult(result);
    addTimelineSegment(result.alarm_on);
    latencyInfo.textContent = Math.round(performance.now() - t0) + ' ms';
    if (result.annotated_frame) drawAnnotatedStatic(result.annotated_frame);
  }
}

async function callAnalyze(dataUrl, resetState = false, annotate = false, options = {}) {
  try {
    const res = await fetch('/api/analyze', {
      method: 'POST',
      headers: sessionHeaders(),
      body: JSON.stringify({
        image: dataUrl,
        reset_state: resetState,
        annotate,
        output_id: options.outputId || undefined,
        source_timestamp_ms: options.sourceTimestampMs,
      }),
    });
    const data = await res.json();
    if (!data.ok) { setStatus('Lỗi: ' + data.error); return null; }
    return data;
  } catch (e) {
    setStatus('Lỗi kết nối server.');
    return null;
  }
}

// Alert level messages (mirrors server-side ALERT_MESSAGES)
const ALERT_MESSAGES_VI = {
  0: 'Tài xế tỉnh táo — Hệ thống đang giám sát',
  1: 'Cấp 1 — Dấu hiệu mệt mỏi, nên nghỉ ngơi sớm',
  2: 'Cấp 2 — Buồn ngủ, cần tập trung hoặc dừng xe',
  3: 'Cấp 3 — Ngủ gật! Cần đánh thức tài xế ngay',
  4: 'Cấp 4 — NGUY HIỂM! Không phục hồi sau cảnh báo',
};

function applyResult(data) {
  const on = data.alarm_on;
  const stateStr = (data.drowsiness_state || (on ? 'DROWSY' : 'NORMAL')).toUpperCase();
  const stateLower = stateStr.toLowerCase();
  const alertLvl = data.alert_level != null ? Number(data.alert_level) : (on ? 2 : 0);

  // Trigger audio alert for level change
  audioAlerts.setLevel(alertLvl);

  // Update card data attributes for 5-state colors
  alertCard.dataset.alarm   = on ? 'on' : 'off';
  alertCard.dataset.state   = stateLower;
  resultPanel.dataset.alarm = on ? 'on' : 'off';
  resultPanel.dataset.state = stateLower;

  alertLabel.textContent    = stateStr;
  alertProb.textContent     = 'Drowsiness Score: ' + fmt(data.drowsiness_score != null ? data.drowsiness_score : data.ema_prob);
  if (alertMessage) alertMessage.textContent = ALERT_MESSAGES_VI[alertLvl] || '';
  mlpProb.textContent       = data.p_mlp_drowsy  != null ? fmt(data.p_mlp_drowsy)  : '—';
  lstmProb.textContent      = data.p_lstm_drowsy != null ? fmt(data.p_lstm_drowsy) : '—';
  emaProb.textContent       = fmt(data.ema_prob);

  // ── Level pip indicators ──
  levelPips.forEach((pip, i) => {
    pip.className = 'level-pip';
    if (i <= alertLvl) pip.classList.add('active-' + i);
  });

  // ── Channel pills ──
  const channels = data.channels || {};
  if (channelSound)     channelSound.classList.toggle('active', !!channels.sound);
  if (channelVibration) channelVibration.classList.toggle('active', !!channels.vibration);
  if (channelBreak)     channelBreak.classList.toggle('active', !!channels.break_suggested);

  // ── HUD overlay ──
  if (overlayStatus) {
    overlayStatus.textContent = stateStr;
    overlayStatus.style.color = alertLvl >= 3 ? 'hsl(0,78%,65%)'
                               : alertLvl === 2 ? 'hsl(28,90%,65%)'
                               : alertLvl === 1 ? 'hsl(42,85%,62%)'
                               : 'hsl(145,55%,55%)';
  }
  if (overlayProb) overlayProb.textContent = 'p=' + fmt(data.ema_prob);
  if (hudDot) hudDot.classList.toggle('alarm', on);

  // ── PERCLOS Gauge ──
  const perclosRatio = (data.perclos_ratio != null ? data.perclos_ratio : (data.perclos || 0));
  const perclosPct   = Math.min(100, Math.max(0, perclosRatio * 100));
  if (perclosBar && perclosValue) {
    perclosValue.textContent = perclosPct.toFixed(1) + '%';
    perclosBar.style.width   = perclosPct.toFixed(1) + '%';
    perclosBar.dataset.level = perclosPct >= 70 ? 'danger' : perclosPct >= 30 ? 'warn' : 'ok';
    // Legacy fallback color (for browsers without data-level CSS)
    if (perclosPct >= 70)      perclosBar.style.background = '';
    else if (perclosPct >= 30) perclosBar.style.background = '';
    else                       perclosBar.style.background = '';
  }
  if (hudPerclos) hudPerclos.textContent = perclosPct.toFixed(0) + '%';

  // ── EAR metric bar ──
  if (data.ear_smooth != null) {
    const ear = data.ear_smooth;
    const earPct = Math.min(100, Math.max(0, ear / 0.4 * 100)); // 0.4 = fully open
    if (earBar)  { earBar.style.width = earPct + '%'; earBar.style.background = ear < earClosedThreshold ? 'var(--drowsy)' : 'var(--normal)'; }
    if (earVal)  earVal.textContent = ear.toFixed(3);
    if (hudEar)  hudEar.textContent = ear.toFixed(3);
  } else if (data.features && data.features.ear_avg != null) {
    const ear = data.features.ear_avg;
    const earPct = Math.min(100, Math.max(0, ear / 0.4 * 100));
    if (earBar)  { earBar.style.width = earPct + '%'; earBar.style.background = ear < earClosedThreshold ? 'var(--drowsy)' : 'var(--normal)'; }
    if (earVal)  earVal.textContent = ear.toFixed(3);
    if (hudEar)  hudEar.textContent = ear.toFixed(3);
  }

  // ── Eye closed streak bar ──
  if (data.eye_closed_streak_ms != null) {
    const streakMs = data.eye_closed_streak_ms;
    const streakPct = Math.min(100, streakMs / 2000 * 100); // 2s = 100%
    if (eyeStreakBar) eyeStreakBar.style.width = streakPct + '%';
    if (eyeStreakVal) eyeStreakVal.textContent = streakMs >= 1000 ? (streakMs/1000).toFixed(1) + 's' : Math.round(streakMs) + 'ms';
  }

  // ── Banners ──
  const toggleBanner = (el, active) => el && (active ? el.classList.remove('hidden') : el.classList.add('hidden'));
  toggleBanner(neckBanner,    data.neck_alarm);
  toggleBanner(eyeBanner,     data.eye_alarm);
  toggleBanner(yawnBanner,    data.yawn_alarm);
  toggleBanner(obsBanner,     data.camera_obstructed);
  toggleBanner(phoneBanner,   data.phone_suspected);
  toggleBanner(lookAwayBanner,data.looking_away);

  // Debug: log every 10 frames to avoid spamming
  if (DEBUG_FUSION) {
    window._dbgFrame = ((window._dbgFrame || 0) + 1);
    if (window._dbgFrame % 10 === 0) {
      console.log('[fusion]', {
        drowsiness_state: data.drowsiness_state,
        alert_level:      data.alert_level,
        alarm_on:         data.alarm_on,
        ema_prob:         data.ema_prob,
        perclos:          perclosRatio,
        neck_alarm:       data.neck_alarm,
        eye_alarm:        data.eye_alarm,
        yawn_alarm:       data.yawn_alarm,
        camera_obstructed:data.camera_obstructed,
      });
    }
  }

  if (data.features) {
    const f = data.features;
    featEarLeft.textContent  = fmtF(f.ear_left);
    featEarRight.textContent = fmtF(f.ear_right);
    featEarAvg.textContent   = fmtF(f.ear_avg);
    featMar.textContent      = fmtF(f.mar);
    featPitch.textContent    = fmtDeg(f.pitch);
    featYaw.textContent      = fmtDeg(f.yaw);
    featRoll.textContent     = fmtDeg(f.roll);
    featNeck.textContent     = fmtDeg(f.neck_tilt);
  }

  if (!data.face_found) setStatus('Không phát hiện mặt — giữ kết quả trước đó.');
}

// ── Helpers ───────────────────────────────────────────────────────────────
function drawAnnotatedStatic(dataUrl) {
  const img = new Image();
  img.onload = () => {
    annotatedCanvas.width  = img.naturalWidth;
    annotatedCanvas.height = img.naturalHeight;
    annotatedCanvas.getContext('2d').drawImage(img, 0, 0);
  };
  img.src = dataUrl;
}

function fmt(v)    { return v != null ? v.toFixed(3) : '—'; }
function fmtF(v)   { return v != null && !isNaN(v) ? v.toFixed(3) : '—'; }
function fmtDeg(v) { return v != null && !isNaN(v) ? v.toFixed(1) + '°' : '—'; }

function addTimelineSegment(alarmOn, state) {
  if (!timeline) return;
  const seg = document.createElement('div');
  const stateName = (state || (alarmOn ? 'drowsy' : 'normal')).toLowerCase();
  seg.className = 'timeline-seg state-' + stateName;
  seg.title = stateName.toUpperCase();
  timeline.appendChild(seg);
  timeline.scrollLeft = timeline.scrollWidth;
  while (timeline.children.length > 200) timeline.removeChild(timeline.firstChild);
}

function addVideoTimelineSegment(alarmOn) {
  const seg = document.createElement('div');
  seg.className = 'tl-seg';
  seg.dataset.alarm = alarmOn ? 'on' : 'off';
  videoTimeline.appendChild(seg);
  while (videoTimeline.children.length > 300) {
    videoTimeline.removeChild(videoTimeline.firstChild);
  }
}

function formatTime(seconds) {
  const value = Number.isFinite(seconds) ? Math.max(0, Math.floor(seconds)) : 0;
  const mins = Math.floor(value / 60);
  const secs = value % 60;
  return `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
}

function formatBytes(bytes) {
  if (!Number.isFinite(bytes) || bytes <= 0) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB'];
  const index = Math.min(units.length - 1, Math.floor(Math.log(bytes) / Math.log(1024)));
  const value = bytes / (1024 ** index);
  return `${value.toFixed(index === 0 ? 0 : 1)} ${units[index]}`;
}

function setBadge(type, text) {
  systemBadge.className = 'badge badge-' + type;
  systemBadge.textContent = text;
  // Update footer dot
  if (footerDot) {
    footerDot.classList.toggle('error', type === 'error');
  }
}
function setStatus(msg) {
  if (statusBar) statusBar.textContent = msg;
}

