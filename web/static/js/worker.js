/**
 * worker.js — Inference Worker
 *
 * Chạy độc lập với display loop.
 * Nhận frame từ main thread, gọi /api/analyze_lite, trả kết quả về.
 *
 * Messages vào (from main):
 *   { type: 'frame', dataUrl, timestamp }
 *   { type: 'stop' }
 *
 * Messages ra (to main):
 *   { type: 'result', data, inferenceMs, timestamp }
 *   { type: 'error', message }
 *   { type: 'log', message }
 */

'use strict';

// Fix: Worker has null origin → absolute fetch fails CORS.
// Empty-string relative path works correctly inside a Worker.
const WORKER_API_BASE = '';

let running = false;
let pendingFrame = null;      // frame mới nhất đang chờ xử lý
let processing  = false;      // đang trong 1 inference call

// H2: worker PHẢI dùng cùng session_id với main thread, nếu không server sẽ
// tạo hai session riêng và vòng lặp live chạy trên state khác với các request
// từ main thread (init / reset / video).
let sessionId = 'default';

// Interval loop — polling pendingFrame (mặc định 100ms, edge profile: 200ms)
let INFERENCE_INTERVAL_MS = 100;

let loopHandle = null;

function startLoop() {
  if (loopHandle) return;
  running = true;
  loopHandle = setInterval(async () => {
    if (!running || processing || !pendingFrame) return;

    const { dataUrl, timestamp } = pendingFrame;
    pendingFrame = null;  // consume
    processing = true;

    const t0 = Date.now();
    try {
      const res = await fetch(WORKER_API_BASE + '/api/analyze_lite', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-Session-Id': sessionId,
        },
        body: JSON.stringify({ image: dataUrl, session_id: sessionId }),
      });

      if (!res.ok) {
        let detail = '';
        try { detail = await res.text(); } catch (_) {}
        self.postMessage({ type: 'error', message: `HTTP ${res.status}: ${detail.slice(0, 120)}` });
        return;
      }

      const data = await res.json();
      const inferenceMs = Date.now() - t0;

      self.postMessage({
        type: 'result',
        data,
        inferenceMs,
        timestamp,
      });
    } catch (err) {
      self.postMessage({ type: 'error', message: err.message });
    } finally {
      processing = false;
    }
  }, INFERENCE_INTERVAL_MS);
}

function stopLoop() {
  if (loopHandle) {
    clearInterval(loopHandle);
    loopHandle = null;
  }
  running    = false;
  processing = false;
  pendingFrame = null;
}

self.onmessage = (e) => {
  const msg = e.data;

  if (msg.type === 'start') {
    running = true;
    if (msg.sessionId) sessionId = msg.sessionId;
    if (msg.profile && msg.profile.inference_interval_ms) {
      INFERENCE_INTERVAL_MS = msg.profile.inference_interval_ms;
    }
    stopLoop();
    startLoop();
    self.postMessage({ type: 'log', message: 'Worker started.' });

  } else if (msg.type === 'frame') {
    if (!running) return;
    // Luôn ghi đè frame cũ — chỉ giữ frame MỚI NHẤT (drop nếu chưa kịp xử lý)
    pendingFrame = { dataUrl: msg.dataUrl, timestamp: msg.timestamp };

  } else if (msg.type === 'stop') {
    stopLoop();
    self.postMessage({ type: 'log', message: 'Worker stopped.' });
  }
};
