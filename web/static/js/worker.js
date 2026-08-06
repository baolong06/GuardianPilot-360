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

let running = false;
let pendingFrame = null;      // frame mới nhất đang chờ xử lý
let processing  = false;      // đang trong 1 inference call

// Interval loop — polling pendingFrame mỗi ~80ms (≈12 FPS inference)
// Điều này tách hoàn toàn khỏi display loop
const INFERENCE_INTERVAL_MS = 100; // ~10 inference/s — đủ cho drowsiness detection

let loopHandle = null;

function startLoop() {
  if (loopHandle) return;
  loopHandle = setInterval(async () => {
    if (!running || processing || !pendingFrame) return;

    const { dataUrl, timestamp } = pendingFrame;
    pendingFrame = null;  // consume
    processing = true;

    const t0 = Date.now();
    try {
      const res = await fetch('/api/analyze_lite', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: dataUrl }),
      });

      if (!res.ok) {
        self.postMessage({ type: 'error', message: `HTTP ${res.status}` });
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
