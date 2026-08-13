// Diagnose GuardianPilot live loop via Playwright
// - Mở trang, capture console + network
// - Tạo fake video stream qua getUserMedia override (canvas captureStream)
// - Click "Phân tích live" và quan sát 10s
// - Trực tiếp POST /api/analyze_lite để xác nhận pipeline tới fused
const { chromium } = require('playwright');
const fs = require('fs');

const URL = 'http://127.0.0.1:5000/';
const OUT = 'e:/KhoiNghiep/GuardianPilot/tools/diagnose_live.log';

function log(...a) {
  const line = '[diag] ' + a.join(' ');
  console.log(line);
  fs.appendFileSync(OUT, line + '\n');
}

(async () => {
  fs.writeFileSync(OUT, '--- start ---\n');
  const browser = await chromium.launch({
    headless: true,
    args: ['--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream'],
  });
  const ctx = await browser.newContext({
    permissions: ['camera', 'microphone'],
  });
  const page = await ctx.newPage();

  page.on('console', m => log('CONSOLE', m.type(), m.text()));
  page.on('pageerror', e => log('PAGEERROR', e.message));
  page.on('request', r => { if (r.url().includes('analyze')) log('REQ', r.method(), r.url()); });
  page.on('response', r => {
    if (r.url().includes('analyze') || r.url().includes('init') || r.url().includes('runtime'))
      log('RES', r.status(), r.url());
  });

  log('open', URL);
  await page.goto(URL, { waitUntil: 'networkidle', timeout: 30000 });

  // Wait for runtime profile
  await page.waitForTimeout(1500);

  // Check initial state
  let state = await page.evaluate(() => ({
    btnInit: !!document.getElementById('btnInit'),
    btnStartCam: { disabled: document.getElementById('btnStartCam')?.disabled },
    btnAnalyzeLive: { disabled: document.getElementById('btnAnalyzeLive')?.disabled },
    status: document.getElementById('statusBar')?.textContent,
  }));
  log('initial', JSON.stringify(state));

  // Click khởi tạo
  log('click btnInit');
  await page.click('#btnInit');
  await page.waitForTimeout(2000);
  state = await page.evaluate(() => ({
    btnStartCam: { disabled: document.getElementById('btnStartCam')?.disabled },
    btnAnalyzeLive: { disabled: document.getElementById('btnAnalyzeLive')?.disabled },
    status: document.getElementById('statusBar')?.textContent,
  }));
  log('after init', JSON.stringify(state));

  // Click bật webcam
  log('click btnStartCam');
  await page.click('#btnStartCam');
  await page.waitForTimeout(3000);
  state = await page.evaluate(() => ({
    btnStartCam: { disabled: document.getElementById('btnStartCam')?.disabled },
    btnAnalyzeLive: { disabled: document.getElementById('btnAnalyzeLive')?.disabled },
    status: document.getElementById('statusBar')?.textContent,
    camReadyState: document.getElementById('webcam')?.readyState,
    camW: document.getElementById('webcam')?.videoWidth,
    camH: document.getElementById('webcam')?.videoHeight,
  }));
  log('after enableCam', JSON.stringify(state));

  // Click Phân tích live
  if (!state.btnAnalyzeLive.disabled) {
    log('click btnAnalyzeLive');
    await page.click('#btnAnalyzeLive');
    await page.waitForTimeout(8000);
    state = await page.evaluate(() => ({
      btnStopLive: { disabled: document.getElementById('btnStopLive')?.disabled },
      status: document.getElementById('statusBar')?.textContent,
      liveActive: window.liveActive,
      hasWorker: !!window.inferenceWorker,
    }));
    log('after startLive', JSON.stringify(state));
  } else {
    log('btnAnalyzeLive still disabled - skip');
  }

  // Direct probe: POST /api/analyze_lite directly from page context
  log('direct probe /api/analyze_lite');
  const direct = await page.evaluate(async () => {
    const r = await fetch('/api/analyze_lite', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        image: 'data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAA0JCgsKCA0LCgsODg0PEyAVExISEyccHhcgLikxMC4pLSwzOko+MzZGNywtQFdBRkxOUlNSMj5aYVpQYEpRUk//2wBDAQ4ODhMREyYVFSZPNS01T09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT0//wAARCABAAEADASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAv/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFAEBAAAAAAAAAAAAAAAAAAAAAP/EABQRAQAAAAAAAAAAAAAAAAAAAAD/2gAMAwEAAhEDEQA/AL+AAAAAAAA//9k='
      }),
    });
    return { status: r.status, body: (await r.text()).slice(0, 800) };
  });
  log('direct', JSON.stringify(direct));

  // Server metrics
  const metrics = await page.evaluate(() => fetch('/api/metrics').then(r => r.json()));
  log('server-metrics', JSON.stringify(metrics));

  await browser.close();
  log('done');
})().catch(e => {
  log('FATAL', e.stack || e.message);
  process.exit(1);
});
