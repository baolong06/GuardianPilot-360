'use strict';

// ── API key (H6) ───────────────────────────────────────────────────────────
// Khi server đặt env GUARDIANPILOT_API_KEY, các endpoint chứa dữ liệu tài xế
// (/api/events, /api/trip/summary, /api/metrics) yêu cầu header X-API-Key.
// Dashboard đọc key từ localStorage — người quản lý đặt một lần bằng:
//     localStorage.setItem('gp_api_key', '<key>')
// Server không bật auth → không set gì cả, mọi thứ chạy như cũ.
function apiHeaders(extra) {
  const headers = Object.assign({}, extra || {});
  let key = null;
  try { key = localStorage.getItem('gp_api_key'); } catch (_) { /* private mode */ }
  if (key) headers['X-API-Key'] = key;
  return headers;
}

/** fetch() kèm API key; báo lỗi rõ ràng khi bị 401. */
async function apiFetch(url, options) {
  const opts = Object.assign({}, options || {});
  opts.headers = apiHeaders(opts.headers);
  const res = await fetch(url, opts);
  if (res.status === 401) {
    console.warn('[dashboard] 401 — cần API key. Đặt bằng: ' +
                 "localStorage.setItem('gp_api_key', '<key>')");
  }
  return res;
}

// ── DOM Refs ──────────────────────────────────────────────────────────────
const btnRefresh         = document.getElementById('btnRefresh');
const statActiveVehicles = document.getElementById('statActiveVehicles');
const statTotalAlerts    = document.getElementById('statTotalAlerts');
const statCriticalEvents = document.getElementById('statCriticalEvents');
const statPeakPerclos    = document.getElementById('statPeakPerclos');

const filterDriver       = document.getElementById('filterDriver');
const filterDate         = document.getElementById('filterDate');
const filterState        = document.getElementById('filterState');
const btnFilter          = document.getElementById('btnFilter');
const eventsTbody        = document.getElementById('eventsTbody');

const eventModal         = document.getElementById('eventModal');
const btnModalClose      = document.getElementById('btnModalClose');
const modalTitle         = document.getElementById('modalTitle');
const modalSnapshotImg   = document.getElementById('modalSnapshotImg');
const snapshotPlaceholder= document.getElementById('snapshotPlaceholder');
const modalInfoGrid      = document.getElementById('modalInfoGrid');
const syncStatus         = document.getElementById('syncStatus');

// Canvas refs
const chartHourlyCanvas  = document.getElementById('chartHourly');
const chartDistCanvas    = document.getElementById('chartDistribution');

// ── State ─────────────────────────────────────────────────────────────────
let loadedEvents = [];

// ── State names map ───────────────────────────────────────────────────────
const LEVEL_STATE_MAP = {
  0: 'NORMAL',
  1: 'FATIGUE',
  2: 'DROWSY',
  3: 'MICROSLEEP',
  4: 'CRITICAL',
};

// ── Init ──────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  // Today date default in picker
  const today = new Date().toISOString().split('T')[0];
  if (filterDate) filterDate.value = today;

  loadDashboardData();

  // Auto-refresh every 10 seconds
  setInterval(loadDashboardData, 10000);
});

if (btnRefresh) btnRefresh.addEventListener('click', loadDashboardData);
if (btnFilter)  btnFilter.addEventListener('click', loadDashboardData);
if (btnModalClose) btnModalClose.addEventListener('click', closeModal);
if (eventModal) {
  eventModal.addEventListener('click', (e) => {
    if (e.target === eventModal) closeModal();
  });
}

// ── Load Dashboard Data ───────────────────────────────────────────────────
async function loadDashboardData() {
  if (syncStatus) syncStatus.textContent = 'Đang cập nhật...';
  try {
    const driverId = filterDriver ? filterDriver.value.trim() : '';
    const dateVal  = filterDate ? filterDate.value : '';
    const stateVal = filterState ? filterState.value : '';

    let url = `/api/events?limit=100`;
    if (driverId) url += `&driver_id=${encodeURIComponent(driverId)}`;
    if (dateVal)  url += `&date=${encodeURIComponent(dateVal)}`;

    const [eventsRes, tripRes] = await Promise.all([
      apiFetch(url),
      apiFetch('/api/trip/summary').catch(() => null),
    ]);

    const eventsData = await eventsRes.json();
    let tripData = null;
    if (tripRes) tripData = await tripRes.json().catch(() => null);

    if (eventsData.ok) {
      let events = eventsData.events || [];
      if (stateVal) {
        events = events.filter(e => {
          const s = LEVEL_STATE_MAP[e.alert_level] || 'NORMAL';
          return s === stateVal;
        });
      }
      loadedEvents = events;
      renderTable(events);
      renderStats(events, tripData);
      renderHourlyChart(events);
      renderDistributionChart(events);
    }
  } catch (err) {
    console.error('[Dashboard] Error loading data:', err);
  } finally {
    if (syncStatus) syncStatus.textContent = 'Cập nhật tự động (10s)';
  }
}

// ── Render Stats Overview ─────────────────────────────────────────────────
function renderStats(events, tripData) {
  const total = events.length;
  const criticals = events.filter(e => e.alert_level >= 4).length;

  let maxPerclos = 0;
  events.forEach(e => {
    if (e.perclos && e.perclos > maxPerclos) maxPerclos = e.perclos;
  });

  if (tripData && tripData.perclos_peak) {
    maxPerclos = Math.max(maxPerclos, tripData.perclos_peak);
  }

  // Count unique vehicles
  const vehicles = new Set(events.map(e => e.vehicle_id || 'vehicle_demo'));
  if (vehicles.size === 0) vehicles.add('vehicle_demo');

  if (statTotalAlerts)    statTotalAlerts.textContent    = total;
  if (statCriticalEvents) statCriticalEvents.textContent = criticals;
  if (statPeakPerclos)    statPeakPerclos.textContent    = (maxPerclos * 100).toFixed(1) + '%';
  if (statActiveVehicles) statActiveVehicles.textContent = vehicles.size;
}

// ── Render Event Log Table ────────────────────────────────────────────────
function renderTable(events) {
  if (!eventsTbody) return;
  if (!events || events.length === 0) {
    eventsTbody.innerHTML = `<tr><td colspan="10" class="empty-msg">Không có sự kiện cảnh báo nào phù hợp.</td></tr>`;
    return;
  }

  eventsTbody.innerHTML = events.map(e => {
    const level = e.alert_level || 0;
    const stateName = LEVEL_STATE_MAP[level] || 'NORMAL';
    const badgeClass = `state-${stateName.toLowerCase()}`;
    const timeStr = formatTimestamp(e.timestamp);
    const perclosStr = e.perclos != null ? (e.perclos * 100).toFixed(1) + '%' : '—';
    const earStr = e.ear_avg != null ? e.ear_avg.toFixed(3) : '—';
    const neckStr = e.neck_tilt != null ? e.neck_tilt.toFixed(1) + '°' : '—';

    let gpsStr = '—';
    if (e.gps_lat != null && e.gps_lng != null) {
      gpsStr = `<a href="https://maps.google.com/?q=${e.gps_lat},${e.gps_lng}" target="_blank" title="Xem vị trí Google Maps">📍 Map</a>`;
    }

    return `
      <tr>
        <td>#${e.id}</td>
        <td>${timeStr}</td>
        <td>${escapeHtml(e.vehicle_id || 'vehicle_demo')}</td>
        <td>${escapeHtml(e.driver_id || 'driver_demo')}</td>
        <td><span class="badge-state ${badgeClass}">${stateName} (Cấp ${level})</span></td>
        <td>${perclosStr}</td>
        <td>${earStr}</td>
        <td>${neckStr}</td>
        <td>${gpsStr}</td>
        <td><button class="btn btn-ghost" onclick="viewEventDetail(${e.id})">🔍 Chi tiết</button></td>
      </tr>
    `;
  }).join('');
}

// ── Event Detail Modal ────────────────────────────────────────────────────
window.viewEventDetail = function(eventId) {
  const ev = loadedEvents.find(e => e.id === eventId);
  if (!ev) return;

  if (modalTitle) modalTitle.textContent = `Chi Tiết Sự Kiện #${ev.id} — Xe ${ev.vehicle_id || 'vehicle_demo'}`;

  // Reset snapshot view
  if (modalSnapshotImg) {
    modalSnapshotImg.classList.add('hidden');
    modalSnapshotImg.src = '';
  }
  if (snapshotPlaceholder) snapshotPlaceholder.classList.remove('hidden');

  // Load snapshot image if present
  if (ev.snapshot_path) {
    apiFetch(`/api/events/${ev.id}/snapshot`)
      .then(res => {
        if (res.ok) return res.blob();
        throw new Error('No snapshot');
      })
      .then(blob => {
        const url = URL.createObjectURL(blob);
        if (modalSnapshotImg && snapshotPlaceholder) {
          modalSnapshotImg.src = url;
          modalSnapshotImg.classList.remove('hidden');
          snapshotPlaceholder.classList.add('hidden');
        }
      })
      .catch(() => {
        if (snapshotPlaceholder) {
          snapshotPlaceholder.innerHTML = `<span>📷 File snapshot không tồn tại<br>(đã bị xóa hoặc ở chế độ metadata-only)</span>`;
        }
      });
  }

  // Populate Grid
  const level = ev.alert_level || 0;
  const stateName = LEVEL_STATE_MAP[level] || 'NORMAL';

  if (modalInfoGrid) {
    modalInfoGrid.innerHTML = `
      <div class="info-item"><span>Mức Cảnh Báo:</span><strong>${stateName} (Cấp ${level})</strong></div>
      <div class="info-item"><span>Thời Gian:</span><strong>${formatTimestamp(ev.timestamp)}</strong></div>
      <div class="info-item"><span>Mã Xe (Vehicle ID):</span><strong>${escapeHtml(ev.vehicle_id || 'vehicle_demo')}</strong></div>
      <div class="info-item"><span>Tài Xế (Driver ID):</span><strong>${escapeHtml(ev.driver_id || 'driver_demo')}</strong></div>
      <div class="info-item"><span>PERCLOS (30s):</span><strong>${ev.perclos != null ? (ev.perclos * 100).toFixed(1) + '%' : '—'}</strong></div>
      <div class="info-item"><span>EAR Trung Bình:</span><strong>${ev.ear_avg != null ? ev.ear_avg.toFixed(3) : '—'}</strong></div>
      <div class="info-item"><span>Neck Tilt Angle:</span><strong>${ev.neck_tilt != null ? ev.neck_tilt.toFixed(1) + '°' : '—'}</strong></div>
      <div class="info-item"><span>Đã Đồng Bộ Cloud:</span><strong>${ev.uploaded ? '✅ Có' : '⏳ Chờ sync'}</strong></div>
    `;
  }

  if (eventModal) eventModal.classList.remove('hidden');
};

function closeModal() {
  if (eventModal) eventModal.classList.add('hidden');
}

// ── Native HTML5 Canvas Charts ────────────────────────────────────────────

// 1. Hourly Chart (Bar Chart)
function renderHourlyChart(events) {
  if (!chartHourlyCanvas) return;
  const ctx = chartHourlyCanvas.getContext('2d');
  const w = chartHourlyCanvas.width;
  const h = chartHourlyCanvas.height;

  ctx.clearRect(0, 0, w, h);

  // Group events by hour (0..23)
  const hourlyCounts = new Array(24).fill(0);
  events.forEach(e => {
    try {
      const d = new Date(e.timestamp);
      const hour = d.getHours();
      if (!isNaN(hour) && hour >= 0 && hour < 24) hourlyCounts[hour]++;
    } catch (_) {}
  });

  const maxVal = Math.max(...hourlyCounts, 5);
  const padLeft = 30, padBottom = 25, padTop = 10, padRight = 10;
  const chartW = w - padLeft - padRight;
  const chartH = h - padBottom - padTop;
  const barW = chartW / 24;

  // Grid lines
  ctx.strokeStyle = '#3a3733';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(padLeft, padTop + chartH);
  ctx.lineTo(padLeft + chartW, padTop + chartH);
  ctx.stroke();

  // Bars
  for (let i = 0; i < 24; i++) {
    const val = hourlyCounts[i];
    const barHeight = (val / maxVal) * chartH;
    const x = padLeft + i * barW + 2;
    const y = padTop + chartH - barHeight;

    ctx.fillStyle = val > 0 ? '#d97757' : '#2e2c2a';
    ctx.fillRect(x, y, barW - 4, barHeight);

    // Label every 4 hours
    if (i % 4 === 0) {
      ctx.fillStyle = '#9a9288';
      ctx.font = '10px monospace';
      ctx.fillText(`${i}h`, x, padTop + chartH + 15);
    }
  }
}

// 2. State Distribution Chart (Doughnut Chart)
function renderDistributionChart(events) {
  if (!chartDistCanvas) return;
  const ctx = chartDistCanvas.getContext('2d');
  const w = chartDistCanvas.width;
  const h = chartDistCanvas.height;

  ctx.clearRect(0, 0, w, h);

  const counts = { FATIGUE: 0, DROWSY: 0, MICROSLEEP: 0, CRITICAL: 0 };
  events.forEach(e => {
    const s = LEVEL_STATE_MAP[e.alert_level];
    if (s && counts[s] !== undefined) counts[s]++;
  });

  const total = Object.values(counts).reduce((a, b) => a + b, 0);

  if (total === 0) {
    ctx.fillStyle = '#9a9288';
    ctx.font = '12px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Chưa có cảnh báo', w / 2, h / 2);
    return;
  }

  const colors = {
    FATIGUE: '#c49a3c',
    DROWSY: '#d97757',
    MICROSLEEP: '#ff6b6b',
    CRITICAL: '#ff4444',
  };

  const cx = w / 3;
  const cy = h / 2;
  const radius = Math.min(cx, cy) - 10;
  let startAngle = -Math.PI / 2;

  Object.keys(counts).forEach(st => {
    const val = counts[st];
    if (val === 0) return;
    const sliceAngle = (val / total) * 2 * Math.PI;

    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.arc(cx, cy, radius, startAngle, startAngle + sliceAngle);
    ctx.closePath();
    ctx.fillStyle = colors[st];
    ctx.fill();

    startAngle += sliceAngle;
  });

  // Legend
  const legendX = (w / 3) * 2 - 10;
  let legendY = 30;
  ctx.textAlign = 'left';

  Object.keys(counts).forEach(st => {
    const val = counts[st];
    ctx.fillStyle = colors[st];
    ctx.fillRect(legendX, legendY, 12, 12);

    ctx.fillStyle = '#f0ebe4';
    ctx.font = '11px monospace';
    ctx.fillText(`${st}: ${val}`, legendX + 18, legendY + 10);
    legendY += 22;
  });
}

// ── Helpers ───────────────────────────────────────────────────────────────
function formatTimestamp(ts) {
  if (!ts) return '—';
  try {
    const d = new Date(ts);
    if (isNaN(d.getTime())) return ts;
    return d.toLocaleString('vi-VN', { hour12: false });
  } catch (_) {
    return ts;
  }
}

function escapeHtml(str) {
  if (!str) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}
