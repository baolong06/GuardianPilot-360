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
    eventsTbody.innerHTML = `<tr><td colspan="9" style="text-align: center; color: var(--text-3); padding: 2.5rem;">Không có sự kiện cảnh báo nào phù hợp với bộ lọc.</td></tr>`;
    return;
  }

  eventsTbody.innerHTML = events.map(e => {
    const level = e.alert_level || 0;
    const stateName = LEVEL_STATE_MAP[level] || 'NORMAL';
    const badgeClass = `level-${level}`;
    const timeStr = formatTimestamp(e.timestamp);
    const perclosStr = e.perclos != null ? (e.perclos * 100).toFixed(1) + '%' : '—';
    const speedStr = e.speed_kmh != null ? `${e.speed_kmh.toFixed(0)} km/h` : '0 km/h';
    const scoreStr = e.drowsiness_score != null ? e.drowsiness_score.toFixed(3) : (e.ema_prob != null ? e.ema_prob.toFixed(3) : '—');
    const driverVehicle = `${escapeHtml(e.driver_id || 'driver_1')} / ${escapeHtml(e.vehicle_id || 'truck_01')}`;

    return `
      <tr>
        <td style="font-family: var(--font-mono); color: var(--text-3)">#${e.id}</td>
        <td style="white-space: nowrap">${timeStr}</td>
        <td><strong>${driverVehicle}</strong></td>
        <td><span class="level-badge ${badgeClass}">Cấp ${level}</span></td>
        <td><strong>${stateName}</strong></td>
        <td class="perclos-pill">${perclosStr}</td>
        <td style="font-family: var(--font-mono); color: var(--text-2)">${scoreStr}</td>
        <td style="font-family: var(--font-mono)">${speedStr}</td>
        <td><button class="btn btn-ghost" style="padding: 0.25rem 0.6rem; min-height: 28px; font-size: 0.75rem;" onclick="viewEventDetail(${e.id})">🔍 Xem</button></td>
      </tr>
    `;
  }).join('');
}

// ── Event Detail Modal ────────────────────────────────────────────────────
window.viewEventDetail = function(eventId) {
  const ev = loadedEvents.find(e => e.id === eventId);
  if (!ev) return;

  if (modalTitle) modalTitle.textContent = `Chi Tiết Sự Kiện #${ev.id} — ${ev.vehicle_id || 'truck_01'}`;

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
          snapshotPlaceholder.innerHTML = `<span>📷 File snapshot không tồn tại hoặc đã được nén lưu trữ</span>`;
        }
      });
  }

  // Populate Grid
  const level = ev.alert_level || 0;
  const stateName = LEVEL_STATE_MAP[level] || 'NORMAL';

  if (modalInfoGrid) {
    modalInfoGrid.innerHTML = `
      <div class="modal-info-item"><span class="modal-info-label">Mức Cảnh Báo</span><strong class="modal-info-value"><span class="level-badge level-${level}">${stateName} (Cấp ${level})</span></strong></div>
      <div class="modal-info-item"><span class="modal-info-label">Thời Gian</span><strong class="modal-info-value">${formatTimestamp(ev.timestamp)}</strong></div>
      <div class="modal-info-item"><span class="modal-info-label">Mã Xe / Tài Xế</span><strong class="modal-info-value">${escapeHtml(ev.vehicle_id || 'truck_01')} / ${escapeHtml(ev.driver_id || 'driver_1')}</strong></div>
      <div class="modal-info-item"><span class="modal-info-label">PERCLOS (30s)</span><strong class="modal-info-value">${ev.perclos != null ? (ev.perclos * 100).toFixed(1) + '%' : '—'}</strong></div>
      <div class="modal-info-item"><span class="modal-info-label">EAR Trung Bình</span><strong class="modal-info-value">${ev.ear_avg != null ? ev.ear_avg.toFixed(3) : '—'}</strong></div>
      <div class="modal-info-item"><span class="modal-info-label">Neck Tilt</span><strong class="modal-info-value">${ev.neck_tilt != null ? ev.neck_tilt.toFixed(1) + '°' : '—'}</strong></div>
      <div class="modal-info-item"><span class="modal-info-label">Vận Tốc Xe</span><strong class="modal-info-value">${ev.speed_kmh != null ? ev.speed_kmh.toFixed(0) + ' km/h' : '0 km/h'}</strong></div>
      <div class="modal-info-item"><span class="modal-info-label">Đồng Bộ Cloud</span><strong class="modal-info-value">${ev.uploaded ? '✅ Đã đồng bộ' : '⏳ Lưu cục bộ'}</strong></div>
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
  const padLeft = 30, padBottom = 25, padTop = 15, padRight = 10;
  const chartW = w - padLeft - padRight;
  const chartH = h - padBottom - padTop;
  const barW = chartW / 24;

  // Grid line
  ctx.strokeStyle = '#2a2e34';
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

    ctx.fillStyle = val > 0 ? '#3b82f6' : '#1c1f22';
    if (val > 0) {
      // Rounded top bar
      ctx.beginPath();
      ctx.roundRect ? ctx.roundRect(x, y, barW - 4, barHeight, [3, 3, 0, 0]) : ctx.rect(x, y, barW - 4, barHeight);
      ctx.fill();
    } else {
      ctx.fillRect(x, y, barW - 4, Math.max(2, barHeight));
    }

    // Label every 4 hours
    if (i % 4 === 0) {
      ctx.fillStyle = '#7e8a97';
      ctx.font = '10px monospace';
      ctx.fillText(`${i}h`, x, padTop + chartH + 16);
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
    ctx.fillStyle = '#7e8a97';
    ctx.font = '12px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Chưa ghi nhận cảnh báo', w / 2, h / 2);
    return;
  }

  const colors = {
    FATIGUE: 'hsl(42, 85%, 52%)',
    DROWSY: 'hsl(28, 90%, 55%)',
    MICROSLEEP: 'hsl(0, 78%, 55%)',
    CRITICAL: 'hsl(350, 85%, 50%)',
  };

  const cx = w / 3;
  const cy = h / 2;
  const outerRadius = Math.min(cx, cy) - 12;
  const innerRadius = outerRadius * 0.55;
  let startAngle = -Math.PI / 2;

  Object.keys(counts).forEach(st => {
    const val = counts[st];
    if (val === 0) return;
    const sliceAngle = (val / total) * 2 * Math.PI;

    ctx.beginPath();
    ctx.arc(cx, cy, outerRadius, startAngle, startAngle + sliceAngle);
    ctx.arc(cx, cy, innerRadius, startAngle + sliceAngle, startAngle, true);
    ctx.closePath();
    ctx.fillStyle = colors[st];
    ctx.fill();

    startAngle += sliceAngle;
  });

  // Center text (Total count)
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillStyle = '#e8edf2';
  ctx.font = 'bold 16px monospace';
  ctx.fillText(`${total}`, cx, cy - 2);
  ctx.fillStyle = '#7e8a97';
  ctx.font = '9px sans-serif';
  ctx.fillText('TỔNG', cx, cy + 12);

  // Legend
  const legendX = (w / 3) * 2 - 10;
  let legendY = 25;
  ctx.textAlign = 'left';
  ctx.textBaseline = 'alphabetic';

  Object.keys(counts).forEach(st => {
    const val = counts[st];
    ctx.fillStyle = colors[st];
    ctx.beginPath();
    ctx.roundRect ? ctx.roundRect(legendX, legendY, 10, 10, 2) : ctx.rect(legendX, legendY, 10, 10);
    ctx.fill();

    ctx.fillStyle = '#e8edf2';
    ctx.font = '11px monospace';
    ctx.fillText(`${st}: ${val}`, legendX + 16, legendY + 9);
    legendY += 24;
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
