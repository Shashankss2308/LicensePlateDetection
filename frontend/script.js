/* ==========================================
   PlateVision AI — Frontend Logic v2
   - Particle background
   - Upload / Drag-drop
   - ML Detection API
   - History (load, add, delete, clear)
   - Sidebar toggle (desktop collapse / mobile overlay)
   - Sidebar panels: History | About
   ========================================== */

// ── Particles ────────────────────────────────────────────────────────────────
(function initParticles() {
  const container = document.getElementById('particles');
  const colors = ['#00d4ff', '#7b2ff7', '#ff2d78'];
  for (let i = 0; i < 55; i++) {
    const p = document.createElement('div');
    p.className = 'particle';
    const size = Math.random() * 3 + 1;
    p.style.cssText = `
      left: ${Math.random() * 100}%;
      width: ${size}px; height: ${size}px;
      background: ${colors[Math.floor(Math.random() * colors.length)]};
      animation-duration: ${Math.random() * 15 + 8}s;
      animation-delay: ${Math.random() * 15}s;
    `;
    container.appendChild(p);
  }
})();


// ── DOM refs ─────────────────────────────────────────────────────────────────
const imageInput = document.getElementById('imageInput');
const dropZone = document.getElementById('dropZone');
const browseBtn = document.getElementById('browseBtn');
const uploadIdle = document.getElementById('uploadIdle');
const uploadPreview = document.getElementById('uploadPreview');
const previewImg = document.getElementById('previewImg');
const detectBtn = document.getElementById('detectBtn');
const resetBtn = document.getElementById('resetBtn');
const newBtn = document.getElementById('newBtn');
const retryBtn = document.getElementById('retryBtn');
const copyBtn = document.getElementById('copyBtn');

const resultIdle = document.getElementById('resultIdle');
const resultProcessing = document.getElementById('resultProcessing');
const resultOutput = document.getElementById('resultOutput');
const resultError = document.getElementById('resultError');

const scanOverlay = document.getElementById('scanOverlay');
const plateBox = document.getElementById('plateBox');
const plateText = document.getElementById('plateText');
const rawText = document.getElementById('rawText');
const processTime = document.getElementById('processTime');
const confidenceValue = document.getElementById('confidenceValue');

const historyList = document.getElementById('historyList');
const historyEmpty = document.getElementById('historyEmpty');
const historyCount = document.getElementById('historyCount');

const sidebar = document.getElementById('sidebar');
const sidebarToggle = document.getElementById('sidebarToggle');
const sidebarOverlay = document.getElementById('sidebarOverlay');
const navHistory = document.getElementById('navHistory');
const navAbout = document.getElementById('navAbout');
const historyPanel = document.getElementById('historyPanel');
const aboutPanel = document.getElementById('aboutPanel');

const steps = ['step1', 'step2', 'step3', 'step4'].map(id => document.getElementById(id));

// ── State ─────────────────────────────────────────────────────────────────────
let currentFile = null;
let detectedPlateText = '';
let historyCache = {};   // id → detection object (for click-to-view)


// ══════════════════════════════════════════════════════════════════════════════
// SIDEBAR
// ══════════════════════════════════════════════════════════════════════════════
function isMobile() { return window.innerWidth <= 768; }

sidebarToggle.addEventListener('click', toggleSidebar);

function toggleSidebar() {
  if (isMobile()) {
    sidebar.classList.toggle('mobile-open');
    sidebarOverlay.classList.toggle('active');
  } else {
    sidebar.classList.toggle('collapsed');
  }
}

function closeSidebar() {
  sidebar.classList.remove('mobile-open');
  sidebarOverlay.classList.remove('active');
}

// On resize — clean up classes that don't apply to current breakpoint
window.addEventListener('resize', () => {
  if (!isMobile()) {
    sidebar.classList.remove('mobile-open');
    sidebarOverlay.classList.remove('active');
  } else {
    // Mobile: if it was "uncollapsed" on desktop, keep it hidden on mobile
    if (!sidebar.classList.contains('mobile-open')) {
      sidebar.classList.remove('collapsed');
    }
  }
});


// ══════════════════════════════════════════════════════════════════════════════
// SIDEBAR PANELS  (History / About)
// ══════════════════════════════════════════════════════════════════════════════
function switchPanel(panel) {
  if (panel === 'history') {
    historyPanel.style.display = 'flex';
    aboutPanel.style.display = 'none';
    navHistory.classList.add('active');
    navAbout.classList.remove('active');
  } else {
    historyPanel.style.display = 'none';
    aboutPanel.style.display = 'flex';
    navAbout.classList.add('active');
    navHistory.classList.remove('active');
  }
}


// ══════════════════════════════════════════════════════════════════════════════
// UPLOAD
// ══════════════════════════════════════════════════════════════════════════════
browseBtn.addEventListener('click', (e) => { e.stopPropagation(); imageInput.click(); });
dropZone.addEventListener('click', () => { if (uploadIdle.style.display !== 'none') imageInput.click(); });
imageInput.addEventListener('change', (e) => { const f = e.target.files[0]; if (f) loadFile(f); });

dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.classList.add('drag-over'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  const f = e.dataTransfer.files[0];
  if (f && f.type.startsWith('image/')) loadFile(f);
});

function loadFile(file) {
  currentFile = file;
  const url = URL.createObjectURL(file);
  previewImg.src = url;
  previewImg.onload = () => URL.revokeObjectURL(url);

  uploadIdle.style.display = 'none';
  uploadPreview.style.display = 'flex';
  scanOverlay.style.display = 'none';
  plateBox.style.display = 'none';
  showResultState('idle');
}


// ══════════════════════════════════════════════════════════════════════════════
// RESET
// ══════════════════════════════════════════════════════════════════════════════
function resetAll() {
  currentFile = null;
  imageInput.value = '';
  previewImg.src = '';
  uploadPreview.style.display = 'none';
  uploadIdle.style.display = 'flex';
  scanOverlay.style.display = 'none';
  plateBox.style.display = 'none';
  showResultState('idle');
}

resetBtn.addEventListener('click', resetAll);
newBtn.addEventListener('click', resetAll);
retryBtn.addEventListener('click', resetAll);


// ══════════════════════════════════════════════════════════════════════════════
// RESULT STATE SWITCHER
// ══════════════════════════════════════════════════════════════════════════════
function showResultState(state) {
  resultIdle.style.display = state === 'idle' ? 'flex' : 'none';
  resultProcessing.style.display = state === 'processing' ? 'flex' : 'none';
  resultOutput.style.display = state === 'output' ? 'flex' : 'none';
  resultError.style.display = state === 'error' ? 'flex' : 'none';
}


// ══════════════════════════════════════════════════════════════════════════════
// DETECTION
// ══════════════════════════════════════════════════════════════════════════════
detectBtn.addEventListener('click', async () => {
  if (!currentFile) return;

  detectBtn.disabled = true;
  scanOverlay.style.display = 'block';
  plateBox.style.display = 'none';
  showResultState('processing');
  resetSteps();

  try {
    await animateStep(0, 350);

    const formData = new FormData();
    formData.append('image', currentFile);

    const [response] = await Promise.all([
      fetch('/api/detect', { method: 'POST', body: formData }),
      animateStep(1, 500),
    ]);

    if (!response.ok) throw new Error(`Server error ${response.status}`);

    const result = await response.json();
    await animateStep(2, 500);
    await animateStep(3, 280);

    if (result.success) {
      if (result.bbox) showPlateBox(result.bbox);

      detectedPlateText = result.plateText || '';
      plateText.textContent = result.plateText || '—';
      rawText.textContent = result.rawText || result.plateText || '—';
      processTime.textContent = result.processTime ? result.processTime + 's' : '—';
      confidenceValue.textContent = result.confidence ? result.confidence + '%' : '—';

      showResultState('output');

      // ── Add to history sidebar ──────────────────────────────────────────────
      prependHistoryItem({
        id: result.id,
        plate_text: result.plateText,
        confidence: result.confidence,
        process_time: result.processTime,
        timestamp: result.timestamp,
        thumbnail: result.thumbnail,
      });

    } else {
      scanOverlay.style.display = 'none';
      document.getElementById('errorDesc').textContent =
        result.error || 'No license plate detected. Try a clearer image.';
      showResultState('error');
    }

  } catch (err) {
    scanOverlay.style.display = 'none';
    document.getElementById('errorDesc').textContent =
      'Cannot reach detection server. Please ensure the backend is running.';
    showResultState('error');
    console.error('[PlateVision]', err);
  } finally {
    detectBtn.disabled = false;
  }
});


// ══════════════════════════════════════════════════════════════════════════════
// HISTORY — Load from API
// ══════════════════════════════════════════════════════════════════════════════
async function loadHistory() {
  try {
    const res = await fetch('/api/history');
    const data = await res.json();
    if (data.success && Array.isArray(data.detections)) {
      renderHistory(data.detections);
    }
  } catch (e) {
    console.warn('[PlateVision] Could not load history:', e);
  }
}

function renderHistory(detections) {
  historyList.innerHTML = '';
  historyCache = {};

  if (detections.length === 0) {
    historyEmpty.style.display = 'flex';
    historyCount.textContent = '0';
    return;
  }

  historyEmpty.style.display = 'none';
  historyCount.textContent = detections.length;

  detections.forEach(det => {
    historyCache[det.id] = det;
    historyList.appendChild(buildHistoryItem(det));
  });
}

function prependHistoryItem(det) {
  historyCache[det.id] = det;
  historyEmpty.style.display = 'none';

  const item = buildHistoryItem(det);
  historyList.insertBefore(item, historyList.firstChild);

  // Animate in
  item.style.opacity = '0';
  item.style.transform = 'translateX(-12px)';
  requestAnimationFrame(() => {
    item.style.transition = 'opacity 0.35s ease, transform 0.35s ease';
    item.style.opacity = '1';
    item.style.transform = 'translateX(0)';
  });

  // Update count
  const cur = parseInt(historyCount.textContent) || 0;
  historyCount.textContent = cur + 1;
}

function buildHistoryItem(det) {
  const div = document.createElement('div');
  div.className = 'hist-item';
  div.id = `hist-${det.id}`;

  const thumbHtml = det.thumbnail
    ? `<img class="hist-thumb" src="${det.thumbnail}" alt="plate scan" />`
    : `<div class="hist-thumb-placeholder">
         <svg width="20" height="20" viewBox="0 0 24 24" fill="none"><rect x="2" y="6" width="20" height="12" rx="3" stroke="currentColor" stroke-width="1.5" stroke-dasharray="3 2"/></svg>
       </div>`;

  div.innerHTML = `
    <div class="hist-thumb-wrap">${thumbHtml}</div>
    <div class="hist-info">
      <div class="hist-plate">${escapeHtml(det.plate_text)}</div>
      <div class="hist-meta">
        <span class="hist-conf">${det.confidence ? det.confidence + '%' : ''}</span>
        <span class="hist-time">${formatTime(det.timestamp)}</span>
      </div>
    </div>
    <button class="hist-del-btn" title="Delete" onclick="deleteHistoryItem(event, ${det.id})">
      <svg width="13" height="13" viewBox="0 0 24 24" fill="none">
        <path d="M18 6L6 18M6 6l12 12" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"/>
      </svg>
    </button>
  `;

  // Click to view result in main panel
  div.addEventListener('click', (e) => {
    if (e.target.closest('.hist-del-btn')) return;
    viewHistoryItem(det.id);
  });

  return div;
}

function viewHistoryItem(id) {
  const det = historyCache[id];
  if (!det) return;

  detectedPlateText = det.plate_text;
  plateText.textContent = det.plate_text || '—';
  rawText.textContent = det.plate_text || '—';
  processTime.textContent = det.process_time ? det.process_time + 's' : '—';
  confidenceValue.textContent = det.confidence ? det.confidence + '%' : '—';

  scanOverlay.style.display = 'none';
  showResultState('output');

  // Scroll result into view on mobile
  document.getElementById('resultPanel').scrollIntoView({ behavior: 'smooth', block: 'nearest' });

  // Close mobile sidebar
  if (isMobile()) closeSidebar();
}


// ══════════════════════════════════════════════════════════════════════════════
// HISTORY — Delete
// ══════════════════════════════════════════════════════════════════════════════
async function deleteHistoryItem(e, id) {
  e.stopPropagation();
  const item = document.getElementById(`hist-${id}`);

  if (item) {
    item.style.transition = 'opacity 0.25s ease, transform 0.25s ease';
    item.style.opacity = '0';
    item.style.transform = 'translateX(-14px)';
    setTimeout(() => item.remove(), 260);
  }

  delete historyCache[id];
  const cur = parseInt(historyCount.textContent) || 1;
  historyCount.textContent = Math.max(0, cur - 1);

  if (historyList.children.length === 0) {
    historyEmpty.style.display = 'flex';
    historyCount.textContent = '0';
  }

  try { await fetch(`/api/history/${id}`, { method: 'DELETE' }); }
  catch (e) { console.warn('[PlateVision] Delete failed:', e); }
}

async function clearAllHistory() {
  const items = historyList.querySelectorAll('.hist-item');
  items.forEach((item, i) => {
    setTimeout(() => {
      item.style.transition = 'opacity 0.2s ease, transform 0.2s ease';
      item.style.opacity = '0';
      item.style.transform = 'translateX(-12px)';
    }, i * 40);
  });

  setTimeout(() => {
    historyList.innerHTML = '';
    historyCache = {};
    historyEmpty.style.display = 'flex';
    historyCount.textContent = '0';
  }, items.length * 40 + 250);

  try { await fetch('/api/history', { method: 'DELETE' }); }
  catch (e) { console.warn('[PlateVision] Clear failed:', e); }
}


// ══════════════════════════════════════════════════════════════════════════════
// COPY
// ══════════════════════════════════════════════════════════════════════════════
copyBtn.addEventListener('click', () => {
  if (!detectedPlateText) return;
  navigator.clipboard.writeText(detectedPlateText)
    .then(() => showToast('✓  Plate text copied!'))
    .catch(() => showToast('Copy failed — please copy manually.'));
});


// ══════════════════════════════════════════════════════════════════════════════
// UTILITIES
// ══════════════════════════════════════════════════════════════════════════════
function resetSteps() { steps.forEach(s => s.classList.remove('active', 'done')); }

async function animateStep(idx, waitMs) {
  if (idx > 0) { steps[idx - 1].classList.remove('active'); steps[idx - 1].classList.add('done'); }
  steps[idx].classList.add('active');
  await delay(waitMs);
}

function showPlateBox(bbox) {
  plateBox.style.left = bbox.x + '%';
  plateBox.style.top = bbox.y + '%';
  plateBox.style.width = bbox.width + '%';
  plateBox.style.height = bbox.height + '%';
  plateBox.style.display = 'block';
}

function showToast(message) {
  let toast = document.querySelector('.toast');
  if (!toast) {
    toast = document.createElement('div');
    toast.className = 'toast';
    toast.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none"><path d="M20 6L9 17l-5-5" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/></svg><span></span>`;
    document.body.appendChild(toast);
  }
  toast.querySelector('span').textContent = message;
  toast.classList.add('show');
  setTimeout(() => toast.classList.remove('show'), 2800);
}

function delay(ms) { return new Promise(r => setTimeout(r, ms)); }

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

function formatTime(ts) {
  if (!ts) return '';
  try {
    const d = new Date(ts.replace(' ', 'T'));
    const now = new Date();
    const diff = now - d;
    const timeStr = d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    const dateStr = d.toLocaleDateString([], { month: 'short', day: 'numeric' });
    if (diff < 86400000 && now.getDate() === d.getDate()) return `Today, ${timeStr}`;
    if (diff < 172800000) return `Yesterday, ${timeStr}`;
    return `${dateStr}, ${timeStr}`;
  } catch { return ts; }
}

// ── Init ─────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  // On mobile — sidebar starts hidden
  if (isMobile()) {
    sidebar.classList.remove('collapsed');  // mobile uses mobile-open, not collapsed
  }

  // Load history from DB
  loadHistory();
});
