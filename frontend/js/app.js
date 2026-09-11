const uploadZone     = document.getElementById('upload-zone');
const fileInput      = document.getElementById('file-input');
const fileInfo       = document.getElementById('file-info');
const fileName       = document.getElementById('file-name');
const fileMeta       = document.getElementById('file-meta');
const removeFileBtn  = document.getElementById('remove-file');

const predictBtn     = document.getElementById('predict-btn');
const liveBtn        = document.getElementById('live-btn');

const resultsSection = document.getElementById('results-section');
const snrBadge       = document.getElementById('snr-badge');
const snrConfidence  = document.getElementById('snr-confidence');
const snrProbsContainer = document.getElementById('snr-probs');

const modResultCard  = document.getElementById('mod-result-card');
const modClass       = document.getElementById('mod-class');
const modConfidence  = document.getElementById('mod-confidence');
const modConfBar     = document.getElementById('mod-conf-bar-fill');
const modNotAvail    = document.getElementById('mod-not-available');
const modNotAvailMsg = document.getElementById('mod-not-avail-msg');

const waveformSection = document.getElementById('waveform-section');

const modalBackdrop  = document.getElementById('modal-backdrop');
const modalCloseBtn  = document.getElementById('modal-close');

const toast          = document.getElementById('toast');

let selectedFile = null;

let waveformChart = null;

uploadZone.addEventListener('click', () => fileInput.click());

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFileSelection(e.target.files[0]);
    }
});

uploadZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadZone.classList.add('drag-over');
});

uploadZone.addEventListener('dragleave', () => {
    uploadZone.classList.remove('drag-over');
});

uploadZone.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadZone.classList.remove('drag-over');
    if (e.dataTransfer.files.length > 0) {
        handleFileSelection(e.dataTransfer.files[0]);
    }
});

removeFileBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    clearFile();
});

function handleFileSelection(file) {
    if (!file.name.toLowerCase().endsWith('.npy')) {
        showToast('Invalid file type. Please upload a .npy file.', 'error');
        return;
    }

    selectedFile = file;

    fileName.textContent = file.name;
    fileMeta.textContent = `${formatBytes(file.size)} · .npy`;
    fileInfo.classList.add('visible');

    predictBtn.disabled = false;

    hideResults();
}

function clearFile() {
    selectedFile = null;
    fileInput.value = '';
    fileInfo.classList.remove('visible');
    predictBtn.disabled = true;
    hideResults();
}

function hideResults() {
    resultsSection.classList.remove('visible');
    waveformSection.classList.remove('visible');
}

predictBtn.addEventListener('click', runPrediction);

async function runPrediction() {
    if (!selectedFile) {
        showToast('Please upload a .npy signal file first.', 'error');
        return;
    }

    predictBtn.classList.add('loading');
    predictBtn.disabled = true;
    hideResults();

    try {
        const formData = new FormData();
        formData.append('file', selectedFile);

        const response = await fetch('/predict', {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.detail || `Server error (${response.status})`);
        }

        const result = await response.json();
        renderResults(result);
    } catch (err) {
        showToast(err.message || 'Prediction failed. Please try again.', 'error');
    } finally {
        predictBtn.classList.remove('loading');
        predictBtn.disabled = false;
    }
}

function renderResults(result) {
    const category = result.snr_category;
    snrBadge.textContent = category;
    snrBadge.className = `snr-badge snr-badge--${category.toLowerCase()}`;
    snrConfidence.textContent = `Confidence: ${(result.snr_confidence * 100).toFixed(1)}%`;

    renderSnrProbabilities(result.snr_probabilities);

    if (result.modulation_available) {
        modResultCard.style.display = 'block';
        modNotAvail.style.display = 'none';

        modClass.textContent = result.modulation_class;
        const confPercent = (result.modulation_confidence * 100).toFixed(1);
        modConfidence.textContent = `Confidence: ${confPercent}%`;

        requestAnimationFrame(() => {
            modConfBar.style.width = `${confPercent}%`;
        });
    } else {
        modResultCard.style.display = 'none';
        modNotAvail.style.display = 'block';
        modNotAvailMsg.textContent = result.modulation_message;
    }

    resultsSection.classList.add('visible');

    if (result.waveform) {
        renderWaveform(result.waveform);
        waveformSection.classList.add('visible');
    }
}

function renderSnrProbabilities(probs) {
    if (!probs) return;

    const order = ['HIGH', 'MEDIUM', 'LOW'];
    snrProbsContainer.innerHTML = '';

    for (const label of order) {
        const value = probs[label] || 0;
        const percent = (value * 100).toFixed(1);

        const bar = document.createElement('div');
        bar.className = 'snr-prob-bar';
        bar.innerHTML = `
            <span class="snr-prob-bar__label">${label}</span>
            <div class="snr-prob-bar__track">
                <div class="snr-prob-bar__fill snr-prob-bar__fill--${label.toLowerCase()}"
                     style="width: 0%"></div>
            </div>
            <span class="snr-prob-bar__value">${percent}%</span>
        `;
        snrProbsContainer.appendChild(bar);

        requestAnimationFrame(() => {
            bar.querySelector('.snr-prob-bar__fill').style.width = `${percent}%`;
        });
    }
}

let activeWaveformView = 'iq';

function renderWaveform(waveform) {
    window.__waveformData = waveform;
    activeWaveformView = 'iq';

    document.querySelectorAll('.waveform-tab').forEach(tab => {
        tab.classList.toggle('active', tab.dataset.view === 'iq');
    });

    drawChart(waveform, 'iq');
}

function drawChart(waveform, view) {
    const ctx = document.getElementById('waveform-canvas').getContext('2d');
    const labels = Array.from({ length: waveform.i_component.length }, (_, i) => i);

    if (waveformChart) {
        waveformChart.destroy();
    }

    let datasets = [];

    if (view === 'iq') {
        datasets = [
            {
                label: 'I (In-phase)',
                data: waveform.i_component,
                borderColor: '#00d4ff',
                backgroundColor: 'rgba(0, 212, 255, 0.12)',
                borderWidth: 2,
                pointRadius: 0,
                fill: true,
                tension: 0.25,
            },
            {
                label: 'Q (Quadrature)',
                data: waveform.q_component,
                borderColor: '#ff00ff',
                backgroundColor: 'rgba(255, 0, 255, 0.12)',
                borderWidth: 2,
                pointRadius: 0,
                fill: true,
                tension: 0.25,
            },
        ];
    } else if (view === 'magnitude') {
        const magnitude = waveform.i_component.map((iVal, idx) => {
            const qVal = waveform.q_component[idx];
            return Math.sqrt(iVal * iVal + qVal * qVal);
        });
        datasets = [
            {
                label: 'Magnitude',
                data: magnitude,
                borderColor: '#00ff88',
                backgroundColor: 'rgba(0, 255, 136, 0.14)',
                borderWidth: 2,
                pointRadius: 0,
                fill: true,
                tension: 0.25,
            },
        ];
    } else if (view === 'phase') {
        const phase = waveform.i_component.map((iVal, idx) => {
            const qVal = waveform.q_component[idx];
            return Math.atan2(qVal, iVal) * (180 / Math.PI);
        });
        datasets = [
            {
                label: 'Phase (deg)',
                data: phase,
                borderColor: '#ffaa00',
                backgroundColor: 'rgba(255, 170, 0, 0.12)',
                borderWidth: 2,
                pointRadius: 0,
                fill: true,
                tension: 0.1,
            },
        ];
    }

    waveformChart = new Chart(ctx, {
        type: 'line',
        data: { labels, datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            plugins: {
                legend: {
                    labels: {
                        color: '#00ff88',
                        font: { family: "'Share Tech Mono', monospace", size: 12, weight: 'bold' },
                        usePointStyle: true,
                        pointStyle: 'rect',
                    },
                },
                tooltip: {
                    backgroundColor: 'rgba(18, 18, 26, 0.95)',
                    titleColor: '#00ff88',
                    bodyColor: '#e0e0e0',
                    borderColor: '#00ff88',
                    borderWidth: 1,
                    padding: 10,
                    titleFont: { family: "'Orbitron', monospace", size: 11 },
                    bodyFont: { family: "'JetBrains Mono', monospace", size: 11 },
                },
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: '> SAMPLE INDEX',
                        color: '#00ff88',
                        font: { family: "'Share Tech Mono', monospace", size: 11, weight: 'bold' },
                    },
                    ticks: {
                        color: '#6b7280',
                        font: { family: "'JetBrains Mono', monospace", size: 10 },
                        maxTicksLimit: 12,
                    },
                    grid: { color: 'rgba(0, 255, 136, 0.08)' },
                },
                y: {
                    title: {
                        display: true,
                        text: view === 'phase' ? '> DEGREES' : '> AMPLITUDE',
                        color: '#00ff88',
                        font: { family: "'Share Tech Mono', monospace", size: 11, weight: 'bold' },
                    },
                    ticks: {
                        color: '#6b7280',
                        font: { family: "'JetBrains Mono', monospace", size: 10 },
                    },
                    grid: { color: 'rgba(0, 255, 136, 0.08)' },
                },
            },
        },
    });
}

document.querySelectorAll('.waveform-tab').forEach(tab => {
    tab.addEventListener('click', () => {
        const view = tab.dataset.view;
        if (!window.__waveformData || view === activeWaveformView) return;

        activeWaveformView = view;
        document.querySelectorAll('.waveform-tab').forEach(t =>
            t.classList.toggle('active', t === tab)
        );
        drawChart(window.__waveformData, view);
    });
});

liveBtn.addEventListener('click', () => {
    modalBackdrop.classList.add('visible');
});

modalCloseBtn.addEventListener('click', () => {
    modalBackdrop.classList.remove('visible');
});

modalBackdrop.addEventListener('click', (e) => {
    if (e.target === modalBackdrop) {
        modalBackdrop.classList.remove('visible');
    }
});

let toastTimeout = null;

function showToast(message, type = 'error') {
    toast.textContent = message;
    toast.className = `toast toast--${type} visible`;

    clearTimeout(toastTimeout);
    toastTimeout = setTimeout(() => {
        toast.classList.remove('visible');
    }, 5000);
}

function formatBytes(bytes) {
    if (bytes === 0) return '0 B';
    const units = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(1024));
    return `${(bytes / Math.pow(1024, i)).toFixed(1)} ${units[i]}`;
}
