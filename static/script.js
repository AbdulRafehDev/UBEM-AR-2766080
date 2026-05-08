// ════════════════════════════════════════════════════════════
//  STATE
// ════════════════════════════════════════════════════════════
const state = {
    buildingMax: 3,
    selectedFiles: [],       // File objects chosen by user
    croppedDataURLs: [],     // base64 strings after cropping, one per image
    cropIndex: 0,            // which image we are currently cropping
    cropMode: null,          // 'upload' | 'streetview'
    streetViewImageURL: null, // URL for address-only street view crop
    pendingAddress: '',
};

// ════════════════════════════════════════════════════════════
//  ELEMENT REFS
// ════════════════════════════════════════════════════════════
const addressInput      = document.getElementById('addressInput');
const runAddressBtn     = document.getElementById('runAddressBtn');
const openUploadBtn     = document.getElementById('openUploadModalBtn');
const uploadOverlay     = document.getElementById('uploadOverlay');
const closeUploadModal  = document.getElementById('closeUploadModal');
const cancelUploadBtn   = document.getElementById('cancelUploadBtn');
const buildingCards     = document.querySelectorAll('.building-type-card');
const maxFileBadge      = document.getElementById('maxFileBadge');
const dropZone          = document.getElementById('dropZone');
const hiddenFileInput   = document.getElementById('hiddenFileInput');
const selectedFilesInfo = document.getElementById('selectedFilesInfo');
const startAnalysisBtn  = document.getElementById('startAnalysisBtn');

const cropOverlay       = document.getElementById('cropOverlay');
const closeCropModal    = document.getElementById('closeCropModal');
const cropCanvas        = document.getElementById('cropCanvas');
const cropCanvasWrap    = document.getElementById('cropCanvasWrap');
const cropCounter       = document.getElementById('cropCounter');
const cropCoordsLabel   = document.getElementById('cropCoordsLabel');
const resetCropBtn      = document.getElementById('resetCropBtn');
const confirmCropBtn    = document.getElementById('confirmCropBtn');

const spinnerOverlay    = document.getElementById('spinnerOverlay');
const spinnerLabel      = document.getElementById('spinnerLabel');
const errorMsg          = document.getElementById('errorMsg');
const jsGenResults      = document.getElementById('jsGenResults');
const jsInferenceResults= document.getElementById('jsInferenceResults');

// ════════════════════════════════════════════════════════════
//  HELPERS
// ════════════════════════════════════════════════════════════
function showToast(msg, duration = 3500) {
    const toast = document.getElementById('toastMsg');
    toast.textContent = msg;
    toast.classList.add('show');
    setTimeout(() => toast.classList.remove('show'), duration);
}

function showSpinner(label = 'Running analysis…') {
    spinnerLabel.textContent = label;
    spinnerOverlay.classList.add('active');
}
function hideSpinner() { spinnerOverlay.classList.remove('active'); }
function showError(msg) { errorMsg.textContent = msg; errorMsg.style.display = msg ? 'block' : 'none'; }
function hideError() { showError(''); }

// ════════════════════════════════════════════════════════════
//  BUILDING TYPE SELECTION
// ════════════════════════════════════════════════════════════
buildingCards.forEach(card => {
    card.addEventListener('click', () => {
        buildingCards.forEach(c => c.classList.remove('selected'));
        card.classList.add('selected');
        state.buildingMax = parseInt(card.dataset.max);
        maxFileBadge.textContent = `Max ${state.buildingMax} file${state.buildingMax > 1 ? 's' : ''}`;
        hiddenFileInput.setAttribute('multiple', state.buildingMax > 1 ? 'true' : '');
        // Re-validate file list if already chosen
        if (state.selectedFiles.length > state.buildingMax) {
            state.selectedFiles = state.selectedFiles.slice(0, state.buildingMax);
            updateFilesInfo();
        }
    });
});

// ════════════════════════════════════════════════════════════
//  UPLOAD MODAL
// ════════════════════════════════════════════════════════════
openUploadBtn.addEventListener('click', () => {
    state.selectedFiles = [];
    updateFilesInfo();
    uploadOverlay.classList.add('active');
});

function closeUpload() { uploadOverlay.classList.remove('active'); }
closeUploadModal.addEventListener('click', closeUpload);
cancelUploadBtn.addEventListener('click', closeUpload);
uploadOverlay.addEventListener('click', e => { if (e.target === uploadOverlay) closeUpload(); });

dropZone.addEventListener('click', () => hiddenFileInput.click());

dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', e => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    handleFiles(Array.from(e.dataTransfer.files));
});

hiddenFileInput.addEventListener('change', () => {
    handleFiles(Array.from(hiddenFileInput.files));
    hiddenFileInput.value = '';
});

function handleFiles(files) {
    const imageFiles = files.filter(f => f.type.startsWith('image/'));
    const combined = [...state.selectedFiles, ...imageFiles];

    if (combined.length > state.buildingMax) {
        // Hard reject — clear selection and show toast
        state.selectedFiles = [];
        updateFilesInfo();
        const typeLabel = document.querySelector('.building-type-card.selected .name').textContent.trim();
        showToast(`Too many images — ${typeLabel} allows max ${state.buildingMax}. Please try again.`);
        return;
    }

    hideError();
    state.selectedFiles = combined;
    updateFilesInfo();
}

function updateFilesInfo() {
    const count = state.selectedFiles.length;
    startAnalysisBtn.disabled = count === 0;
    if (count === 0) {
        selectedFilesInfo.style.display = 'none';
    } else {
        selectedFilesInfo.style.display = 'block';
        selectedFilesInfo.textContent = `${count} file${count > 1 ? 's' : ''} selected: ${state.selectedFiles.map(f => f.name).join(', ')}`;
    }
}

startAnalysisBtn.addEventListener('click', () => {
    if (!state.selectedFiles.length) return;
    closeUpload();
    // Begin crop flow for uploaded images
    state.cropMode = 'upload';
    state.croppedDataURLs = [];
    state.cropIndex = 0;
    beginCropSession();
});

// ════════════════════════════════════════════════════════════
//  ADDRESS-ONLY → FETCH STREET VIEW → CROP → PIPELINE
// ════════════════════════════════════════════════════════════
runAddressBtn.addEventListener('click', async () => {
    const address = addressInput.value.trim();
    if (!address) { showError('Please enter an address.'); return; }
    hideError();
    showSpinner('Fetching Street View image…');

    try {
        const resp = await fetch('/fetch-street-view', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ address }),
        });
        const data = await resp.json();
        if (!resp.ok || data.error) throw new Error(data.error || 'Failed to fetch street view');

        hideSpinner();
        state.pendingAddress = address;
        state.streetViewImageURL = data.image_url;
        state.cropMode = 'streetview';
        state.croppedDataURLs = [];
        state.cropIndex = 0;
        state.selectedFiles = [{ _svURL: data.image_url }]; // pseudo-file
        beginCropSession();
    } catch (err) {
        hideSpinner();
        showError('Street View error: ' + err.message);
    }
});

// ════════════════════════════════════════════════════════════
//  CROP SESSION
// ════════════════════════════════════════════════════════════
let cropImg = null;
let cropRect = null;   // { x, y, w, h } in canvas-space
let isDragging = false;
let dragStart = null;

function beginCropSession() {
    loadCropImage(state.cropIndex);
}

function loadCropImage(index) {
    const total = state.selectedFiles.length;
    cropCounter.textContent = `Image ${index + 1} of ${total}`;
    confirmCropBtn.textContent = index < total - 1 ? 'Confirm & Next →' : 'Confirm & Run ✓';
    cropRect = null;
    cropCoordsLabel.textContent = '';

    cropImg = new Image();
    cropImg.crossOrigin = 'anonymous';

    cropImg.onload = () => {
        // Scale image to fit modal width (max 660px)
        const maxW = Math.min(660, window.innerWidth - 80);
        const scale = Math.min(1, maxW / cropImg.naturalWidth);
        const dispW = Math.round(cropImg.naturalWidth * scale);
        const dispH = Math.round(cropImg.naturalHeight * scale);

        cropCanvas.width  = dispW;
        cropCanvas.height = dispH;
        drawCropCanvas();
        cropOverlay.classList.add('active');
    };

    const file = state.selectedFiles[index];
    if (file._svURL) {
        // Street view pseudo-file — load from URL
        cropImg.src = file._svURL;
    } else {
        cropImg.src = URL.createObjectURL(file);
    }
}

function drawCropCanvas() {
    const ctx = cropCanvas.getContext('2d');
    ctx.clearRect(0, 0, cropCanvas.width, cropCanvas.height);
    ctx.drawImage(cropImg, 0, 0, cropCanvas.width, cropCanvas.height);

    if (!cropRect) return;

    const { x, y, w, h } = cropRect;
    // Darken outside
    ctx.fillStyle = 'rgba(0,0,0,0.45)';
    ctx.fillRect(0, 0, cropCanvas.width, cropCanvas.height);
    ctx.clearRect(x, y, w, h);
    ctx.drawImage(cropImg, x, y, w, h, x, y, w, h);

    // Selection border
    ctx.strokeStyle = '#0e7490';
    ctx.lineWidth = 2;
    ctx.setLineDash([6, 3]);
    ctx.strokeRect(x, y, w, h);
    ctx.setLineDash([]);

    // Corner handles
    const hs = 8;
    ctx.fillStyle = '#0e7490';
    [[x,y],[x+w,y],[x,y+h],[x+w,y+h]].forEach(([cx,cy]) => {
        ctx.fillRect(cx - hs/2, cy - hs/2, hs, hs);
    });
}

function canvasPos(e) {
    const rect = cropCanvas.getBoundingClientRect();
    const scaleX = cropCanvas.width  / rect.width;
    const scaleY = cropCanvas.height / rect.height;
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    const clientY = e.touches ? e.touches[0].clientY : e.clientY;
    return {
        x: (clientX - rect.left) * scaleX,
        y: (clientY - rect.top)  * scaleY,
    };
}

cropCanvas.addEventListener('mousedown',  startDrag);
cropCanvas.addEventListener('touchstart', startDrag, { passive: true });
cropCanvas.addEventListener('mousemove',  onDrag);
cropCanvas.addEventListener('touchmove',  onDrag, { passive: true });
document.addEventListener('mouseup',  endDrag);
document.addEventListener('touchend', endDrag);

function startDrag(e) {
    isDragging = true;
    dragStart = canvasPos(e);
    cropRect = null;
}

function onDrag(e) {
    if (!isDragging) return;
    const pos = canvasPos(e);
    const x = Math.min(dragStart.x, pos.x);
    const y = Math.min(dragStart.y, pos.y);
    const w = Math.abs(pos.x - dragStart.x);
    const h = Math.abs(pos.y - dragStart.y);
    cropRect = { x, y, w, h };
    drawCropCanvas();
    cropCoordsLabel.textContent = `${Math.round(w)} × ${Math.round(h)} px`;
}

function endDrag() { isDragging = false; }

resetCropBtn.addEventListener('click', () => {
    cropRect = null;
    drawCropCanvas();
    cropCoordsLabel.textContent = '';
});

closeCropModal.addEventListener('click', () => {
    cropOverlay.classList.remove('active');
});

confirmCropBtn.addEventListener('click', () => {
    // Export cropped region (or full image if no rect drawn)
    const offscreen = document.createElement('canvas');
    let sx, sy, sw, sh;

    if (cropRect && cropRect.w > 10 && cropRect.h > 10) {
        // Scale back to natural image size
        const scaleX = cropImg.naturalWidth  / cropCanvas.width;
        const scaleY = cropImg.naturalHeight / cropCanvas.height;
        sx = cropRect.x * scaleX;
        sy = cropRect.y * scaleY;
        sw = cropRect.w * scaleX;
        sh = cropRect.h * scaleY;
    } else {
        sx = 0; sy = 0;
        sw = cropImg.naturalWidth;
        sh = cropImg.naturalHeight;
    }

    offscreen.width  = sw;
    offscreen.height = sh;
    offscreen.getContext('2d').drawImage(cropImg, sx, sy, sw, sh, 0, 0, sw, sh);
    const dataURL = offscreen.toDataURL('image/jpeg', 0.92);
    state.croppedDataURLs.push(dataURL);

    state.cropIndex++;
    if (state.cropIndex < state.selectedFiles.length) {
        // More images to crop
        loadCropImage(state.cropIndex);
    } else {
        // All done — close crop modal and run analysis
        cropOverlay.classList.remove('active');
        runAnalysis();
    }
});

// ════════════════════════════════════════════════════════════
//  RUN ANALYSIS (AJAX → render results)
// ════════════════════════════════════════════════════════════
async function runAnalysis() {
    const address = addressInput.value.trim();
    const hasAddress = !!address;
    const buildingType = document.querySelector('.building-type-card.selected')?.dataset.type || 'detached';

    // Clear previous results
    jsGenResults.innerHTML = '';
    jsInferenceResults.innerHTML = '';

    showSpinner('Running AI analysis…');
    hideError();

    try {
        if (hasAddress) {
            const formData = new FormData();
            formData.append('address', address);
            formData.append('building_type', buildingType);
            state.croppedDataURLs.forEach(d => formData.append('cropped_images', d));

            const resp = await fetch('/run-pipeline', { method: 'POST', body: formData });
            const data = await resp.json();
            if (!resp.ok || data.error) throw new Error(data.error || 'Pipeline failed');
            hideSpinner();
            renderGenResults(data);
        } else {
            submitCroppedForm(buildingType);
        }
    } catch (err) {
        hideSpinner();
        showToast('Analysis error: ' + err.message);
    }
}

function submitCroppedForm(buildingType) {
    const form = document.createElement('form');
    form.method = 'POST';
    form.action = '/';
    form.enctype = 'multipart/form-data';

    const addrInput = document.createElement('input');
    addrInput.type = 'hidden';
    addrInput.name = 'address';
    addrInput.value = addressInput.value.trim();
    form.appendChild(addrInput);

    const btInput = document.createElement('input');
    btInput.type = 'hidden';
    btInput.name = 'building_type';
    btInput.value = buildingType || 'detached';
    form.appendChild(btInput);

    state.croppedDataURLs.forEach(dataURL => {
        const inp = document.createElement('input');
        inp.type = 'hidden';
        inp.name = 'cropped_images';
        inp.value = dataURL;
        form.appendChild(inp);
    });

    document.body.appendChild(form);
    showSpinner('Processing images…');
    form.submit();
}

// ════════════════════════════════════════════════════════════
//  RENDER PIPELINE RESULTS (JS, no page reload)
// ════════════════════════════════════════════════════════════
function renderGenResults(data) {
    const eAI      = data.energy_ai   || {};
    const eBase    = data.energy_baseline || {};
    const delta    = (eAI.total_kwh || 0) - (eBase.total_kwh || 0);
    const deltaPct = eBase.total_kwh ? (delta / eBase.total_kwh) * 100 : 0;
    const isNeg    = delta > 0;

    const deltaHTML = delta < 0
        ? `↓ ${Math.abs(delta).toFixed(1)} kWh (${Math.abs(deltaPct).toFixed(2)}% less)`
        : delta > 0
        ? `↑ ${delta.toFixed(1)} kWh (${deltaPct.toFixed(2)}% more)`
        : 'No difference';
    const deltaDesc = delta < 0
        ? 'The AI-predicted WWR results in lower simulated energy use than the 20% assumption.'
        : delta > 0
        ? 'The AI-predicted WWR results in higher simulated energy use than the 20% assumption.'
        : 'Both models predict identical energy consumption.';

    // Build all image pair rows
    const pairs = data.image_pairs && data.image_pairs.length > 0
        ? data.image_pairs
        : (data.image_url ? [{ image_url: data.image_url, annotated_url: data.annotated_url }] : []);

    const multipleImages = pairs.length > 1;
    const pairsHTML = pairs.map((pair, i) => `
        <div class="gen-images" style="margin-bottom:${i < pairs.length - 1 ? '16px' : '20px'}">
            <figure>
                <img src="/static/${pair.image_url}" alt="Original ${i+1}">
                <figcaption>Original${multipleImages ? ` (Image ${i+1})` : ''}</figcaption>
            </figure>
            <figure>
                <img src="/static/${pair.annotated_url}" alt="Annotated ${i+1}">
                <figcaption>AI Detections${multipleImages ? ` (Image ${i+1})` : ''}</figcaption>
            </figure>
        </div>
    `).join('');

    jsGenResults.innerHTML = `
    <div class="gen-section">
        <div class="gen-header"><h2>Energy Analysis — ${escHtml(data.address)}</h2></div>
        <div class="gen-body">
            ${pairsHTML}
            ${data.class_counts ? `<p class="detections">Detections: ${JSON.stringify(data.class_counts)}</p>` : ''}
            <div class="wwr-row">
                <div class="wwr-badge ai">
                    <span class="label">AI-Predicted WWR</span>
                    <span class="value">${data.wwr_ai}%</span>
                    <span class="sublabel">From facade analysis${multipleImages ? ' (averaged)' : ''}</span>
                </div>
                <div class="wwr-badge baseline">
                    <span class="label">Baseline WWR</span>
                    <span class="value">${data.wwr_baseline}%</span>
                    <span class="sublabel">Industry assumption</span>
                </div>
            </div>
            <div class="energy-section">
                <h3>Annual Energy Consumption (EnergyPlus Simulation)</h3>
                <div class="energy-grid">
                    <div class="energy-card ai">
                        <div class="card-title">AI WWR Model</div>
                        <div class="energy-row"><span class="ekey">Heating</span><span class="eval">${(eAI.heating_kwh||0).toFixed(1)} kWh</span></div>
                        <div class="energy-row"><span class="ekey">Cooling</span><span class="eval">${(eAI.cooling_kwh||0).toFixed(1)} kWh</span></div>
                        <div class="energy-row"><span class="ekey">Total</span><span class="eval">${(eAI.total_kwh||0).toFixed(1)} kWh</span></div>
                    </div>
                    <div class="energy-card baseline">
                        <div class="card-title">Baseline 20% Model</div>
                        <div class="energy-row"><span class="ekey">Heating</span><span class="eval">${(eBase.heating_kwh||0).toFixed(1)} kWh</span></div>
                        <div class="energy-row"><span class="ekey">Cooling</span><span class="eval">${(eBase.cooling_kwh||0).toFixed(1)} kWh</span></div>
                        <div class="energy-row"><span class="ekey">Total</span><span class="eval">${(eBase.total_kwh||0).toFixed(1)} kWh</span></div>
                    </div>
                </div>
                <div class="delta-box ${isNeg ? 'negative' : ''}">
                    <div class="delta-label">AI model vs 20% baseline — total energy difference</div>
                    <div class="delta-value">${deltaHTML}</div>
                    <div class="delta-desc">${deltaDesc}</div>
                </div>
            </div>
        </div>
    </div>`;

    jsGenResults.scrollIntoView({ behavior: 'smooth' });
}

function escHtml(str) {
    return String(str).replace(/[&<>"']/g, m => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[m]));
}
