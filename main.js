import './style.css'

// --- CONSTANTS ---
const CONFIG = {
    audioThreshold: 2000,
    lifeFrames: 15,
    minSize: 15,
    maxSize: 40,
    neighborLinks: 3,
    maxSpawnPerTrigger: 10,
    fps: 30,
}

// --- STATE ---
let state = {
    isRunning: false,
    activePoints: [],
    prevGray: null,
    video: null,
    canvas: null,
    ctx: null,
    audioContext: null,
    analyser: null,
    dataArray: null,
    cvReady: false,
    width: 640,
    height: 480,
    loopId: null
}

class TrackedPoint {
    constructor(x, y, life, size) {
        this.pos = { x, y };
        this.life = life;
        this.size = size;
    }
}

// --- UI ELEMENTS ---
const elements = {
    threshold: document.getElementById('threshold'),
    life: document.getElementById('life'),
    boxSize: document.getElementById('boxSize'),
    maxSpawn: document.getElementById('maxSpawn'),
    startButton: document.getElementById('startButton'),
    status: document.getElementById('cv-status'),
    blobCount: document.getElementById('blobCount'),
    ampBar: document.getElementById('ampBar'),
    loadingOverlay: document.getElementById('loadingOverlay'),
}

// Update UI value displays
const updateDisplays = () => {
    document.getElementById('thresholdVal').innerText = elements.threshold.value;
    document.getElementById('lifeVal').innerText = elements.life.value;
    document.getElementById('sizeVal').innerText = elements.boxSize.value;
    document.getElementById('spawnVal').innerText = elements.maxSpawn.value;
}

['threshold', 'life', 'boxSize', 'maxSpawn'].forEach(id => {
    elements[id].addEventListener('input', updateDisplays);
});

// --- OPENCV STATUS ---
window.onOpenCvReady = () => {
    state.cvReady = true;
    elements.status.innerText = "CV READY";
    elements.status.style.borderColor = "var(--success)";
    elements.status.style.color = "var(--success)";
    console.log('OpenCV.js is ready.');
};

// Check CV load
if (typeof cv !== 'undefined' && cv.onRuntimeInitialized) {
    onOpenCvReady();
} else {
    let checkCv = setInterval(() => {
        if (typeof cv !== 'undefined' && cv.Mat) {
            window.onOpenCvReady();
            clearInterval(checkCv);
        }
    }, 100);
}

// --- INITIALIZATION ---
async function initMedia() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ 
            video: { width: state.width, height: state.height }, 
            audio: true 
        });
        
        state.video = document.getElementById('videoElement');
        state.video.srcObject = stream;
        await state.video.play();

        // Audio Setup
        state.audioContext = new (window.AudioContext || window.webkitAudioContext)();
        const source = state.audioContext.createMediaStreamSource(stream);
        state.analyser = state.audioContext.createAnalyser();
        state.analyser.fftSize = 256;
        source.connect(state.analyser);
        state.dataArray = new Uint8Array(state.analyser.frequencyBinCount);

        // Canvas Setup
        state.canvas = document.getElementById('canvasOutput');
        state.ctx = state.canvas.getContext('2d', { willReadFrequently: true });
        state.canvas.width = state.video.videoWidth;
        state.canvas.height = state.video.videoHeight;
        state.width = state.canvas.width;
        state.height = state.canvas.height;

        elements.loadingOverlay.style.opacity = '0';
        setTimeout(() => elements.loadingOverlay.style.display = 'none', 500);
        
        return true;
    } catch (err) {
        console.error("Media error:", err);
        alert("Please allow camera and microphone access.");
        return false;
    }
}

// --- CORE LOGIC ---
function getAmplitude() {
    if (!state.analyser) return 0;
    state.analyser.getByteTimeDomainData(state.dataArray);
    let max = 0;
    for (let i = 0; i < state.dataArray.length; i++) {
        const val = Math.abs(state.dataArray[i] - 128);
        if (val > max) max = val;
    }
    // Scale to match the Python THRESHOLD roughly (0-128 map to 0-32768)
    return (max / 128) * 10000;
}

let orb = null;
let noneMat = null;

function processFrame() {
    if (!state.isRunning || !state.cvReady) return;

    // 1. Capture current frame
    // flip horizontally
    state.ctx.save();
    state.ctx.scale(-1, 1);
    state.ctx.drawImage(state.video, -state.width, 0, state.width, state.height);
    state.ctx.restore();

    let frame = cv.imread(state.canvas);
    let gray = new cv.Mat();
    cv.cvtColor(frame, gray, cv.COLOR_RGBA2GRAY);

    // 2. Audio Trigger & Spawn
    const amp = getAmplitude();
    elements.ampBar.style.width = Math.min(100, (amp / 10000) * 100) + '%';
    
    if (amp > elements.threshold.value) {
        if (!orb) orb = new cv.ORB();
        if (!noneMat) noneMat = new cv.Mat();
        
        let keypoints = new cv.KeyPointVector();
        orb.detect(gray, keypoints, noneMat);
        
        const count = Math.min(keypoints.size(), parseInt(elements.maxSpawn.value));
        const indices = Array.from({length: keypoints.size()}, (_, i) => i);
        // Shuffle for random sample
        for (let i = indices.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [indices[i], indices[j]] = [indices[j], indices[i]];
        }

        for (let i = 0; i < count; i++) {
            const kp = keypoints.get(indices[i]);
            const size = Math.floor(Math.random() * (parseInt(elements.boxSize.value) + 10)) + 15;
            state.activePoints.push(new TrackedPoint(kp.pt.x, kp.pt.y, parseInt(elements.life.value), size));
        }
        keypoints.delete();
    }

    // 3. Optical Flow Tracking
    if (state.prevGray && state.activePoints.length > 0) {
        let prevPts = new cv.Mat(state.activePoints.length, 1, cv.CV_32FC2);
        for (let i = 0; i < state.activePoints.length; i++) {
            prevPts.data32F[i * 2] = state.activePoints[i].pos.x;
            prevPts.data32F[i * 2 + 1] = state.activePoints[i].pos.y;
        }

        let nextPts = new cv.Mat();
        let status = new cv.Mat();
        let err = new cv.Mat();
        
        try {
            cv.calcOpticalFlowPyrLK(state.prevGray, gray, prevPts, nextPts, status, err);

            let newActive = [];
            for (let i = 0; i < state.activePoints.length; i++) {
                const ok = status.data[i];
                const pt = state.activePoints[i];
                if (ok && pt.life > 0) {
                    pt.pos.x = nextPts.data32F[i * 2];
                    pt.pos.y = nextPts.data32F[i * 2 + 1];
                    pt.life--;
                    newActive.push(pt);
                }
            }
            state.activePoints = newActive;
        } catch (e) {
            console.error("OF error", e);
        }

        prevPts.delete(); nextPts.delete(); status.delete(); err.delete();
    }

    // 4. Rendering to Canvas
    // Clear and redraw background
    state.ctx.save();
    state.ctx.scale(-1, 1);
    state.ctx.drawImage(state.video, -state.width, 0, state.width, state.height);
    state.ctx.restore();

    // Draw Links
    state.ctx.strokeStyle = 'rgba(255, 200, 200, 0.5)';
    state.ctx.lineWidth = 1;
    const links = parseInt(CONFIG.neighborLinks);
    for (let i = 0; i < state.activePoints.length; i++) {
        const p1 = state.activePoints[i];
        for (let j = Math.max(0, i - links); j < Math.min(state.activePoints.length, i + links); j++) {
            if (i === j) continue;
            const p2 = state.activePoints[j];
            state.ctx.beginPath();
            state.ctx.moveTo(p1.pos.x, p1.pos.y);
            state.ctx.lineTo(p2.pos.x, p2.pos.y);
            state.ctx.stroke();
        }
    }

    // Draw Inversion Boxes
    state.activePoints.forEach(p => {
        const s = p.size;
        const x = p.pos.x - s / 2;
        const y = p.pos.y - s / 2;
        
        // Negative effect (Invert)
        // Canvas globalCompositeOperation 'difference' with white fill creates inversion
        state.ctx.save();
        state.ctx.globalCompositeOperation = 'difference';
        state.ctx.fillStyle = 'white';
        state.ctx.fillRect(x, y, s, s);
        state.ctx.restore();

        // Border
        state.ctx.strokeStyle = '#ffc8c8';
        state.ctx.strokeRect(x, y, s, s);
    });

    elements.blobCount.innerText = state.activePoints.length;

    // Cleanup Mats
    if (state.prevGray) state.prevGray.delete();
    state.prevGray = gray;
    frame.delete();

    state.loopId = requestAnimationFrame(processFrame);
}

// --- EVENTS ---
elements.startButton.addEventListener('click', async () => {
    if (state.isRunning) {
        state.isRunning = false;
        cancelAnimationFrame(state.loopId);
        elements.startButton.innerText = "START TRACKING";
        elements.startButton.classList.remove('stop');
        return;
    }

    if (!state.audioContext) {
        const success = await initMedia();
        if (!success) return;
    }

    state.isRunning = true;
    elements.startButton.innerText = "STOP TRACKING";
    elements.startButton.classList.add('stop');
    processFrame();
});
