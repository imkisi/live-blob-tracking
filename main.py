import cv2
import numpy as np
import random
import pyaudio
import struct
from pathlib import Path

# --- Configuration ---
CHUNK = 1024             # Audio buffer size
FORMAT = pyaudio.paInt16 # Audio format
CHANNELS = 1             # Mono
RATE = 44100             # Sample rate
THRESHOLD = 2000         # Audio trigger sensitivity (adjust based on mic)

class TrackedPoint:
    def __init__(self, pos: tuple[float, float], life: int, size: int):
        self.pos = np.array(pos, dtype=np.float32)
        self.life = life
        self.size = size

def _sample_size_bell(min_s: int, max_s: int) -> int:
    return int(np.random.randint(min_s, max_s))

def run_live_tracked_effect():
    # 1. Initialize Camera
    cap = cv2.VideoCapture(0) # 0 is usually the default webcam
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # 2. Initialize Audio for Live Triggering
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, 
                    input=True, frames_per_buffer=CHUNK)

    # 3. Setup Tracking Logic
    orb = cv2.ORB_create(nfeatures=500)
    active: list[TrackedPoint] = []
    prev_gray = None
    
    # Visual Params
    life_frames = 15
    min_size, max_size = 15, 40
    neighbor_links = 3

    print("Live Tracking Started! Press 'q' to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1) # Mirror for natural feel
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape

            # --- AUDIO TRIGGER LOGIC ---
            data = stream.read(CHUNK, exception_on_overflow=False)
            shorts = struct.unpack("%dh" % (len(data) / 2), data)
            amplitude = max(shorts)
            
            # If volume hits threshold, spawn new points
            if amplitude > THRESHOLD:
                kps = orb.detect(gray, None)
                for kp in random.sample(kps, min(len(kps), 10)):
                    active.append(TrackedPoint(kp.pt, life_frames, _sample_size_bell(min_size, max_size)))

            # --- OPTICAL FLOW (Tracking) ---
            if prev_gray is not None and active:
                prev_pts = np.array([p.pos for p in active], dtype=np.float32).reshape(-1, 1, 2)
                next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None)
                
                new_active = []
                for tp, new_pt, ok in zip(active, next_pts.reshape(-1, 2), status.reshape(-1)):
                    if ok and tp.life > 0:
                        tp.pos = new_pt
                        tp.life -= 1
                        new_active.append(tp)
                active = new_active

            # --- RENDERING ---
            coords = [tp.pos for tp in active]
            # Draw Links
            # Changed 'p' to 'pt' here to avoid overwriting the PyAudio object 'p'
            for i, pt in enumerate(coords):
                # Simple optimization: only link to nearby points in the list
                for j in range(max(0, i-neighbor_links), min(len(coords), i+neighbor_links)):
                    if i != j:
                        # Use pt instead of p
                        cv2.line(frame, tuple(pt.astype(int)), tuple(coords[j].astype(int)), (255, 200, 200), 1)

            # Draw Inversion Boxes
            for tp in active:
                x, y = tp.pos
                s = tp.size
                tl = (max(0, int(x - s // 2)), max(0, int(y - s // 2)))
                br = (min(w - 1, int(x + s // 2)), min(h - 1, int(y + s // 2)))
                
                roi = frame[tl[1]:br[1], tl[0]:br[0]]
                if roi.size:
                    frame[tl[1]:br[1], tl[0]:br[0]] = 255 - roi
                cv2.rectangle(frame, tl, br, (255, 200, 200), 1)

            cv2.imshow('Live Blob Tracking Effect', frame)
            prev_gray = gray

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Cleanup
        cap.release()
        stream.stop_stream()
        stream.close()
        p.terminate()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run_live_tracked_effect()