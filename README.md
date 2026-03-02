# Live Blob Tracker (Web Version)

This is a web port of the Python blob tracking script. It uses **OpenCV.js** for real-time optical flow and feature detection, and the **Web Audio API** for sound-triggered effects.

## Features

- **Real-time Optical Flow**: Tracks points using Lucas-Kanade algorithm.
- **Audio Sensitivity**: Spawns tracking boxes based on microphone amplitude.
- **Inversion Effect**: Visual boxes that invert colors of the live feed.
- **Modern UI**: A sleek, dark-themed dashboard with glassmorphism controls.

## How to Run

1. Install dependencies:
   ```bash
   npm install
   ```
2. Start the development server:
   ```bash
   npm run dev
   ```
3. Open the provided `localhost` URL in your browser.
4. Grant Camera and Microphone permissions.
5. Click **START TRACKING**.

## Technical Details

- **Frontend**: Vanilla JS with Vite.
- **Computer Vision**: OpenCV.js (loaded via CDN).
- **Styling**: Vanilla CSS with modern typography (Google Fonts).

Inspired by the original `Blob-Track-Lite`[https://github.com/Code-X-Sakthi/Blob-Track-Lite].
