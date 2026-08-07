# Violin Posture & Bow Hold Tutor

A real-time computer vision application that monitors violin bowing posture and bow hand finger placements using MediaPipe and OpenCV.

## Launch Options

### Option 1: Live Web App (No Installation)
Try the web version directly in your browser:
**[Launch Web App](https://YOUR_GITHUB_USERNAME.github.io/violin-posture-tutor/)**

---

### Option 2: Local Python Desktop App

#### Prerequisites
- Python 3.10+
- Built-in or external webcam

#### Setup & Run
1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR_GITHUB_USERNAME/violin-posture-tutor.git](https://github.com/YOUR_GITHUB_USERNAME/violin-posture-tutor.git)
   cd violin-posture-tutor
   
**Automated Setup & Launch (macOS/Linux):**
run in your terminal:
 - chmod +x setup.sh
 - ./setup.sh

**Manual Launch:**
 - python3 -m venv venv
 - source venv/bin/activate
 - pip install opencv-python numpy mediapipe
 - python app.py
   (Note: Vision task models download automatically on first launch).

#### Features

**Arm Joint Angle Tracking:** Real-time calculation of the elbow angle (Shoulder > Elbow > Wrist) to identify bowing positions (Frog, Mid-Stroke, Tip) and flag dropped elbows.
**Bow Hand & Finger Alignment:** Keypoint tracking mapped to essential bow hold anchor points: 
 - Wrist: Main anchor point.
 - Thumb Base & Knuckles: Joint position indicators for bow hand flexibility.
 - Finger Unit Box: Grouped bounding box tracking index, middle, ring, and pinky tip alignment on the bow stick.

 - Non-Obstructive Telemetry HUD: Dark-mode UI panel appended below the camera feed displaying live telemetry and posture guidance without obscuring the video feed.

*(Remember to replace `YOUR_GITHUB_USERNAME` with your actual GitHub username).*

---

### 2. Commit & Push to GitHub

Push all your new web files (`index.html`, `style.css`, `script.js`), `setup.sh`, and updated `README.md`:

```bash
git add index.html style.css script.js setup.sh app.py README.md .gitignore
git commit -m "Feat: Add web app interface for GitHub Pages deployment and retain Python desktop app"
git push origin main
