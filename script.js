import {
    HandLandmarker,
    PoseLandmarker,
    FilesetResolver
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest";

const video = document.getElementById("webcam");
const canvasElement = document.getElementById("output_canvas");
const canvasCtx = canvasElement.getContext("2d");

const angleDisplay = document.getElementById("angle-display");
const statusDisplay = document.getElementById("status-display");

let handLandmarker;
let poseLandmarker;
let lastVideoTime = -1;

// 1. Initialize MediaPipe Tasks for Browser
async function initializeDetectors() {
    statusDisplay.innerText = "Loading AI Models...";
    
    const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
    );

    handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
            delegate: "GPU"
        },
        runningMode: "VIDEO",
        numHands: 1
    });

    poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task",
            delegate: "GPU"
        },
        runningMode: "VIDEO"
    });

    startWebcam();
}

// 2. Request Webcam Stream
function startWebcam() {
    navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720 } })
        .then((stream) => {
            video.srcObject = stream;
            video.addEventListener("loadeddata", renderLoop);
            statusDisplay.innerText = "Detecting Pose...";
            statusDisplay.className = "metric-value status-neutral";
        })
        .catch((err) => {
            console.error("Camera access denied: ", err);
            statusDisplay.innerText = "Camera Permission Denied";
            statusDisplay.className = "metric-value status-alert";
        });
}

// Vector Angle Math (Shoulder -> Elbow -> Wrist)
function calculateAngle(a, b, c) {
    const radians = Math.atan2(c.y - b.y, c.x - b.x) - Math.atan2(a.y - b.y, a.x - b.x);
    let angle = Math.abs((radians * 180.0) / Math.PI);
    if (angle > 180.0) {
        angle = 360.0 - angle;
    }
    return Math.round(angle);
}

// 3. Frame Render & Detection Loop
async function renderLoop() {
    canvasElement.width = video.videoWidth;
    canvasElement.height = video.videoHeight;

    let startTimeMs = performance.now();
    if (lastVideoTime !== video.currentTime) {
        lastVideoTime = video.currentTime;

        canvasCtx.save();
        canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

        const w = canvasElement.width;
        const h = canvasElement.height;

        // Run Inferences
        const poseResult = poseLandmarker.detectForVideo(video, startTimeMs);
        const handResult = handLandmarker.detectForVideo(video, startTimeMs);

        // --- A. BODY POSE EVALUATION ---
        if (poseResult.landmarks && poseResult.landmarks.length > 0) {
            const poseLms = poseResult.landmarks[0];

            // Right Arm Landmarks: 12 = Shoulder, 14 = Elbow, 16 = Wrist
            const rShoulder = { x: poseLms[12].x * w, y: poseLms[12].y * h };
            const rElbow = { x: poseLms[14].x * w, y: poseLms[14].y * h };
            const rWrist = { x: poseLms[16].x * w, y: poseLms[16].y * h };

            // Draw Arm Connectors
            canvasCtx.strokeStyle = "#ffc800";
            canvasCtx.lineWidth = 4;
            canvasCtx.beginPath();
            canvasCtx.moveTo(rShoulder.x, rShoulder.y);
            canvasCtx.lineTo(rElbow.x, rElbow.y);
            canvasCtx.lineTo(rWrist.x, rWrist.y);
            canvasCtx.stroke();

            // Draw Shoulder & Elbow Points
            drawCircle(canvasCtx, rShoulder, 8, "#ff7800", true);
            drawCircle(canvasCtx, rElbow, 8, "#ff7800", true);

            // Compute Angle
            const angle = calculateAngle(rShoulder, rElbow, rWrist);
            angleDisplay.innerText = `${angle} deg`;

            // Posture Evaluation Rules
            if (rElbow.y > rShoulder.y + 120) {
                statusDisplay.innerText = "Warning: Dropped Elbow";
                statusDisplay.className = "metric-value status-alert";
            } else if (angle < 50) {
                statusDisplay.innerText = "Frog Position (Bent)";
                statusDisplay.className = "metric-value status-warn";
            } else if (angle >= 50 && angle <= 130) {
                statusDisplay.innerText = "Optimal Mid-Stroke";
                statusDisplay.className = "metric-value status-good";
            } else {
                statusDisplay.innerText = "Tip Position (Extended)";
                statusDisplay.className = "metric-value status-info";
            }
        }

        // --- B. HAND & FINGER DIAGRAM OVERLAYS ---
        if (handResult.landmarks && handResult.landmarks.length > 0) {
            const handLms = handResult.landmarks[0];

            const wristPt = { x: handLms[0].x * w, y: handLms[0].y * h };
            const thumbPt = { x: handLms[2].x * w, y: handLms[2].y * h };
            const middleKnuckle = { x: handLms[9].x * w, y: handLms[9].y * h };
            const indexKnuckle = { x: handLms[5].x * w, y: handLms[5].y * h };

            // 1. Wrist Point (Filled Blue)
            drawCircle(canvasCtx, wristPt, 12, "#ff7800", true);

            // 2. Thumb & Knuckle Points (Unfilled Blue)
            drawCircle(canvasCtx, thumbPt, 10, "#ffa032", false);
            drawCircle(canvasCtx, middleKnuckle, 10, "#ffa032", false);
            drawCircle(canvasCtx, indexKnuckle, 10, "#ffa032", false);

            // 3. Finger Unit Bounding Box
            const fingerTips = [
                handLms[8],  // Index Tip
                handLms[12], // Middle Tip
                handLms[16], // Ring Tip
                handLms[20]  // Pinky Tip
            ];

            const xCoords = fingerTips.map(p => p.x * w);
            const yCoords = fingerTips.map(p => p.y * h);

            const minX = Math.min(...xCoords) - 15;
            const maxX = Math.max(...xCoords) + 15;
            const minY = Math.min(...yCoords) - 15;
            const maxY = Math.max(...yCoords) + 15;

            // Draw Finger Unit Box
            canvasCtx.strokeStyle = "#ffb450";
            canvasCtx.lineWidth = 2;
            canvasCtx.setLineDash([6, 4]); // Dotted border
            canvasCtx.strokeRect(minX, minY, maxX - minX, maxY - minY);
            canvasCtx.setLineDash([]); // Reset dash

            canvasCtx.fillStyle = "#ffb450";
            canvasCtx.font = "12px sans-serif";
            canvasCtx.fillText("Finger Unit", minX, minY - 6);
        }

        canvasCtx.restore();
    }

    requestAnimationFrame(renderLoop);
}

function drawCircle(ctx, pt, radius, color, filled) {
    ctx.beginPath();
    ctx.arc(pt.x, pt.y, radius, 0, 2 * Math.PI);
    if (filled) {
        ctx.fillStyle = color;
        ctx.fill();
    } else {
        ctx.strokeStyle = color;
        ctx.lineWidth = 2.5;
        ctx.stroke();
    }
}

initializeDetectors();