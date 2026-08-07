import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time

# 1. SETUP HAND & POSE DETECTORS (MediaPipe Tasks API)
hand_model_path = 'hand_landmarker.task'
pose_model_path = 'pose_landmarker.task'

# Hand Landmarker Options
hand_options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=hand_model_path),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
hand_detector = vision.HandLandmarker.create_from_options(hand_options)

# Pose Landmarker Options (Shoulder, Elbow, Wrist)
pose_options = vision.PoseLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=pose_model_path),
    running_mode=vision.RunningMode.VIDEO,
    min_pose_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
pose_detector = vision.PoseLandmarker.create_from_options(pose_options)

def calculate_angle(a, b, c):
    """Calculates 2D angle between three points (a -> b -> c) in degrees."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360.0 - angle
    return angle

cap = cv2.VideoCapture(0)
print("Starting Dual Body & Hand Posture HUD...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_timestamp_ms = int(time.time() * 1000)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # Run Inferences
    hand_result = hand_detector.detect_for_video(mp_image, frame_timestamp_ms)
    pose_result = pose_detector.detect_for_video(mp_image, frame_timestamp_ms)

    annotated_frame = frame.copy()
    h, w, _ = annotated_frame.shape

    arm_angle = 0
    posture_status = "Detecting Pose..."
    status_color = (200, 200, 200)

    # --- A. BODY POSE (Shoulder & Elbow Joint Movements) ---
    if pose_result.pose_landmarks:
        pose_lms = pose_result.pose_landmarks[0]
        
        # Right Arm Indices: 12 = R_Shoulder, 14 = R_Elbow, 16 = R_Wrist
        r_shoulder = (int(pose_lms[12].x * w), int(pose_lms[12].y * h))
        r_elbow = (int(pose_lms[14].x * w), int(pose_lms[14].y * h))
        r_wrist_pose = (int(pose_lms[16].x * w), int(pose_lms[16].y * h))

        # Draw Arm Joint Connectors and Dots
        cv2.line(annotated_frame, r_shoulder, r_elbow, (255, 200, 0), 3)
        cv2.line(annotated_frame, r_elbow, r_wrist_pose, (255, 200, 0), 3)
        
        cv2.circle(annotated_frame, r_shoulder, 8, (255, 120, 0), -1)  # Shoulder Dot
        cv2.circle(annotated_frame, r_elbow, 8, (255, 120, 0), -1)     # Elbow Dot

        # Angle Calculation (Shoulder -> Elbow -> Wrist)
        arm_angle = calculate_angle(r_shoulder, r_elbow, r_wrist_pose)

        # Posture Rules
        if r_elbow[1] > r_shoulder[1] + 120:
            posture_status = "Warning: Dropped Elbow"
            status_color = (0, 0, 255)
        elif arm_angle < 50:
            posture_status = "Frog Position (Bent)"
            status_color = (255, 200, 0)
        elif 50 <= arm_angle <= 130:
            posture_status = "Optimal Mid-Stroke"
            status_color = (0, 255, 0)
        else:
            posture_status = "Tip Position (Extended)"
            status_color = (0, 180, 255)

    # --- B. HAND & FINGER DIAGRAM POINTS ---
    if hand_result.hand_landmarks:
        hand_lms = hand_result.hand_landmarks[0]

        # 1. Wrist Point (Filled Blue - Diagram Base)
        wrist_pt = (int(hand_lms[0].x * w), int(hand_lms[0].y * h))
        cv2.circle(annotated_frame, wrist_pt, 12, (255, 120, 0), -1)

        # 2. Thumb Joint (Unfilled Circle)
        thumb_pt = (int(hand_lms[2].x * w), int(hand_lms[2].y * h))
        cv2.circle(annotated_frame, thumb_pt, 10, (255, 160, 50), 2)

        # 3. Middle Knuckle (Unfilled Circle)
        middle_pt = (int(hand_lms[9].x * w), int(hand_lms[9].y * h))
        cv2.circle(annotated_frame, middle_pt, 10, (255, 160, 50), 2)

        # 4. Index Knuckle (Unfilled Circle)
        index_pt = (int(hand_lms[5].x * w), int(hand_lms[5].y * h))
        cv2.circle(annotated_frame, index_pt, 10, (255, 160, 50), 2)

        # 5. Finger Unit Box
        finger_tips = [hand_lms[8], hand_lms[12], hand_lms[16], hand_lms[20]]
        finger_pts = [(int(lm.x * w), int(lm.y * h)) for lm in finger_tips]
        
        x_c = [p[0] for p in finger_pts]
        y_c = [p[1] for p in finger_pts]
        box_tl = (min(x_c) - 12, min(y_c) - 12)
        box_br = (max(x_c) + 12, max(y_c) + 12)

        cv2.rectangle(annotated_frame, box_tl, box_br, (255, 180, 80), 2, cv2.LINE_AA)
        cv2.putText(annotated_frame, "Finger Unit", (box_tl[0], box_tl[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 180, 80), 1)

    # --- C. UI METRICS DASHBOARD (Dark Panel Below Feed) ---
    panel_height = 100
    dashboard_panel = np.zeros((panel_height, w, 3), dtype=np.uint8)
    dashboard_panel[:] = (20, 20, 20)  # Dark Charcoal

    # Column 1: Joint Angle Readout
    cv2.putText(dashboard_panel, "ELBOW JOINT ANGLE", (30, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)
    cv2.putText(dashboard_panel, f"{int(arm_angle)} deg", (30, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)

    # Column 2: Posture Evaluation Status
    cv2.putText(dashboard_panel, "POSTURE EVALUATION", (int(w * 0.45), 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)
    cv2.putText(dashboard_panel, posture_status, (int(w * 0.45), 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, status_color, 2)

    combined_display = np.vstack((annotated_frame, dashboard_panel))
    cv2.imshow("Violin Posture HUD", combined_display)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

hand_detector.close()
pose_detector.close()
cap.release()
cv2.destroyAllWindows()