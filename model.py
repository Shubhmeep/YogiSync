# model_prediction.py

import mediapipe as mp
import cv2
import numpy as np
import time
import pickle
import warnings

warnings.filterwarnings('ignore')

# Load the trained ML model from the pickle file.
with open('yoga_v1.pkl', 'rb') as f:
    model = pickle.load(f)

# Initialize MediaPipe drawing and pose modules.
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Precompute CLAHE and gamma correction table.
CLAHE = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
GAMMA = 1.2
invGamma = 1.0 / GAMMA
gamma_table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(256)]).astype("uint8")

def preprocess_frame(frame):
    """
    Enhance the frame using CLAHE and gamma correction.
    """
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    cl = CLAHE.apply(l)
    lab = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    final = cv2.LUT(enhanced, gamma_table)
    return final

def draw_skeleton(frame, landmarks, connections, color, thickness, alpha=0.6):
    """
    Draw a skeleton on the frame.
    """
    overlay = frame.copy()
    body_connections = [
        (11, 12),  # Shoulders
        (11, 13), (13, 15),  # Left arm
        (12, 14), (14, 16),  # Right arm
        (11, 23), (12, 24),  # Torso
        (23, 24),           # Hips
        (23, 25), (25, 27),  # Left leg
        (24, 26), (26, 28),  # Right leg
    ]
    for connection in body_connections:
        start_idx, end_idx = connection
        start_point = (int(landmarks[start_idx].x * frame.shape[1]), int(landmarks[start_idx].y * frame.shape[0]))
        end_point = (int(landmarks[end_idx].x * frame.shape[1]), int(landmarks[end_idx].y * frame.shape[0]))
        cv2.line(overlay, start_point, end_point, color, thickness)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

def extract_landmarks(results):
    """
    Extract pose landmarks and return them as a flattened array.
    """
    row = []
    if results.pose_landmarks:
        for landmark in results.pose_landmarks.landmark:
            row.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
    else:
        row.extend([0] * (33 * 4))
    return row

# Processing resolution for efficiency.
PROCESS_WIDTH, PROCESS_HEIGHT = 640, 480

cap = cv2.VideoCapture(0)
prev_time = 0

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_resized = cv2.resize(frame, (PROCESS_WIDTH, PROCESS_HEIGHT))
        display_frame = frame_resized.copy()
        proc_frame = preprocess_frame(frame_resized)
        proc_frame_rgb = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2RGB)
        proc_frame_rgb.flags.writeable = False

        results = pose.process(proc_frame_rgb)

        # Default display values.
        skeleton_color = (0, 0, 255)  # Red if no pose detected
        pose_text = "No Pose Detected"
        accuracy_text = ""

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            # Draw skeleton.
            draw_skeleton(display_frame, landmarks, mp_pose.POSE_CONNECTIONS, skeleton_color, 8, alpha=0.3)

            # Extract landmarks for ML model.
            landmarks_ml = extract_landmarks(results)
            features = np.array(landmarks_ml).reshape(1, -1)

            if np.count_nonzero(features) == 0:
                pred_class = None
                max_prob = 0.0
            else:
                probabilities = model.predict_proba(features)
                max_prob = np.max(probabilities)
                if max_prob >= 0.7:
                    pred_class = model.classes_[np.argmax(probabilities)]
                else:
                    pred_class = None

            if pred_class is not None:
                pose_text = f"{pred_class} Detected (ML)"
                accuracy_text = f"Accuracy: {max_prob * 100:.1f}%"
            else:
                pose_text = "Pose not confidently predicted"

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time else 0
        prev_time = curr_time

        cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(display_frame, pose_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2, cv2.LINE_AA)
        if accuracy_text:
            cv2.putText(display_frame, accuracy_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("ML Pose Prediction", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
