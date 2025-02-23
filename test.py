import mediapipe as mp
import cv2
import numpy as np

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def compute_angle(landmarks, i, j, k):
    """Calculate angle between three points"""
    a = np.array([landmarks[i].x, landmarks[i].y])
    b = np.array([landmarks[j].x, landmarks[j].y])
    c = np.array([landmarks[k].x, landmarks[k].y])
    
    ba = a - b
    bc = c - b
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
    
    return round(angle, 1)

# Initialize webcam
cap = cv2.VideoCapture(0)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        # Make detection
        results = pose.process(image)
        
        # Convert back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # Calculate plank angles (same for triangle pose)
            left_elbow = compute_angle(landmarks, 11, 13, 15)  # Left elbow angle
            right_elbow = compute_angle(landmarks, 12, 14, 16)  # Right elbow angle
            
            left_shoulder = compute_angle(landmarks, 13, 11, 23)  # Left shoulder-hip angle
            right_shoulder = compute_angle(landmarks, 14, 12, 24)  # Right shoulder-hip angle
            
            left_hip = compute_angle(landmarks, 11, 23, 25)  # Left hip angle
            right_hip = compute_angle(landmarks, 12, 24, 26)  # Right hip angle
            
            left_knee = compute_angle(landmarks, 23, 25, 27)  # Left knee angle
            right_knee = compute_angle(landmarks, 24, 26, 28)  # Right knee angle

            #Calculate triangle pose angles
            elbows_ok= left_elbow >= 160 and right_elbow >= 160
            shoulders_ok = (70 <= left_shoulder <= 100 and 130 <= right_shoulder <= 150) or (70 <= right_shoulder <= 100 and 130 <= left_shoulder <= 150)
            hips_ok = (130 <= left_hip <= 160 and 70 <= right_hip <= 90) or (130 <= right_hip <= 160 and 70 <= left_hip <= 90)
            knees_ok = left_knee >= 160 and right_knee >= 160
            
            triangle_pose = elbows_ok and shoulders_ok and hips_ok and knees_ok
            
            # Display angles on frame
            angle_text = [
                f"L Elbow: {left_elbow}° (Target: >=160°)",
                f"R Elbow: {right_elbow}° (Target: >=160°)",
                f"L Shoulder: {left_shoulder}° (Target: 50-100°)",
                f"R Shoulder: {right_shoulder}° (Target: 50-100°)",
                f"L Hip: {left_hip}° (Target: 165-185°)",
                f"R Hip: {right_hip}° (Target: 165-185°)",
                f"L Knee: {left_knee}° (Target: 165-185°)",
                f"R Knee: {right_knee}° (Target: 165-185°)"
            ]
            
            # Draw skeleton
            mp_drawing.draw_landmarks(
                image, 
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )
            
            # Display angles
            y_pos = 30
            for text in angle_text:
                cv2.putText(image, text, (10, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                y_pos += 30
                
        cv2.imshow('Plank Pose Angle Detection', image)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()