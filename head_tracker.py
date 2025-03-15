import cv2
import mediapipe as mp
import numpy as np
import math

# Initialize MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# 3D model points (approximate coordinates based on average face)
model_points = np.array([
    (0.0, 0.0, 0.0),             # Nose tip (4)
    (0.0, -330.0, -65.0),        # Chin (152)
    (-225.0, 170.0, -135.0),     # Left eye left corner (33)
    (225.0, 170.0, -135.0),      # Right eye right corner (263)
    (-150.0, -150.0, -125.0),    # Left mouth corner (61)
    (150.0, -150.0, -125.0)      # Right mouth corner (291)
], dtype="double")

# Corresponding 2D landmark indices from MediaPipe FaceMesh
landmark_indices = [4, 152, 33, 263, 61, 291]

def rotation_matrix_to_euler_angles(r_matrix):
    """Convert rotation matrix to Euler angles (pitch, yaw, roll)"""
    sy = math.sqrt(r_matrix[0,0] * r_matrix[0,0] +  r_matrix[1,0] * r_matrix[1,0])
    singular = sy < 1e-6

    if not singular:
        x = math.atan2(r_matrix[2,1], r_matrix[2,2])  # Pitch
        y = math.atan2(-r_matrix[2,0], sy)            # Yaw
        z = math.atan2(r_matrix[1,0], r_matrix[0,0])  # Roll
    else:
        x = math.atan2(-r_matrix[1,2], r_matrix[1,1])
        y = math.atan2(-r_matrix[2,0], sy)
        z = 0

    return np.array([x, y, z]) * 180 / math.pi

# Initialize webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    # Convert to RGB and process with FaceMesh
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    h, w, _ = image.shape
    
    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]

        # Extract 2D image points
        image_points = []
        for idx in landmark_indices:
            lm = face_landmarks.landmark[idx]
            image_points.append([lm.x * w, lm.y * h])
        
        image_points = np.array(image_points, dtype="double")

        # Camera matrix approximation
        focal_length = w
        center = (w/2, h/2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype="double")

        # Solve PnP to get rotation vector
        success, rotation_vector, _ = cv2.solvePnP(
            model_points,
            image_points,
            camera_matrix,
            np.zeros((4, 1)),  # No distortion
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if success:
            # Convert rotation vector to matrix
            r_matrix, _ = cv2.Rodrigues(rotation_vector)
            
            # Get Euler angles
            euler_angles = rotation_matrix_to_euler_angles(r_matrix)
            
            # Display angles
            cv2.putText(image, f"Pitch: {euler_angles[0]:.2f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(image, f"Yaw: {euler_angles[1]:.2f}", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(image, f"Roll: {euler_angles[2]:.2f}", (10, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Display output
    cv2.imshow('Head Pose Estimation', image)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()