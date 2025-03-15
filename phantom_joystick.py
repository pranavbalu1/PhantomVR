import cv2
import mediapipe as mp
import pyvjoy
import numpy as np

def calculate_hand_orientation(landmarks):
    # Extract key points (wrist, index finger base, pinky base)
    wrist = np.array([landmarks[0].x, landmarks[0].y, landmarks[0].z])
    index_base = np.array([landmarks[5].x, landmarks[5].y, landmarks[5].z])
    pinky_base = np.array([landmarks[17].x, landmarks[17].y, landmarks[17].z])
    
    # Calculate vectors
    hand_vector = index_base - pinky_base
    up_vector = np.cross(hand_vector, wrist - index_base)
    
    # Normalize vectors
    hand_vector = hand_vector / np.linalg.norm(hand_vector)
    up_vector = up_vector / np.linalg.norm(up_vector)
    
    # Compute angles
    yaw = np.arctan2(hand_vector[0], hand_vector[2])  # Left-Right
    pitch = np.arctan2(up_vector[1], up_vector[2])     # Up-Down
    roll = np.arctan2(hand_vector[1], hand_vector[0])  # Rotation
    
    return np.degrees(pitch), np.degrees(roll), np.degrees(yaw)

def main():
    cap = cv2.VideoCapture(0)
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
    j = pyvjoy.VJoyDevice(1)  # Initialize virtual joystick
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)  # Mirror effect
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                pitch, roll, yaw = calculate_hand_orientation(hand_landmarks.landmark)
                
                # Normalize and map angles to joystick range (0-32767)
                x_axis = int((yaw + 90) / 180 * 32767)
                y_axis = int((pitch + 90) / 180 * 32767)
                z_rotation = int((roll + 90) / 180 * 32767)
                
                # Send to virtual joystick
                j.set_axis(pyvjoy.HID_USAGE_X, x_axis)
                j.set_axis(pyvjoy.HID_USAGE_Y, y_axis)
                j.set_axis(pyvjoy.HID_USAGE_RZ, z_rotation)
        
        cv2.imshow("Hand Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
