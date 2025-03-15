import cv2
import mediapipe as mp
import math
import time

class HandTracker:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5)
        
        self.gesture_callbacks = []
        self.last_gesture_time = {}
        self.cooldown = 0.5  # Seconds between gesture detections

    def register_gesture_callback(self, callback):
        self.gesture_callbacks.append(callback)

    def _trigger_gesture(self, hand_label, gesture_name):
        current_time = time.time()
        key = f"{hand_label}_{gesture_name}"
        
        if current_time - self.last_gesture_time.get(key, 0) > self.cooldown:
            for callback in self.gesture_callbacks:
                callback(hand_label, gesture_name)
            self.last_gesture_time[key] = current_time

    def _get_distance(self, point1, point2):
        return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

    def process_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                hand = results.multi_handedness[hand_idx].classification[0].label
                self._detect_gestures(hand, hand_landmarks.landmark)
        
        return frame

    def _detect_gestures(self, hand_label, landmarks):
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        ring_tip = landmarks[16]
        pinky_tip = landmarks[20]
        palm_base = landmarks[0]

        # Double Finger Trigger Press (Index & Middle Finger Touch Thumb)
        if self._get_distance(thumb_tip, index_tip) < 0.05 and self._get_distance(thumb_tip, middle_tip) < 0.05:
            self._trigger_gesture(hand_label, "trigger_press")
            return 

        # Primary Action (Thumb-Index Circle / OK Gesture)
        if self._get_distance(thumb_tip, index_tip) < 0.05 and self._get_distance(middle_tip, palm_base) > 0.1:
            self._trigger_gesture(hand_label, "primary_action")
            return

        # Secondary Action (Thumb-Middle Finger Touch)
        if self._get_distance(thumb_tip, middle_tip) < 0.05:
            self._trigger_gesture(hand_label, "secondary_action")
            return


        # Button A/B (Thumb to Pinky or Ring Finger)
        if self._get_distance(thumb_tip, pinky_tip) < 0.05:
            self._trigger_gesture(hand_label, "button_a")
            return
                
        if self._get_distance(thumb_tip, ring_tip) < 0.05:
            self._trigger_gesture(hand_label, "button_b")
            return

if __name__ == "__main__":
    tracker = HandTracker()
    
    # Example callback for gestures
    def handle_gesture(hand, gesture):
        print(f"Gesture detected: {hand} hand - {gesture}")
    
    tracker.register_gesture_callback(handle_gesture)

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = tracker.process_frame(frame)
        cv2.imshow('Hand Tracking', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()