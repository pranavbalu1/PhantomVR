###Hand and Face Tracking with Gesture Recognition and VR Control
This repository implements real-time hand and face tracking using OpenCV and MediaPipe, along with gesture recognition for controlling virtual environments and applications such as VR control systems. The project integrates both hand gestures and head pose estimation to interact with virtual devices such as a virtual joystick or VR controllers.

Features
Hand Tracking:

Detects hand gestures using MediaPipe's Hands solution.
Gesture recognition includes:
Trigger press (index & middle finger touch thumb)
Primary action (thumb-index circle / OK gesture)
Secondary action (thumb-middle finger touch)
Button A/B (thumb to pinky or ring finger)
Gesture events are customizable via callback functions.
Face Tracking:

Estimates head pose (pitch, yaw, and roll) from the 3D facial landmarks using MediaPipe's FaceMesh solution.
Head pose data can be used for controlling virtual environments.
VR Control Integration:

Uses a virtual joystick via the pyvjoy library to send hand orientation data (yaw, pitch, and roll) as joystick axis values.
Hand gestures are translated into virtual button presses for VR interactions.
Real-Time Processing:

Real-time processing of webcam video feed with live feedback of tracked hand gestures and face pose.
