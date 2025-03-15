# Hand and Face Tracking with Gesture Recognition and VR Control

This repository implements real-time hand and face tracking using OpenCV and MediaPipe, along with gesture recognition for controlling virtual environments and applications, such as VR control systems. The project integrates both hand gestures and head pose estimation to interact with virtual devices, such as a virtual joystick or VR controllers.

## Features

### Hand Tracking
- **Real-time hand tracking** using MediaPipe's `Hands` solution.
- **Gesture recognition** includes:
  - **Trigger press**: Index and middle finger touching the thumb.
  - **Primary action**: Thumb-index circle (OK gesture).
  - **Secondary action**: Thumb-middle finger touch.
  - **Button A/B**: Thumb to pinky or ring finger.
- **Customizable gestures**: Gesture events can be mapped to specific actions using callback functions.

### Face Tracking
- **Head pose estimation**: Determines pitch, yaw, and roll from 3D facial landmarks using MediaPipe's `FaceMesh` solution.
- **Head pose data**: Can be used for controlling virtual environments, allowing head movements to simulate actions in VR.

### VR Control Integration
- **Virtual joystick**: Integrates with the `pyvjoy` library to send hand orientation data (yaw, pitch, and roll) as joystick axis values.
- **Hand gesture control**: Translates hand gestures into virtual button presses for VR interactions, allowing users to control the virtual environment using natural hand movements.

### Real-Time Processing
- **Live webcam feed**: Processes the webcam video stream in real-time for dynamic hand gesture tracking and face pose estimation.
- **Immediate feedback**: Displays tracked hand gestures and head poses live for interaction with the virtual environment.

### Prerequisites
To run this project, you need to have the following libraries installed:
- OpenCV
- MediaPipe
- pyvjoy (for virtual joystick integration)

