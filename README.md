# 3D-Face-Replacement-with-MediaPipe

# MediaPipe Face Transform

A real-time face transformation system built using **Python, OpenCV, and MediaPipe Face Mesh**.

This project detects facial landmarks in real time and overlays a captured face onto the detected face region.

---

## Features

- Real-time face detection
- 468 facial landmark detection using MediaPipe
- Face region transformation
- Webcam-based face capture
- Real-time visualization with OpenCV

---

## Technologies Used

- Python
- OpenCV
- MediaPipe
- NumPy

---

## How It Works

1. The webcam captures a frame.
2. MediaPipe Face Mesh detects facial landmarks.
3. The bounding box of the face is calculated.
4. A replacement face image is resized to match the detected face region.
5. The replacement face is blended with the detected face using a mask.

---
