#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Face detection + image-to-ray mapping using OpenCV Haar Cascade
@author: manish
"""

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

# 1. CAMERA INTRINSICS (replace with calibrated values)
CAMERA_INTRINSICS = {
    'fx': 600.0,
    'fy': 600.0,
    'cx': 320.0,
    'cy': 240.0,
}

# Webcam properties
IMAGE_WIDTH, IMAGE_HEIGHT = 640, 480
IMAGE_CENTER = np.array([CAMERA_INTRINSICS['cx'], CAMERA_INTRINSICS['cy']])

# --------------------------------------------------
# Geometry
# --------------------------------------------------
def image_to_camera_ray(pixel, intrinsics):
    """Pixel (u,v) -> unit ray in camera frame"""
    u, v = pixel
    fx, fy = intrinsics['fx'], intrinsics['fy']
    cx, cy = intrinsics['cx'], intrinsics['cy']

    x = (u - cx) / fx
    y = (v - cy) / fy
    z = 1.0

    ray = np.array([x, y, z])
    return ray / np.linalg.norm(ray)

def webcam_to_local_ray(ray_cam, body_attitude=[0, 0, 0]):
    """Camera -> body/local frame"""
    roll, pitch, yaw = np.radians(body_attitude)
    R_body = R.from_euler('xyz', [roll, pitch, yaw]).as_matrix()
    ray_local = R_body @ ray_cam
    return ray_local / np.linalg.norm(ray_local)

# --------------------------------------------------
# Haar Cascade Face Detector
# --------------------------------------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# --------------------------------------------------
# Main loop
# --------------------------------------------------
cap = cv2.VideoCapture(0)  # <-- your Logitech cam
cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMAGE_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMAGE_HEIGHT)

print("Press 'q' to quit. Face ray tracking running...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    if len(faces) > 0:
        # Pick the largest face (more stable)
        x, y, w, h = max(faces, key=lambda b: b[2] * b[3])

        centroid = np.array([
            x + w // 2,
            y + h // 2
        ])

        # Compute rays
        ray_cam = image_to_camera_ray(centroid, CAMERA_INTRINSICS)
        ray_local = webcam_to_local_ray(ray_cam)

        # Draw
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(frame, centroid.astype(int), 6, (0, 255, 0), -1)
        cv2.circle(frame, IMAGE_CENTER.astype(int), 4, (0, 0, 255), -1)

        # Debug prints
        print(f"Centroid: {centroid}")
        print(f"ray_cam:   {ray_cam}")
        print(f"ray_local: {ray_local}")
        print("-" * 40)

        # Overlay text
        cv2.putText(
            frame,
            f"ray_cam: [{ray_cam[0]:.2f}, {ray_cam[1]:.2f}, {ray_cam[2]:.2f}]",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

    cv2.imshow("Face Ray Tracking (OpenCV only)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
