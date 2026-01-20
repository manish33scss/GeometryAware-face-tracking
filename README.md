# Face Ray Tracking (OpenCV + Docker)

This project captures live video from a USB camera, detects faces, and prints the camera ray direction.

## Requirements
- Ubuntu Linux
- Docker installed
- USB camera (usually /dev/video0)
- X11 display (for OpenCV window)

## Files
- main.py : main application
- Dockerfile : Docker build setup
- requirements.txt : Python dependencies

## Build

docker build -t face-ray-app .

## Allow docker to use display
xhost +local:docker


## Run the container 
docker run --rm --device=/dev/video0 -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix face-ray-app
