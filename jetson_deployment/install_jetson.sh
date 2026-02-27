#!/bin/bash
echo "Installing dependencies for Jetson Nano Face Recognition..."
sudo apt-get update
sudo apt-get install -y libopenblas-base libopenmpi-dev libomp-dev
pip3 install onnxruntime-gpu insightface opencv-python numpy
echo "Setup Complete. Run 'python3 main_attendance.py' to start."