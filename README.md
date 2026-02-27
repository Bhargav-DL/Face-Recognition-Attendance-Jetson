# AI-Powered Face Recognition & Attendance System

An end-to-end Computer Vision pipeline that transitions from Cloud-based Deep Learning to Edge-AI Deployment (NVIDIA Jetson Nano). This system utilizes state-of-the-art InsightFace (ArcFace) to provide high-accuracy biometric identification.

## Overview
This repository features a robust face recognition system capable of identifying individuals in real-time. By utilizing 512-dimensional feature embeddings, the system maintains high precision even under varying lighting and pose conditions.

## Key Features
* ArcFace Backbone: High-dimensional vector embeddings for superior identity matching.
* Centroid-Based Registration: Averages multiple facial angles to create a Master Identity for users.
* Hybrid Workflow: Research Phase: Developed and validated in Google Colab Pro+.
* Deployment Phase: Optimized for NVIDIA Jetson Nano using CUDA acceleration.
* Hardware Ready: Custom script included for USB camera integration and automated attendance logging.

## Repository Structure
* Face_Detection_Colab.ipynb: The core research notebook for model training and video validation.
* jetson_deployment/: Contains the hardware-optimized Python script and shell installer.
* data/: Sample identity database and registration structures.

## Installation (Jetson Nano)
To deploy on hardware, run the following:
```bash
cd jetson_deployment
chmod +x install_jetson.sh
./install_jetson.sh
python3 main_attendance.py