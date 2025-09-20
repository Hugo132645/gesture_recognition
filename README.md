# Prosthetic Arm Gesture Data Project

This project is part of our research into building an intelligent **prosthetic robotic arm** that can understand human hand gestures through computer vision and AI.

## Overview
We are collecting and preprocessing image data of different hand gestures to train machine learning models. The ultimate goal is to connect these models to a prosthetic arm, enabling it to perform natural movements based on user intent.

## Files
- **data_creation.py** – Script for recording hand gesture images using a webcam.
- **merge_data.py** – Script for merging multiple gesture datasets into one.
- **gesture_dataset.ipynb** – Jupyter notebook for testing, training, and experimenting with models.

## Research Goals
- Investigate how AI can interpret hand gestures reliably.
- Build datasets that can generalize to different users and conditions.
- Lay the foundation for **gesture-controlled prosthetics**, where computer vision acts as an interface between human movement and robotic motion.

## Next Steps
- Train CNN or transformer-based models on the gesture dataset.
- Integrate trained models with Arduino Mega + servos for real-time control.
- Experiment with multimodal input (e.g., EMG signals + vision).
