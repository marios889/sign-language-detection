# Sign Language Detection using Random Forest

This project is a **Sign Language Detection system** built from scratch using **Python**, **OpenCV**, **MediaPipe**, and **scikit-learn**. The system detects hand signs for letters in real-time using a **Random Forest classifier**.

## Features

- Collects video data for each letter of the alphabet.
- Divides each video into frames and extracts hand landmarks from each frame.
- Stores all extracted hand points in a structured data file.
- Trains a Random Forest model using the collected data.
- Predicts letters in real-time from live webcam input.
- Fully built from scratch with no pre-existing datasets.

## Requirements

- Python 3.10+
- OpenCV
- MediaPipe
- NumPy
- scikit-learn
- pickle

Install required packages:

```bash
pip install opencv-python mediapipe numpy scikit-learn
