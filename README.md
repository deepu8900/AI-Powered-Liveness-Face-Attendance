AI-Powered Liveness Face Attendance System
 Project Overview

This project implements a real-time face attendance system with liveness detection using Deep Learning and Computer Vision. It can identify employees/students from a camera feed and verify they are real (not a photo or video) before marking attendance.

Key Goals:

Real-time face recognition

Liveness detection

Scalable API-based architecture

Production-ready deployment (planned Dockerization & batching)

 Features

Detect and recognize faces in real-time using TensorFlow

Ensure liveness detection to prevent spoofing

Fast and efficient inference for multiple users

Ready for API-based service deployment with FastAPI

Optimizations planned: inference batching, threshold tuning, Dockerization

 Technologies Used
Layer	Technology / Library
Deep Learning	TensorFlow, Keras
Computer Vision	OpenCV, NumPy
Backend / API	FastAPI, Uvicorn
Optimization	Inference batching, threshold tuning
Deployment	Docker (planned)
 Project Structure
AI_Live_Attendance/
│
├─ app/
│   ├─ main.py           # FastAPI entrypoint
│   ├─ model.py          # Face recognition & liveness model
│   ├─ utils.py          # Helper functions (preprocessing, etc.)
│
├─ data/                 # Optional folder for training/testing datasets
├─ requirements.txt      # Python dependencies
└─ README.md             # Project documentation

 Installation

Clone this repository:

git clone https://github.com/deepu8900/AI-Powered-Liveness-Face-Attendance.git
cd AI_Live_Attendance


Create a virtual environment and activate:

python -m venv venv
venv\Scripts\activate   # Windows
source venv/bin/activate  # Linux/Mac


Install dependencies:

pip install -r requirements.txt

 Running the Project

Note: The current version may require minor fixes to run locally.

To run the API server:

python -m uvicorn app.main:app --reload


The API will run at:

http://127.0.0.1:8000
