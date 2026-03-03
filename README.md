# 🖐️ Hand Gesture Recognition System

A real-time hand gesture recognition system built using MediaPipe and OpenCV.  
This project detects hand landmarks from a webcam feed, classifies gestures, and maps them to interactive actions such as media control and system volume adjustment.

## 🚀 Features
- Real-time hand landmark detection using MediaPipe Tasks API
- Gesture classification:
  - ✋ Open Palm → Play video
  - ✊ Fist → Pause video
  - 👍 Thumbs Up → Gesture recognition
- 🤏 Thumb–Index distance controls system volume
- Visual UI overlays:
  - Play/Pause icons
  - Dynamic volume bar
- Works with live webcam input

## 🛠️ Tech Stack
- Python
- OpenCV
- MediaPipe (Hand Landmarker – latest API)
- NumPy
- Pycaw (Windows volume control)

## 📂 Project Structure

hand-gesture-recognition/
│
├── app.py
├── gesture_utils.py
├── hand_landmarker.task
├── requirements.txt
└── README.md


## 📦 Installation
```bash
pip install opencv-python mediapipe numpy pycaw comtypes
▶️ How to Run
python app.py
🧠 Gesture Logic

Gestures are classified using geometric relationships between hand landmarks rather than simple coordinate comparisons, making the system more robust to hand orientation and scale.

🎯 Use Cases

Human–Computer Interaction

Touchless media control

Assistive technology demos

Computer vision learning projects

📌 Internship Context

This project was developed as part of an AI & Computer Vision internship at SyntecxHub, focusing on practical, real-world applications of machine learning and computer vision.

📜 License

This project is for educational purposes.
