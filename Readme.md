✋ SavyGest: Smart Real-Time Hand Gesture Recognition
A sleek, real-time hand gesture recognition system built with Python, MediaPipe, TensorFlow, and OpenCV.
SavyGest enables natural and intuitive control of games and applications using custom hand gestures.

🚀 Features
✅ Real-Time Detection
Capture hand gestures instantly from webcam feed using MediaPipe’s fast hand tracking.

✅ Custom Gesture Set
Recognizes A–Z, 0–9, and special characters. Fully extendable for custom symbols.

✅ Deep Learning Powered
Backed by a trained TensorFlow Keras model for high-accuracy classification.

✅ App/Game Controller
Simulate keyboard or mouse inputs via gestures — control any game or app intuitively.

✅ Plug & Play
Just run the script — no complex setup. Easy to integrate into any Python-based project.

✅ Open-Source & Extensible
Designed for developers: easy to read, modify, and expand with new features or gestures.

🛠️ Tech Stack
🐍 Python 3.10

🧠 TensorFlow / Keras – Gesture classification

👋 MediaPipe – Hand landmark tracking

🎥 OpenCV – Webcam and image processing

🧮 scikit-learn – Model utilities

🎮 pyautogui – Control keyboard and mouse with gestures

🧪 How It Works
1️⃣ Collect Gesture Data
Run the script to capture webcam hand landmarks and label them as custom gestures.

2️⃣ Train the Model
Use train_model.py to train a neural network on your collected data.

3️⃣ Real-Time Prediction
Launch real_time_predictor.py to recognize gestures live from webcam feed.

4️⃣ Control Applications
Optionally run game_controller.py to simulate key/mouse events and control games or apps.

📥 Run Locally
bash
Copy
Edit
# Install dependencies
pip install tensorflow mediapipe opencv-python pandas scikit-learn pyautogui

# Train the model
python train_model.py

# Start real-time gesture recognition
python real_time_predictor.py

# Optional: Enable game/app control
python game_controller.py
💡 Ensure your webcam is working and Python 3.10 is installed.

💬 Author
Crafted with ❤️ by Shubhang Shrivastav
🔗 GitHub: https://github.com/Shubhangthepro
