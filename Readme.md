✋ SavyGest: Smart Real-Time Hand Gesture Recognition
A sleek, modern, and versatile hand gesture recognition system built with Python, MediaPipe, TensorFlow, and OpenCV. Effortlessly recognizes custom gestures in real time to enable natural and intuitive control of apps and devices.

🚀 Features
✅ Real-time Gesture Detection
Detects hand landmarks and classifies gestures instantly from webcam input.

✅ Custom Gesture Set
Supports alphabets (A-Z), digits (0-9), and special characters tailored to your needs.

✅ Deep Learning Powered
Uses a neural network trained with TensorFlow Keras for high accuracy.

✅ Plug & Play Integration
Use recognized gestures to control games, software, or custom applications.

✅ Open & Extensible
Python-based, easy to set up, customize, and extend for your projects.

🛠️ Built With
Python 3.10

TensorFlow / Keras

MediaPipe

OpenCV

scikit-learn

pyautogui

🧪 How to Use
Collect Gesture Data
Capture hand landmark data via webcam using the data collection script.

Train the Model
Train your custom gesture recognition model (train_model.py).

Run Real-Time Prediction
Launch live gesture detection with real_time_predictor.py.

Control Applications
Use gestures to interact with apps via keyboard/mouse simulation.

📥 Run Locally
bash
Copy code
pip install tensorflow mediapipe opencv-python pandas scikit-learn pyautogui
python train_model.py
python real_time_predictor.py
python game_controller.py # Optional
💬 Author
Crafted with ❤️ by Shubhang Shrivastav
GitHub Profile : https://github.com/Shubhangthepro
