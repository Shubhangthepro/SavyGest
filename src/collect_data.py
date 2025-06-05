import cv2
import mediapipe as mp
import csv
import os

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

save_path = "../data/gesture_data.csv"
os.makedirs(os.path.dirname(save_path), exist_ok=True)

cap = cv2.VideoCapture(0)
data = []

def extract_landmarks(hand_landmarks):
    return [coord for lm in hand_landmarks.landmark for coord in (lm.x, lm.y)]

while True:
    _, frame = cap.read()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
    
    cv2.imshow("Collect Gesture", frame)
    key = cv2.waitKey(1)
    
    if key == ord('q'):
        break
    elif key != -1:
        label = chr(key)
        if results.multi_hand_landmarks:
            landmarks = extract_landmarks(results.multi_hand_landmarks[0])
            landmarks.append(label)
            data.append(landmarks)
            print(f"Captured: {label}")

cap.release()
cv2.destroyAllWindows()

with open(save_path, "a", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(data)

print("Data saved.")
