import cv2
import mediapipe as mp
import torch
import joblib
import numpy as np
from model import HandGestureNet  # Your model class

# Load model + label encoder
model = HandGestureNet(num_classes=6)  # Change if needed
model.load_state_dict(torch.load("gesture_model.pt"))
model.eval()
label_encoder = joblib.load("gesture_labels.pkl")

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Music control
import subprocess
def set_volume(level):
    level = max(0, min(100, level))
    subprocess.call(f"osascript -e 'set volume output volume {level}'", shell=True)

def play_music():
    subprocess.call("osascript -e 'tell application \"Music\" to play'", shell=True)

def pause_music():
    subprocess.call("osascript -e 'tell application \"Music\" to pause'", shell=True)

def next_track():
    subprocess.call("osascript -e 'tell application \"Music\" to next track'", shell=True)

def prev_track():
    subprocess.call("osascript -e 'tell application \"Music\" to previous track'", shell=True)

# Gesture → Action Map
def handle_gesture(label):
    print(f"🧠 Predicted Gesture: {label}")
    if label == "fist":
        pause_music()
    elif label == "palm":
        play_music()
    elif label == "one_finger":
        set_volume(20)
    elif label == "two_fingers":
        set_volume(40)
    elif label == "swipe_left":
        prev_track()
    elif label == "swipe_right":
        next_track()

# Start camera
cap = cv2.VideoCapture(0)
prediction_cooldown = 0
last_label = None
COOLDOWN_FRAMES = 40

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            # Flatten landmark
            row = []
            for lm in hand.landmark:
                row.extend([lm.x, lm.y, lm.z])
            x = torch.tensor([row], dtype=torch.float32)

            # Predict
            with torch.no_grad():
                logits = model(x)
                pred = torch.argmax(logits, dim=1).item()
                label = label_encoder.inverse_transform([pred])[0]

            # Display
            cv2.putText(frame, f"Gesture: {label}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            # Handle cooldown & trigger
            if prediction_cooldown == 0 and label != last_label:
                handle_gesture(label)
                last_label = label
                prediction_cooldown = COOLDOWN_FRAMES

    # Cooldown
    if prediction_cooldown > 0:
        prediction_cooldown -= 1

    cv2.imshow("Gesture Inference", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
