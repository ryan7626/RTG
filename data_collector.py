import cv2
import mediapipe as mp
import csv
import os

# Setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Output file
csv_file = "gesture_data.csv"
if not os.path.exists(csv_file):
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([f"x{i}" for i in range(63)] + ["label"])

# Label mapping
gesture_keys = {
    "f": "fist",
    "p": "palm",
    "1": "one_finger",
    "2": "two_fingers",
    "l": "swipe_left",
    "r": "swipe_right"
}

# Start camera
cap = cv2.VideoCapture(0)
print("Press keys to label gestures. (f=Fist, p=Palm, 1=One Finger, l=Left Swipe, r=Right Swipe)")
print("Press ESC to exit.")

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            # Flatten landmark vector
            data_row = []
            for lm in hand.landmark:
                data_row.extend([lm.x, lm.y, lm.z])

            # Show which key to press
            cv2.putText(frame, "Press key to label (f, p, 1, 2, l, r)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                cap.release()
                cv2.destroyAllWindows()
                exit()

            elif chr(key) in gesture_keys:
                label = gesture_keys[chr(key)]
                with open(csv_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(data_row + [label])
                print(f"Saved: {label}")

    cv2.imshow("Data Collection - Press key to label", frame)

cap.release()
cv2.destroyAllWindows()
