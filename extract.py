import cv2
import mediapipe as mp
import pandas as pd
import os

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

# Define paths
dataset_path = 'archive'
output_csv = 'hand_landmarks.csv'

# Prepare CSV file
columns = [f'x{i}' for i in range(21)] + [f'y{i}' for i in range(21)] + [f'z{i}' for i in range(21)] + ['label']
df = pd.DataFrame(columns=columns)

# Process each subset ('train' and 'test')
for subset in ['train', 'test']:
    subset_path = os.path.join(dataset_path, subset)
    if not os.path.isdir(subset_path):
        continue

    # Process each gesture folder within the subset
    for gesture in os.listdir(subset_path):
        gesture_path = os.path.join(subset_path, gesture)
        if not os.path.isdir(gesture_path):
            continue

        # Process each image in the gesture folder
        for image_name in os.listdir(gesture_path):
            image_path = os.path.join(gesture_path, image_name)
            image = cv2.imread(image_path)
            if image is None:
                continue

            # Convert the BGR image to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Perform hand landmark detection
            results = hands.process(image_rgb)

            # If landmarks are detected, extract them
            if results.multi_hand_landmarks:
                landmarks = results.multi_hand_landmarks[0]
                row = []
                for lm in landmarks.landmark:
                    row.extend([lm.x, lm.y, lm.z])
                row.append(gesture)
                df.loc[len(df)] = row

# Save the DataFrame to a CSV file
df.to_csv(output_csv, index=False)
