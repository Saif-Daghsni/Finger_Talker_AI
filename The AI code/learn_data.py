import cv2
import mediapipe as mp
import csv
import os

# Init Mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Create folder & database file
if not os.path.exists("data"):
    os.makedirs("data")

csv_path = "data/signs_database.csv"

# Create the file with header if not exists
if not os.path.exists(csv_path):
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        header = ["letter"]
        for i in range(21):  # 21 landmarks
            header += [f"x{i}", f"y{i}", f"z{i}"]
        writer.writerow(header)

# Capture webcam
cap = cv2.VideoCapture(0)

with mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7) as hands:
    letter = input("Enter the letter to record: ").strip().upper()
    print(f"Show the hand sign for '{letter}' (press 's' to save, 'q' to quit).")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # Mirror
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Save landmarks when 's' pressed
                key = cv2.waitKey(1) & 0xFF
                if key == ord('s'):
                    row = [letter]
                    for lm in hand_landmarks.landmark:
                        row += [lm.x, lm.y, lm.z]

                    with open(csv_path, mode="a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow(row)
                    print(f"Saved one sample for '{letter}'")

        cv2.imshow("Collecting Signs", frame)

        # Quit with 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
