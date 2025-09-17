import cv2
import mediapipe as mp
import csv
import os
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Create folder for data
if not os.path.exists("data"):
    os.makedirs("data")

csv_path = "data/hand_landmarks.csv"

# Check if file exists, if not, create with header
file_exists = os.path.isfile(csv_path)

csv_file = open(csv_path, mode="a", newline="")
csv_writer = csv.writer(csv_file)

if not file_exists:
    # Header: label + 42 landmarks (x, y for 21 points)
    header = ["label"]
    for i in range(21):
        header += [f"x{i}", f"y{i}"]
    csv_writer.writerow(header)

cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    max_num_hands=1,  # one hand at a time
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:

    while True:
        # Ask user which letter to record
        label = input("Enter the letter to record (or type 'exit' to quit): ")
        if label.lower() == "exit":
            break
        print(f"[INFO] Starting recording for letter: {label}")

        # Collect 25 samples per letter
        for j in range(25):
            print(f"[INFO] Starting sample {j+1}/25 for letter {label}")

            # 1️⃣ Record for 2 seconds
            start_time = time.time()
            duration = 2  # seconds to record each sample

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        landmarks = []
                        for lm in hand_landmarks.landmark:
                            landmarks.append(lm.x)
                            landmarks.append(lm.y)

                        row = [label] + landmarks
                        csv_writer.writerow(row)

                        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                elapsed = time.time() - start_time
                remaining = int(duration - elapsed)
                cv2.putText(frame, f"Recording {label} | Time left: {remaining}s",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                cv2.imshow("Collecting Data", frame)

                if elapsed >= duration:
                    print(f"[INFO] Finished recording sample {j+1}/25")
                    break

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # 2️⃣ Pause for 2 seconds before next sample
            print("[INFO] Adjust your hand position...")
            time.sleep(2)

# Cleanup
cap.release()
csv_file.close()
cv2.destroyAllWindows()
