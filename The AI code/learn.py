import cv2
import mediapipe as mp
import pandas as pd
import numpy as np

# Load your hand landmarks database
df = pd.read_csv("data/hand_landmarks.csv")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Ask the user which letter to practice
target_letter = input("Enter the letter you want to practice: ").upper()

# Filter the CSV for that letter using 'label'
letter_data = df[df["label"] == target_letter]
if letter_data.empty:
    print(f"No data found for letter '{target_letter}'.")
    exit()

# Convert all reference samples to numpy arrays
references = letter_data.drop("label", axis=1).to_numpy()
# Compute average landmarks for “ideal hand”
avg_hand = np.mean(references, axis=0)

# Prepare the MediaPipe hand connections for drawing
hand_connections = mp_hands.HAND_CONNECTIONS

# Function to convert flat array to landmark list
def array_to_landmarks(arr, img_w, img_h):
    return [(int(arr[i]*img_w), int(arr[i+1]*img_h)) for i in range(0, len(arr), 2)]

# Open webcam
cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:
    feedback = "Show your hand for the letter!"

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        # Draw the ideal/reference hand as a skeleton
        landmarks_points = array_to_landmarks(avg_hand, w, h)
        for connection in hand_connections:
            start_idx, end_idx = connection
            cv2.line(frame, landmarks_points[start_idx], landmarks_points[end_idx], (255, 0, 0), 3)
        for point in landmarks_points:
            cv2.circle(frame, point, 5, (255, 0, 0), -1)

        # Detect user's hand and compare
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.append(lm.x)
                    landmarks.append(lm.y)
                user_hand = np.array(landmarks)

                # Compute distance to reference samples
                distances = np.linalg.norm(references - user_hand, axis=1)
                min_dist = np.min(distances)

                # Decide feedback
                if min_dist < 0.09:
                    feedback = "Perfect"
                elif min_dist < 0.1:
                    feedback = "Good"
                else:
                    feedback = "Bad"

                # Draw user's hand landmarks
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Show feedback
        cv2.putText(frame, f"Practice: {target_letter}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, f"Feedback: {feedback}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "Blue: Ideal Hand", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        cv2.imshow("Learn Mode", frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
