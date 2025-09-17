import cv2
import mediapipe as mp
import numpy as np
import joblib
import time

# Load trained model
model = joblib.load("letter_model.pkl")  # model path

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Open default camera
cap = cv2.VideoCapture(0)

last_time = 0
predicted_letter = ""

with mp_hands.Hands(
    max_num_hands=1,                  
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Predict every 3 seconds
                if time.time() - last_time >= 3:
                    last_time = time.time()
                    landmarks = []
                    for lm in hand_landmarks.landmark:
                        landmarks.append(lm.x)
                        landmarks.append(lm.y)
                    landmarks = np.array(landmarks).reshape(1, -1)

                    predicted_letter = model.predict(landmarks)[0]
                    print(f"[INFO] Detected letter: {predicted_letter}")

        # Display the predicted letter on screen
        cv2.putText(frame, f"Letter: {predicted_letter}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

        cv2.imshow("Hand Sign Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
