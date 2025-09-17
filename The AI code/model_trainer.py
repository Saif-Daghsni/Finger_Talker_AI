import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

csv_path = "data/hand_landmarks.csv"
model_path = "letter_model.pkl"

# Step 1: Load CSV (first row is header now)
data = pd.read_csv(csv_path)

# Step 2: Remove rows with missing data
initial_rows = data.shape[0]
data = data.dropna()
removed_rows = initial_rows - data.shape[0]
print(f"[INFO] Removed {removed_rows} rows with missing data.")

# Step 3: Features and labels
X = data.drop("label", axis=1)  # all columns except 'label'
y = data["label"]               # target column

# Step 4: Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 5: Train Random Forest classifier
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)

# Step 6: Accuracy
accuracy = clf.score(X_test, y_test)
print(f"[INFO] Model accuracy: {accuracy*100:.2f}%")

# Step 7: Save trained model
joblib.dump(clf, model_path)
print(f"[INFO] Model saved as {model_path}")
