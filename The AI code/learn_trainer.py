import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load database
df = pd.read_csv("data/signs_database.csv")

# Features = all landmark values, Labels = letter
X = df.drop("letter", axis=1)
y = df["letter"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_train, y_train)

# Test accuracy
print("Accuracy:", clf.score(X_test, y_test))

# Save model
joblib.dump(clf, "signs_model.pkl")
