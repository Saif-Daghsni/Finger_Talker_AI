import pandas as pd

# Path to your file
csv_path = "data/hand_landmarks.csv"
output_path = "data/hand_landmarks_clean.csv"

# Load CSV (no header, first column = label)
df = pd.read_csv(csv_path, header=None)

# Define corrections
corrections = {
    "a": "A",
    "b": "B",
    "t": "T",
    "FF": "F",
    "SSPACE": "SPACE",
    "BACKSPACE": None  # None = delete row
}

# Apply corrections
def fix_label(label):
    if label in corrections:
        return corrections[label]
    return label

df[0] = df[0].apply(fix_label)

# Remove rows where label became None (BACKSPACE)
df = df.dropna(subset=[0])

# Save cleaned file
df.to_csv(output_path, index=False, header=False)

print(f"[INFO] Cleaned file saved as {output_path}")
