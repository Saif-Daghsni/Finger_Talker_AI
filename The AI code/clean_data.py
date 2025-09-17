import pandas as pd
import string
import os

csv_path = "data/hand_landmarks.csv"

# Check if file exists
if not os.path.isfile(csv_path):
    print(f"[ERROR] File {csv_path} does not exist.")
    exit()

# Load dataset
df = pd.read_csv(csv_path)

# 1️⃣ Remove rows with missing values
df = df.dropna()

# 2️⃣ Keep only valid alphabet letters
valid_letters = list(string.ascii_letters)
df = df[df['label'].isin(valid_letters)]

# Ask user which letter to delete
letter_to_delete = input("Enter the letter to delete from dataset: ").strip()

if letter_to_delete not in valid_letters:
    print("[ERROR] Not a valid letter. Exiting.")
    exit()

# 3️⃣ Delete all rows with that letter
before_count = len(df)
df = df[df['label'] != letter_to_delete]
after_count = len(df)

# 4️⃣ Reset index
df = df.reset_index(drop=True)

# Save updated CSV (overwrite original)
df.to_csv(csv_path, index=False)

print(f"[INFO] Deleted all rows for letter '{letter_to_delete}'")
print(f"Rows before: {before_count}, Rows after: {after_count}")
print(f"Updated file saved as '{csv_path}'")
