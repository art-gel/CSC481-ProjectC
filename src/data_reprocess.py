import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("extracted_features.csv")

X = df[["eye_length_ratio", "eye_distance_ratio", "nose_ratio",
        "lip_size_ratio", "lip_length_ratio", "eyebrow_length_ratio",
        "aggressive_ratio"]].values

y = df["label"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

target_names = ["neutral", "smile", "anger"]

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))