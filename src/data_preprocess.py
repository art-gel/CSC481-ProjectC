import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the extracted features from the CSV file
df = pd.read_csv("extracted_features.csv")

# Select input features (X)
X = df[[
    "eye_length_ratio",
    "eye_distance_ratio",
    "nose_ratio",
    "lip_size_ratio",
    "lip_length_ratio",
    "eyebrow_length_ratio",
    "aggressive_ratio"
]]

# Select target labels (y)
y = df["label"]

# Split the dataset into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Z-score normalization
# (value - mean) / standard deviation
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define target names for classification report
target_names = ["neutral", "smile", "anger"]