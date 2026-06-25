import pandas as pd
from src.read_pts import load_dataset
from src.features import add_features

def main():
    dataset = load_dataset()
    dataset = add_features(dataset) # Add features to the dataset
    print(f"Extracted features for {len(dataset)} images!")

    rows = []
    # Convert the dataset into a DataFrame for a readable CSV format
    for entry in dataset: 
        feats = entry["features"]
        rows.append({
            "file":                 entry["file"],
            "label":                entry["label"],
            "eye_length_ratio":     round(feats["eye_length_ratio"], 4),
            "eye_distance_ratio":   round(feats["eye_distance_ratio"], 4),
            "nose_ratio":           round(feats["nose_ratio"], 4),
            "lip_size_ratio":       round(feats["lip_size_ratio"], 4),
            "lip_length_ratio":     round(feats["lip_length_ratio"], 4),
            "eyebrow_length_ratio": round(feats["eyebrow_length_ratio"], 4),
            "aggressive_ratio":     round(feats["aggressive_ratio"], 4),
        })

    # Save the extracted features to a CSV file
    df = pd.DataFrame(rows)
    df.to_csv("extracted_features.csv", index=False)
    print("Saved extracted_features.csv")

if __name__ == "__main__":
    main()