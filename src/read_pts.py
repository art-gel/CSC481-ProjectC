import os

def read_pts(file_path):
    points = []
    with open(file_path, "r") as f:
        start = False

        for line in f: # Read each line in the file
            line = line.strip()
            if line == "{":
                start = True
                continue
            if line == "}":
                break

            if start: # Only read points within the curly braces
                x, y = line.split() # Split the line into x and y coordinates
                points.append((float(x), float(y)))
    return points

# Load the dataset from the specified folder
def load_dataset(folder="data/points_22"):
    dataset = []

    for item_folder in os.listdir(folder):
        item_path = os.path.join(folder, item_folder)
        if not os.path.isdir(item_path):
            continue
        for file in os.listdir(item_path):
            if not file.endswith(".pts"):
                continue
            
            # Extract the expression label from the filename( 01, 02, 03)
            parts = file.split("-")
            expr_str = parts[2].replace(".pts", "")

            # Skip expression 5 (lighting condition is not needed)
            if expr_str.startswith("5") or expr_str.startswith("05"):
                continue

            label = int(expr_str)

            file_path = os.path.join(item_path, file)
            points = read_pts(file_path)

            dataset.append({
                "file": file,
                "label": label,
                "points": points
            })

    return dataset