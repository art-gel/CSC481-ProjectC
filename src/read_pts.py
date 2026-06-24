import os

def read_pts(file_path):
    points = []
    with open(file_path, "r") as f:
        start = False

        for line in f:
            line = line.strip()
            if line == "{":
                start = True
                continue
            if line == "}":
                break
            if start:
                x, y = line.split()
                points.append((float(x), float(y)))
    return points


def load_dataset(folder="data/points_22"):
    dataset = []

    for item_folder in os.listdir(folder):
        item_path = os.path.join(folder, item_folder)
        if not os.path.isdir(item_path):
            continue
        for file in os.listdir(item_path):
            if not file.endswith(".pts"):
                continue

            parts = file.split("-")
            expr_str = parts[2].replace(".pts", "")

            # Skip expression 5
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