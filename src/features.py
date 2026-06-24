import math

def dist(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def extract_features(points):

    face_width = dist(points[8],  points[13])   
    ref_20_21  = dist(points[20], points[21])  

    # 1. Eye length ratio
    eye1 = dist(points[9],  points[10])         
    eye2 = dist(points[11], points[12])     
    eye_length_ratio = max(eye1, eye2) / face_width if face_width != 0 else 0

    # 2. Eye distance ratio
    right_eye_center = ((points[9][0]  + points[10][0]) / 2, (points[9][1]  + points[10][1]) / 2)
    left_eye_center  = ((points[11][0] + points[12][0]) / 2, (points[11][1] + points[12][1]) / 2)
    eye_distance_ratio = dist(right_eye_center, left_eye_center) / face_width if face_width != 0 else 0

    # 3. Nose ratio
    nose_ratio = dist(points[15], points[16]) / ref_20_21 if ref_20_21 != 0 else 0

    # 4. Lip size ratio
    lip_denom = dist(points[17], points[18])
    lip_size_ratio = dist(points[2], points[3]) / lip_denom if lip_denom != 0 else 0

    # 5. Lip length ratio
    lip_length_ratio = dist(points[2], points[3]) / ref_20_21 if ref_20_21 != 0 else 0

    # 6. Eyebrow length ratio
    right_brow = dist(points[4], points[5])
    left_brow  = dist(points[6], points[7])
    eyebrow_length_ratio = max(right_brow, left_brow) / face_width if face_width != 0 else 0

    # 7. Aggressive ratio
    aggressive_ratio = dist(points[10], points[19]) / ref_20_21 if ref_20_21 != 0 else 0

    return {
        "eye_length_ratio":     eye_length_ratio,
        "eye_distance_ratio":   eye_distance_ratio,
        "nose_ratio":           nose_ratio,
        "lip_size_ratio":       lip_size_ratio,
        "lip_length_ratio":     lip_length_ratio,
        "eyebrow_length_ratio": eyebrow_length_ratio,
        "aggressive_ratio":     aggressive_ratio,
    }

def add_features(dataset):
    for entry in dataset:
        entry["features"] = extract_features(entry["points"])
    return dataset