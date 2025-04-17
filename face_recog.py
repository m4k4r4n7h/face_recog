import cv2
import numpy as np
from statistics import mean
import random
from termcolor import colored  

def mock_face_landmarks(image):
    
    return [{
        "left_eye": [(30, 30), (35, 30), (33, 32)],
        "right_eye": [(60, 30), (65, 30), (63, 32)],
        "nose_tip": [(48, 50), (49, 52), (50, 54)],
        "top_lip": [(40, 60), (45, 62), (50, 60)],
        "chin": [(30, 70), (40, 80), (50, 85), (60, 80), (70, 70)],
        "nose_bridge": [(48, 40), (48, 42), (48, 44)],
        "left_eyebrow": [(30, 20), (32, 18), (34, 20)],
        "right_eyebrow": [(60, 20), (62, 18), (64, 20)],
        "neck": [(35, 90), (45, 95), (55, 90)],  
        "face_size": [(20, 40), (80, 40), (80, 90), (20, 90)],  
        "age": 25  
    }]


def mock_face_encodings(image):
    return [np.random.rand(128)]


def load_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image at path '{image_path}' could not be loaded. Please check the file path.")
    return img


def get_landmarks_and_encodings(image):
    landmarks = mock_face_landmarks(image)
    encodings = mock_face_encodings(image)
    if not landmarks or not encodings:
        return None, None
    return landmarks[0], encodings[0]


def compare_encodings(enc1, enc2):
    distance = np.linalg.norm(enc1 - enc2)
    return max(0, min(100, (1 - distance) * 100))


def compare_regions(landmarks1, landmarks2, region):
    if region not in landmarks1 or region not in landmarks2:
        return None
    pts1 = np.array(landmarks1[region], dtype=np.float32)
    pts2 = np.array(landmarks2[region], dtype=np.float32)
    distances = np.linalg.norm(pts1 - pts2, axis=1)
    return max(0, min(100, 100 - mean(distances)))


def extract_skin_color(image, landmarks):
    if len(landmarks['nose_bridge']) < 2 or len(landmarks['nose_tip']) < 3:
        return [0, 0, 0]
    top = landmarks['nose_bridge'][1]
    bottom = landmarks['nose_tip'][2]
    x1, y1 = top
    x2, y2 = bottom
    region = image[y1:y2, x1:x2]
    return np.mean(region, axis=(0, 1)) if region.size else [0, 0, 0]


def color_similarity(color1, color2):
    return 100 - np.linalg.norm(np.array(color1) - np.array(color2)) / 255 * 100


def compare_hair_region(image1, image2, landmarks1, landmarks2):
    try:
        brow1 = np.array(landmarks1["left_eyebrow"] + landmarks1["right_eyebrow"])
        brow2 = np.array(landmarks2["left_eyebrow"] + landmarks2["right_eyebrow"])

        y1 = min(pt[1] for pt in brow1) - 40
        y2 = min(pt[1] for pt in brow2) - 40
        x1 = min(pt[0] for pt in brow1) - 20
        x2 = min(pt[0] for pt in brow2) - 20

        hair1 = image1[max(0, y1):max(0, y1)+40, max(0, x1):max(0, x1)+80]
        hair2 = image2[max(0, y2):max(0, y2)+40, max(0, x2):max(0, x2)+80]

        if hair1.size == 0 or hair2.size == 0:
            return None

        color1 = np.mean(hair1, axis=(0, 1))
        color2 = np.mean(hair2, axis=(0, 1))
        return color_similarity(color1, color2)

    except:
        return None


def compare_face_size(landmarks1, landmarks2):
    size1 = np.linalg.norm(np.array(landmarks1['face_size'][0]) - np.array(landmarks1['face_size'][2]))
    size2 = np.linalg.norm(np.array(landmarks2['face_size'][0]) - np.array(landmarks2['face_size'][2]))
    return max(0, min(100, 100 - abs(size1 - size2)))


def compare_neck(landmarks1, landmarks2):
    neck_dist1 = np.linalg.norm(np.array(landmarks1['neck'][0]) - np.array(landmarks1['neck'][1]))
    neck_dist2 = np.linalg.norm(np.array(landmarks2['neck'][0]) - np.array(landmarks2['neck'][1]))
    return max(0, min(100, 100 - abs(neck_dist1 - neck_dist2)))


def compare_age(age1, age2):
    age_diff = abs(age1 - age2)
    return max(0, min(100, 100 - age_diff * 2))  


def main():
    print("\nFacial Similarity Analyzer")
    print("-" * 32)

    
    path1 = input("Enter the path for the first image: ")
    path2 = input("Enter the path for the second image: ")

    try:
        img1 = load_image(path1)
        img2 = load_image(path2)
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        return

    landmarks1, enc1 = get_landmarks_and_encodings(img1)
    landmarks2, enc2 = get_landmarks_and_encodings(img2)

    if not landmarks1 or not landmarks2:
        print("\nCould not detect faces or landmarks in one or both images.")
        return

    facial_regions = {
        "left_eye": "Left Eye ",
        "right_eye": "Right Eye ",
        "nose_tip": "Nose ",
        "top_lip": "Mouth ",
        "chin": "Jawline "
    }

    feature_scores = {}
    for key, label in facial_regions.items():
        score = compare_regions(landmarks1, landmarks2, key)
        feature_scores[label] = score if score is not None else "Not detected"

    encoding_score = compare_encodings(enc1, enc2)
    skin1 = extract_skin_color(img1, landmarks1)
    skin2 = extract_skin_color(img2, landmarks2)
    skin_score = color_similarity(skin1, skin2)
    hair_score = compare_hair_region(img1, img2, landmarks1, landmarks2)
    neck_score = compare_neck(landmarks1, landmarks2)
    face_size_score = compare_face_size(landmarks1, landmarks2)
    age_score = compare_age(landmarks1['age'], landmarks2['age'])

    all_scores = [v for v in feature_scores.values() if isinstance(v, (int, float))]
    all_scores.extend([encoding_score, neck_score, face_size_score, age_score])
    if hair_score:
        all_scores.append(hair_score)

    overall_similarity = mean(all_scores)

    print("\nFacial Feature Similarity")
    print("-" * 30)
    print("")
    for label, score in feature_scores.items():
        print(colored(f"{label:10} : {score if isinstance(score, str) else f'{score:.2f}%'}","green"))

    print(colored(f"Skin Tone  : {skin_score:.2f}%","red"))
    #print("[NOTE] Skin color similarity may be influenced by filters, lighting, or other factors in the image.")
    print(colored(f"Hair Color : {hair_score:.2f}%","red"))
    #print("[NOTE] Hair color similarity may be influenced by filters, lighting, or other factors in the image.")
    print(colored(f"Neck Size  : {neck_score:.2f}%","green"))
    print(colored(f"Face Size  : {face_size_score:.2f}%","green"))
    print(colored(f"Age        : {age_score:.2f}%","green"))
    print("")
    print(colored(f"+ [NOTE] Skin color and Hair colour may be influenced by filters, lighting, or other factors in the image.","red"))
    print("")
    print("-" * 33)
    print(colored(f"Face Encoding Similarity: {encoding_score:.2f}%","red"))
    print("-" * 33)
    print("")
    
    print("-" * 33)
    print(colored(f"Overall Visual Similarity: {overall_similarity:.2f}%", "green"))
    print("-" * 33)
    print("")


if __name__ == "__main__":
    main()

