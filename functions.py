import os
import cv2
import random
from sklearn.metrics import precision_score, recall_score


def load_images_from_folder(folder):
    images = []
    filenames = sorted(os.listdir(folder))
    for filename in filenames:
        img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
    return images


def create_image_subset(images, num_images, seed=42):

    if num_images > len(images):
        raise ValueError(
            "Number of images to select is greater than the total number of available images.")

    # Set the random seed for reproducibility
    random.seed(seed)

    # Randomly select a subset of indices
    subset_indices = random.sample(range(len(images)), num_images)
    subset_indices.sort()  # Optional: sort indices to maintain order

    # Extract the subset of images using the selected indices
    subset_images = [images[i] for i in subset_indices]

    return subset_images, subset_indices


def extract_features(images):
    orb = cv2.ORB_create()
    keypoints_list = []
    descriptors_list = []
    for img in images:
        keypoints, descriptors = orb.detectAndCompute(img, None)
        keypoints_list.append(keypoints)
        descriptors_list.append(descriptors)
    return keypoints_list, descriptors_list


def match_features(descriptors_list):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches_list = []
    num_images = len(descriptors_list)
    for i in range(num_images):
        matches = []
        for j in range(num_images):
            if i != j:
                matches_ij = bf.match(descriptors_list[i], descriptors_list[j])
                matches_ij = sorted(matches_ij, key=lambda x: x.distance)
                matches.append((j, matches_ij))
        matches_list.append(matches)
    return matches_list


def identify_revisited_locations(matches_list, threshold=30):
    revisited_pairs = []
    for i, matches in enumerate(matches_list):
        for j, match in matches:
            good_matches = [m for m in match if m.distance < 50]
            if len(good_matches) > threshold:
                revisited_pairs.append((i, j, len(good_matches)))
    return revisited_pairs


def simulate_ground_truth(revisited_pairs, images):
    ground_truth = set()
    for i in range(0, len(images), 10):
        ground_truth.add((i, i))
    predicted = set((i, j) for i, j, _ in revisited_pairs)

    return predicted, ground_truth


def calculate_precision_recall(predicted, ground_truth, images):
    # Create a list of all possible pairs
    all_pairs = set((i, j) for i in range(len(images))
                    for j in range(len(images)) if i != j)

    # Create labels
    y_true = [1 if pair in ground_truth else 0 for pair in all_pairs]
    y_pred = [1 if pair in predicted else 0 for pair in all_pairs]

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    return precision, recall
