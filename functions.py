from scipy.spatial.distance import cdist
import numpy as np
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


def extract_features_with_brief(images):
    # Initialize the FAST detector
    fast = cv2.FastFeatureDetector_create()

    # Initialize the BRIEF extractor
    brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()

    keypoints_list = []
    descriptors_list = []

    for img in images:
        # Detect keypoints using FAST
        keypoints = fast.detect(img, None)

        # Compute descriptors using BRIEF
        keypoints, descriptors = brief.compute(img, keypoints)

        keypoints_list.append(keypoints)
        descriptors_list.append(descriptors)

    return keypoints_list, descriptors_list


def create_image_subset(images, num_images, seed=42):
    import random
    random.seed(seed)
    subset_indices = random.sample(range(len(images)), num_images)
    subset_indices.sort()
    subset_images = [images[i] for i in subset_indices]
    return subset_images, subset_indices


def extract_features_with_orb(images):
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


def match_features_sift(descriptors_list):
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
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


def extract_features_with_sift(images):
    sift = cv2.SIFT_create()
    keypoints_list = []
    descriptors_list = []

    for img in images:
        keypoints, descriptors = sift.detectAndCompute(img, None)
        keypoints_list.append(keypoints)
        descriptors_list.append(descriptors)

    return keypoints_list, descriptors_list


def identify_revisited_locations(matches_list, threshold=30):
    revisited_pairs = []
    for i, matches in enumerate(matches_list):
        for j, match in matches:

            good_matches = [m for m in match if m.distance < 70]
            if len(good_matches) > threshold:
                revisited_pairs.append((i, j, len(good_matches)))
    return revisited_pairs


def simulate_ground_truth(revisited_pairs, images):
    ground_truth = set()
    for i in range(0, len(images), 10):
        if i >= 10 and i % 10 == 0:
            ground_truth.add((i - 10, i))
    predicted = set((i, j) for i, j, _ in revisited_pairs)

    return predicted, ground_truth


def load_ground_truth_poses(file_path):
    """
    Load ground truth poses from a text file.

    Args:
        file_path (str): Path to the ground truth file.

    Returns:
        poses (list): List of poses as 4x4 numpy arrays.
    """
    poses = []
    with open(file_path, 'r') as f:
        for line in f:
            data = line.strip().split()
            if len(data) != 12:
                continue  # Skip invalid lines
            pose = np.eye(4)
            pose[:3, :4] = np.array(data, dtype=float).reshape(3, 4)
            poses.append(pose)
    return poses


def extract_positions(poses):
    """
    Extract positions (translations) from poses.

    Args:
        poses (list): List of 4x4 pose matrices.

    Returns:
        positions (numpy.ndarray): Array of positions of shape (N, 3).
    """
    positions = np.array([pose[:3, 3] for pose in poses])
    return positions


def compute_pairwise_distances(positions):
    """
    Compute pairwise Euclidean distances between positions.

    Args:
        positions (numpy.ndarray): Array of positions (N, 3).

    Returns:
        distances (numpy.ndarray): Pairwise distance matrix of shape (N, N).
    """
    distances = cdist(positions, positions, metric='euclidean')
    return distances


def generate_ground_truth_pairs(distances, threshold):
    """
    Generate ground truth pairs where the distance is below the threshold.

    Args:
        distances (numpy.ndarray): Pairwise distance matrix (N, N).
        threshold (float): Distance threshold for revisits.

    Returns:
        ground_truth_pairs (set): Set of tuples (i, j) where i and j are indices of revisited frames.
    """
    N = distances.shape[0]
    ground_truth_pairs = set()
    for i in range(N):
        for j in range(i+1, N):  # Avoid duplicates and self-comparison
            if distances[i, j] < threshold:
                ground_truth_pairs.add((i, j))
    return ground_truth_pairs


def calculate_metrics(predicted_pairs, ground_truth_pairs, num_frames):
    """
    Calculate precision, recall, and accuracy based on predicted and ground truth pairs.

    Args:
        predicted_pairs (set): Set of predicted revisited location pairs (i, j).
        ground_truth_pairs (set): Set of ground truth revisited location pairs (i, j).
        num_frames (int): Total number of frames considered.

    Returns:
        precision (float): Precision of the predictions.
        recall (float): Recall of the predictions.
        accuracy (float): Accuracy of the predictions.
    """

    true_positives = predicted_pairs & ground_truth_pairs
    TP = len(true_positives)

    false_positives = predicted_pairs - ground_truth_pairs
    FP = len(false_positives)

    false_negatives = ground_truth_pairs - predicted_pairs
    FN = len(false_negatives)

    total_pairs = num_frames * (num_frames - 1) // 2

    TN = total_pairs - (TP + FP + FN)

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0

    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0

    accuracy = (TP + TN) / total_pairs if total_pairs > 0 else 0.0

    return precision, recall, accuracy
