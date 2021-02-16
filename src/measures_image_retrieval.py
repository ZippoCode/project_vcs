import pandas
import cv2
import os

# Custom importing
from parameters import *
from painting_retrieval import match_paintings

folder = '../output/paintings_rectified/TEST/'
path_paintings = []


def check_image_retrieval():
    # df = pandas.read_csv(PATH_DATA_CSV)
    # name_images = df['Image']
    matches = dict()
    true_positive = 0
    false_negative = 0
    for root, _, file_names in os.walk(folder):
        for filename in file_names:
            if filename.lower().endswith('.jpg') or filename.lower().endswith('png'):
                path_paintings.append(os.path.join(root, filename))
    sorted(path_paintings)
    print(f"Found {len(path_paintings)} paintings")
    for name_image in path_paintings:
        image = cv2.imread(name_image)
        basename_image = os.path.basename(name_image)
        num_painting = os.path.splitext(basename_image)[0]

        list_retrieval = match_paintings(image)
        best_match, similarity = list_retrieval[0]
        test_painting = num_painting.split('_')[0]
        predicted_paintings = best_match.split('.')[0]
        if test_painting in matches:
            matches[test_painting].append((predicted_paintings, similarity))
        else:
            matches[test_painting] = list((predicted_paintings, similarity))
        print(f"Test: {test_painting} - Predicted: {predicted_paintings} with similarity: {similarity}")
        if test_painting == predicted_paintings:
            true_positive += 1
        else:
            false_negative += 1
    print(matches)
    accuracy = (true_positive + 0.5) / (true_positive + false_negative + 0.5 + 0.5)
    recall = true_positive / (true_positive + false_negative)
    precision = true_positive / (true_positive + 0.5)
    print(f"Accuracy: {accuracy} - Recall: {recall} - Precision: {precision}")


check_image_retrieval()
