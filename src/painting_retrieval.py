import cv2
import os
import _pickle as pickle

# Custom importing
from parameters import *


def save_keypoints(images_name, detector):
    print('\t> Save key-points')
    database_features = dict()
    for path_image in images_name:
        image = cv2.imread(PATH + path_image, cv2.IMREAD_GRAYSCALE)
        kp, des = detector.detectAndCompute(image, None)
        if kp is not None and des is not None:
            database_features[path_image] = des
    with open(PATH_KEYPOINTS_DB, 'wb') as file:
        pickle.dump(database_features, file)
    return database_features


def read_keypoints():
    print('\t> Read key-points ...')
    database_features = dict()
    with open(PATH_KEYPOINTS_DB, 'rb') as file:
        database = pickle.load(file)
    for elem in database:
        database_features[elem] = database[elem]
    return database_features


def match_paitings(query_painting):
    """
        Given a rectified painting return a ranked list of all the images in the painting DB,
        sorted by descending similarity with the detected painting.

    :param query_painting: numpy.ndarray with shape (H, W)
    :return: a list of (string, similarity)
    """
    print('Start Retrieval ...')
    images_name = [file for file in os.listdir(PATH)]

    # Init detector and matcher
    detector = cv2.xfeatures2d.SIFT_create()
    flann_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    matcher = cv2.FlannBasedMatcher(flann_params, search_params)

    kp_q, des_q = detector.detectAndCompute(query_painting, None)

    if not os.path.exists(PATH_KEYPOINTS_DB):
        database_features = save_keypoints(images_name, detector)
    else:
        database_features = read_keypoints()

    database_matches = dict()
    print('\t> Matching key-points ...')
    for image, des_t in database_features.items():
        raw_matches_one = matcher.knnMatch(des_q, des_t, k=2)
        good_matches_one = []
        for m in raw_matches_one:
            if len(m) == 2 and m[0].distance < m[1].distance * RATIO:
                good_matches_one.append(m[0])
        raw_matches_two = matcher.knnMatch(des_t, des_q, k=2)
        good_matches_two = []
        for m in raw_matches_two:
            if len(m) == 2 and m[0].distance < m[1].distance * RATIO:
                good_matches_two.append(m[0])

        database_matches[image] = 0
        for match_one in good_matches_one:
            for match_two in good_matches_two:
                if match_one.queryIdx == match_two.trainIdx and match_one.trainIdx == match_two.queryIdx:
                    database_matches[image] += 1

    print('\t> Sorting ....')
    sorted_matches = sorted(database_matches.items(), key=lambda x: x[1], reverse=True)
    print('End matching')
    return sorted_matches
