import cv2
from os import listdir

# Custom importing
from parameters import *


def match_paitings(query_painting):
    """
        Given a rectified painting return a ranked list of all the images in the painting DB,
        sorted by descending similarity with the detected painting.
    :param query_painting: numpy.ndarray with shape (H, W)
    :return: a list of (string, similarity)
    """
    images_name = [file for file in listdir(PATH)]

    # Init detector and matcher
    detector = cv2.xfeatures2d.SIFT_create()
    FLANN_INDEX_KDTREE = 1
    flann_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    matcher = cv2.FlannBasedMatcher(flann_params, search_params)

    #   Read only image 045 because I'm sure that it is contained -> QUERY IMAGE
    kp_q, des_q = detector.detectAndCompute(query_painting, None)

    # Detect -> TRAIN
    database_features = dict()
    for path_image in images_name:
        image = cv2.imread(PATH + path_image, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, query_painting.shape[:2])
        kp, des = detector.detectAndCompute(image, None)
        if kp is not None and des is not None:
            database_features[path_image] = (kp, des)

    database_matches = dict()
    for image, (kp_t, des_t) in database_features.items():
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

    sorted_matches = sorted(database_matches.items(), key=lambda x: x[1], reverse=True)
    return sorted_matches
