import cv2
from os import listdir

path = '../data/paintings_db/'
RATIO = 0.7
MIN_MATCH_COUNT = 20


def extract_features(path_images, detector):
    """
        Given a vector of paths return a dictionary of key-points and descriptor fpr each image
    :return: 
    """
    features = dict()
    if path_images is None:
        return
    for path_image in path_images:
        image = cv2.imread(path + path_image, cv2.IMREAD_GRAYSCALE)
        kp, des = detector.detectAndCompute(image, None)
        if kp is not None and des is not None:
            features[path_image] = (kp, des)
    return features


def match_paitings(query_painting):
    """
        Given a rectified painting return a ranked list of all the images in the painting DB,
        sorted by descending similarity with the detected painting.

    :param query_painting: numpy.ndarray with shape (H, W)
    :return: a list of (string, similarity)
    """
    images_name = [file for file in listdir(path)]

    # Init detector and matcher
    detector = cv2.xfeatures2d.SIFT_create()
    FLANN_INDEX_KDTREE = 1
    flann_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    matcher = cv2.FlannBasedMatcher(flann_params, search_params)

    #   Read only image 045 because I'm sure that it is contained -> QUERY IMAGE
    kp_q, des_q = detector.detectAndCompute(query_painting, None)

    # Detect -> TRAIN
    database_paiting = extract_features(path_images=images_name, detector=detector)

    database_matches = dict()
    for image, (kp, des) in database_paiting.items():
        raw_matches = matcher.knnMatch(des_q, des, k=2)
        good_matches = []
        for m, n in raw_matches:
            if m.distance < n.distance * RATIO:
                good_matches.append(m)
        distance = 0
        if len(good_matches) > MIN_MATCH_COUNT:
            for m in good_matches:
                distance += m.distance
            database_matches[image] = distance

    sorted_matches = sorted(database_matches.items(), key=lambda x: x[1])
    return sorted_matches
