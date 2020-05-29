# Folders
PATH = '../data/paintings_db/'
ROOT_PATH_VIDEOS = '../data/videos/'
ROOT_PATH_DETECTED = '../output/paintings/'
PATH_KEYPOINTS_DB = '../output/key-points.pck'

DILATE_KERNEL_SIZE = (5, 5)
DILATE_ITERATIONS = 2
EROSION_ITERATIONS = 3

# Rectification
ENTROPY_THRESHOLD = 2.0

# Retrieval
FLANN_INDEX_LSH = 6
FLANN_INDEX_KDTREE = 1
RATIO = 0.7
MIN_MATCH_COUNT = 150
