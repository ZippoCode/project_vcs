# Folders
PATH = '../data/paintings_db/'
ROOT_PATH_VIDEOS = '../data/videos/'
ROOT_PATH_DETECTED = '../output/paintings/'
PATH_KEYPOINTS_DB = '../output/key-points.pck'
PATH_DESTINATION_PERSON_DETECTED = '../output/person_detected/'
PATH_DATA_CSV = '../data/data.csv'

# Yolo
PATH_YOLO_CFG = '../yolo/cfg/yolov3-obj-test.cfg'
PATH_YOLO_WEIGHTS = '../yolo/yolov3-obj-train_last.weights'
PATH_ORIGINAL = '../../yolo/obj/original/frames/'
PATH_EDIT = '../../yolo/obj/edit/frames/'

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

# YOLO
confThreshold = 0.5  # Confidence threshold
nmsThreshold = 0.4  # Non-maximum suppression threshold
inpWidth = 416  # Width of network's input image
inpHeight = 416  # Height of network's input image
classesFile = "../yolo/coco.names"
