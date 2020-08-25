# Folders
import os
from pathlib import Path

__location__ = Path(__file__).parent.parent.absolute()

PATH_PAINTINGS_DB = os.path.join(__location__, 'data/paintings_db/')
SOURCE_PATH_VIDEOS = os.path.join(__location__, 'data/videos/')

PATH_DATA_CSV = os.path.join(__location__, 'data/data.csv')
PATH_ROOMS_ROI_CSV = os.path.join(__location__, 'data/rooms_roi.csv')
PATH_MAP = os.path.join(__location__, 'data/map.png')
PATH_KEYPOINTS_DB = os.path.join(__location__, 'output/key-points.pck')
PATH_OUTPUT = os.path.join(__location__, 'output/')

DESTINATION_PAINTINGS_DETECTED = os.path.join(__location__, 'output/paintings_detected/')
DESTINATION_PAINTINGS_RECTIFIED = os.path.join(__location__, 'output/paintings_rectified/')
DESTINATION_PEOPLE_DETECTED = os.path.join(__location__, 'output/person_detected/')


PATH_OUTPUT_DETECTED_BBOX = os.path.join(__location__, 'output/bbox/')
DESTINATION_PAINTING_BBOX = os.path.join(__location__, 'output/bbox/painting/')

# Yolo
PATH_YOLO_CFG = os.path.join(__location__, 'yolo/cfg/yolov3-obj-test.cfg')
PATH_YOLO_WEIGHTS = os.path.join(__location__, 'yolo/yolov3-obj-train_last.weights')
PATH_COCO_NAMES = os.path.join(__location__, 'yolo/coco.names')

PATH_ORIGINAL = '/fhntjti/custom_data/obj/original/people/'
PATH_EDIT = '/homegng'

DILATE_KERNEL_SIZE = (5, 5)
DILATE_ITERATIONS = 2
EROSION_ITERATIONS = 3

# MULTI-SCALE RETINEX
SIGMA = [7, 40, 125]

# PAINTING DETECTION

# Rectification
ENTROPY_THRESHOLD = 2.0

# Retrieval
FLANN_INDEX_KDTREE = 1
RATIO = 0.7
MIN_MATCH_COUNT = 150

# YOLO
confThreshold = 0.5  # Confidence threshold
nmsThreshold = 0.4  # Non-maximum suppression threshold
inpWidth = 416  # Width of network's input image
inpHeight = 416  # Height of network's input image
classesFile = "../yolo/coco.names"


# MESSAGE COLORS
HEADER = '\033[95m'
OKBLUE = '\033[94m'
OKGREEN = '\033[92m'
WARNING = '\033[93m'
FAIL = '\033[91m'
ENDC = '\033[0m'
BOLD = '\033[1m'
UNDERLINE = '\033[4m'

# COLORS
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (255, 0, 0)
COLOR_BLUE = (0, 0, 255)
COLOR_YELLOW = (0, 255, 255)
COLOR_WHITE = (255, 255, 255)