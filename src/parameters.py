# Folders
import os
from pathlib import Path
import numpy as np

__location__ = Path(__file__).parent.parent.absolute()

# Source folders
SOURCE_PAINTINGS_DB = os.path.join(__location__, 'data/paintings_db/')
SOURCE_PATH_VIDEOS = os.path.join(__location__, 'data/videos/')
PATH_DATA_CSV = os.path.join(__location__, 'data/data.csv')
PATH_ROOMS_ROI_CSV = os.path.join(__location__, 'data/rooms_roi.csv')
PATH_MAP = os.path.join(__location__, 'data/map.png')
PATH_KEYPOINTS_DB = os.path.join(__location__, 'output/db_descriptors.json')
PATH_MEANS = os.path.join(__location__, 'output/means.json')
PATH_WEIGHTS = os.path.join(__location__, 'output/weights.json')
PATH_COVARIANCES = os.path.join(__location__, 'output/covariances.json')
PATH_PROBABILITIES = os.path.join(__location__, 'output/probabilities.json')
PATH_FISHER_VECTOR_DB = os.path.join(__location__, 'output/db_fisher_vectors')
PATH_GMM_DB = os.path.join(__location__, 'output/db_gmm.json')
PATH_OUTPUT = os.path.join(__location__, 'output/')
PATH_TEST_DATASET = os.path.join(__location__, 'data/test_retrieval')

# Destination folders
DESTINATION_PAINTINGS_DETECTED = os.path.join(__location__, 'output/paintings_detected/')
DESTINATION_PAINTINGS_RECTIFIED = os.path.join(__location__, 'output/paintings_rectified/')
DESTINATION_PEOPLE_DETECTED = os.path.join(__location__, 'output/person_detected/')
DESTINATION_PEOPLE_FACIAL = os.path.join(__location__, 'output/person_facial/')

# Yolo
PATH_YOLO_CFG = os.path.join(__location__, 'yolo/cfg/yolov3-obj-test.cfg')
# https://drive.google.com/file/d/1bkADs1lT8ayXwDwYnS1rU0Nz8rmTNmOt/view?usp=sharing
PATH_YOLO_WEIGHTS = os.path.join(__location__, 'yolo/yolov3-obj-train_final.weights')
PATH_COCO_NAMES = os.path.join(__location__, 'yolo/coco.names')

PATH_ORIGINAL = '/fhntjti/custom_data/obj/original/people/'
PATH_EDIT = '/homegng'

DILATE_KERNEL_SIZE = (5, 5)
DILATE_ITERATIONS = 2
EROSION_ITERATIONS = 3

# MULTI-SCALE RETINEX
SIGMA = [7, 40, 125]

# PAINTING DETECTION
KERNEL_3x3 = np.ones((3, 3), dtype=np.uint8)
NUM_ITERATIONS_DILATE = 1
NUM_ITERATIONS_ERODE = 3
SPATIAL_WINDOW_RADIUS = 16
COLOR_WINDOW_RADIUS = 16

# Rectification
ENTROPY_THRESHOLD = 2.0

# PAINTING RETRIEVAL
MAX_ITER = 3000
COVARIANCE_TYPE = 'diag'

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
