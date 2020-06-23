import cv2
import pandas as pd

# Custom importing
from parameters import *


def room_dict(image_name):
    if image_name is None:
        return

    data = pd.read_csv(PATH_DATA_CSV, sep=",")
    curr_row = data[data["Image"] == image_name]

    room = curr_row["Room"].values[0]

    room_paintings = data[data["Room"] == room]["Image"].values

    location = rooms_map_highlight(room=room, color=(0, 255, 0))
    return location


def rooms_map_highlight(room, color):
    rooms_roi = pd.read_csv(PATH_ROOMS_ROI_CSV)

    curr_roi = rooms_roi[rooms_roi["room"] == room]
    map = cv2.imread(PATH_MAP)

    cv2.rectangle(map, (curr_roi['x'], curr_roi['y']),
                  (curr_roi['x']+curr_roi['w'], curr_roi['y']+curr_roi['h']), (36, 255, 12), 3)
    return map


def roi_labeling(id, image, coordinate, image_name=None):
    upper_left, upper_right, down_left, down_right = coordinate
    x_min = min(upper_left[0], upper_right[0], down_left[0], down_right[0])
    y_min = min(upper_left[1], upper_right[1], down_left[1], down_right[1])
    x_max = max(upper_left[0], upper_right[0], down_left[0], down_right[0])
    y_max = max(upper_left[1], upper_right[1], down_left[1], down_right[1])
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (36, 255, 12), 3)

    data = pd.read_csv(PATH_DATA_CSV, sep=",")
    if image_name is None:
        image_name = "Unknown"
        cv2.putText(image, 'ID: ' + str(id) + ' - Unknown', (x_min, y_max + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (36, 255, 12), 2)
    else:
        curr_row = data[data["Image"] == image_name]
        title = curr_row["Title"].values[0]
        cv2.putText(image, 'ID: ' + str(id) + ' - Name: ' + image_name, (x_min, y_max + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(image, 'Title: ' + title, (x_min, y_max + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)

    return image
