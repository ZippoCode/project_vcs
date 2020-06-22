import sys
import cv2
import numpy as np
import pandas as pd
import os

# Custom importing
from parameters import *


def room_dict(img_name):
    if img_name is None:
        return

    data = pd.read_csv(PATH_DATA_CSV, sep=",")
    header = data.columns.values
    curr_row = data[data["Image"] == img_name]

    room = curr_row["Room"].values[0]

    room_paintings = data[data["Room"] == room]["Image"].values

    location = rooms_map_highlight(room=room, color=(0, 255, 0))
    return location


def rooms_map_highlight(room, color):
    rooms_roi = pd.read_csv(PATH_ROOMS_ROI_CSV)
    header = rooms_roi.columns.values

    curr_roi = rooms_roi[rooms_roi["room"] == room]
    map = cv2.imread(PATH_MAP)

    cv2.rectangle(map, (curr_roi['x'], curr_roi['y']),
                  (curr_roi['x']+curr_roi['w'], curr_roi['y']+curr_roi['h']), color, 3)

    return map
