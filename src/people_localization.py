import cv2
import pandas as pd
import matplotlib.pyplot as plt
import sys

import numpy as np
import argparse, random

# Custom importing
from parameters import *
from read_write import read_video, read_bounding_boxes
from painting_retrieval import match_paitings


def people_localization(video_name):
    filename = video_name.split('/')[-1].split('.')[-2]
    bbox = read_bounding_boxes(filename)
    real_person = [True if 'real person' in classes_founded else False for classes_founded in bbox.values()]
    if not real_person:
        print('Real person not found. Return ...')
        return
    print('\t> Real person found')
    frames = read_video(video_name, reduce_size=False)
    if len(frames) != len(bbox.keys()):
        print("The video and bounding boxes don\'t have same size. Return!")
        return

    unique_paintings = dict()
    list_retrieval = dict()
    for frame, (num_frames, classes_founded) in zip(frames, bbox.items()):
        if 'real person' in classes_founded and 'painted person' in classes_founded:
            bounding_boxes_rp = classes_founded['real person']
            bounding_boxes_pp = classes_founded['painted person']
            for x, y, width, height in bounding_boxes_pp:
                unique_found = False
                x_center = int(x + width / 2)
                y_center = int(y + height / 2)
                for x, y in unique_paintings.keys():
                    distance = np.sqrt((x - x_center) ** 2 + (y - y_center) ** 2)
                    if distance < 100:
                        unique_found = True
                if not unique_found:
                    print('Found new unique detected painting ...')
                    unique_paintings[(x_center, y_center)] = (x, y, width, height)
                    result_retrieval_frame = match_paitings(frame[y: y + height, x: x + width, :])
                    for name_painting, sim in result_retrieval_frame:
                        if name_painting in list_retrieval:
                            list_retrieval[name_painting] += sim
                        else:
                            list_retrieval[name_painting] = sim

    total_retrieval = sorted(list_retrieval.items(), key=lambda x: x[1], reverse=True)
    best_locations = dict()
    data = pd.read_csv(PATH_DATA_CSV, sep=",")
    for image_name, sim in total_retrieval[:5]:
        curr_row = data[data["Image"] == image_name]
        room = curr_row["Room"].values[0]
        if room in best_locations:
            best_locations[room] += 1
        else:
            best_locations[room] = 1

    best_locations = sorted(best_locations.items(), key=lambda x: x[1], reverse=True)
    room, sim = best_locations[0]
    location = rooms_map_highlight(room, (0, 255, 0))
    plt.imshow(location)
    plt.show()


def rooms_map_highlight(room, color):
    rooms_roi = pd.read_csv(PATH_ROOMS_ROI_CSV)

    curr_roi = rooms_roi[rooms_roi["room"] == room]
    map = cv2.imread(PATH_MAP)

    cv2.rectangle(map, (curr_roi['x'], curr_roi['y']),
                  (curr_roi['x'] + curr_roi['w'], curr_roi['y'] + curr_roi['h']), (36, 255, 12), 3)
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
