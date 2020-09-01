import cv2
import matplotlib.pyplot as plt
import pandas as pd

import numpy as np
import argparse

# Custom importing
from parameters import *
from read_write import read_video, read_pickle_file
from painting_retrieval import match_paitings


def people_localization(video_name):
    filename = (os.path.split(video_name)[1]).split('.')[0]
    print(f"[INFO] Elaborate {filename}")

    bbox = read_pickle_file(filename, path=DESTINATION_PEOPLE_DETECTED)
    real_person = False
    for classes_found in bbox.values():
        if 'real person' in classes_found:
            real_person = True
    if not real_person:
        print('Real person not found. Return ...')
        return
    print('[INFO] Real person found')
    frames = read_video(video_name)

    # if len(frames) != len(bbox.keys()):
    #     print("The video and bounding boxes don\'t have same size. Return!")
    #     return

    unique_paintings = dict()
    list_retrieval = dict()
    for frame, (num_frames, classes_founded) in zip(frames, bbox.items()):
        if 'painted person' in classes_founded:
            bounding_boxes_pp = classes_founded['painted person']
            for x, y, width, height in bounding_boxes_pp:
                unique_found = False
                x_center = int(x + width / 2)
                y_center = int(y + height / 2)
                for x_up, y_up in unique_paintings.keys():
                    distance = np.sqrt((x_up - x_center) ** 2 + (y_up - y_center) ** 2)
                    if distance < 100:
                        unique_found = True
                if not unique_found:
                    print('[INFO] Found new unique detected painting. ')
                    unique_paintings[(x_center, y_center)] = (x, y, width, height)
                    result_retrieval_frame = match_paitings(frame[y: y + height, x: x + width, :],
                                                            folder_database=SOURCE_PAINTINGS_DB)
                    for name_painting, sim in result_retrieval_frame:
                        if name_painting in list_retrieval:
                            list_retrieval[name_painting] += sim
                        else:
                            list_retrieval[name_painting] = sim

    total_retrieval = sorted(list_retrieval.items(), key=lambda x: x[1], reverse=True)
    print(total_retrieval)
    if len(total_retrieval) >= 3:
        print(f"Name painting founded: {total_retrieval[0][0]} - {total_retrieval[1][0]} - {total_retrieval[2][0]}")
    best_locations = dict()
    data = pd.read_csv(PATH_DATA_CSV, sep=',')
    for image_name, sim in total_retrieval[:3]:
        curr_row = data[data["Image"] == image_name]
        room = curr_row["Room"].values[0]
        if room in best_locations:
            best_locations[room] += 1
        else:
            best_locations[room] = 1

    best_locations = sorted(best_locations.items(), key=lambda x: x[1], reverse=True)
    if len(best_locations) != 0:
        room, sim = best_locations[0]
        location = rooms_map_highlight(room, color=COLOR_GREEN)
        plt.imshow(location)
        plt.show()
    else:
        print("Room not found!")


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
    p1 = (curr_roi['x'], curr_roi['y'])
    p2 = (curr_roi['x'] + curr_roi['w'], curr_roi['y'] + curr_roi['h'])
    cv2.rectangle(map, pt1=p1, pt2=p2, color=color, thickness=3)
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


def get_args():
    parser = argparse.ArgumentParser('Test your image or video by trained model.')
    parser.add_argument('--video', type=str, default="../output/person_detected/VIRB0397.avi",
                        help='Path of cfg file', dest='video')
    return parser.parse_args()


args = get_args()
if args.video:
    people_localization(args.video)
