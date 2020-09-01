import cv2
import sys
import os

import numpy as np
import argparse

# Custom importing
from read_write import read_video, read_pickle_file, store_video
from parameters import *


def get_args():
    parser = argparse.ArgumentParser('Test your image or video by trained model.')
    parser.add_argument('-video', type=str, default="../output/person_detected/VIRB0399.avi",
                        help='Path of cfg file', dest='video')
    return parser.parse_args()


args = get_args()
video_name = args.video

if not args.video:
    sys.exit("[ERROR]")

filename = os.path.split(video_name)[1].split('.')[0]
bbox = read_pickle_file(filename, path=DESTINATION_PEOPLE_DETECTED)
real_person = False
for classes_found in bbox.values():
    if 'real person' in classes_found:
        real_person = True
if not real_person:
    print('Real person not found. Return ...')
    sys.exit()
print('[INFO] Real person found into video')
frames = read_video(video_name)

if not frames:
    sys.exit("[ERROR] Frame not found!")

frames_result = []
for frame, (num_frames, classes_founded) in zip(frames, bbox.items()):
    if 'real person' in classes_founded:
        bounding_boxes_rp = classes_founded['real person']
        if 'painted person' in classes_founded:
            bounding_boxes_pp = classes_founded['painted person']
            for x_rp, y_rp, w_rp, h_rp in bounding_boxes_rp:
                rect_person = np.zeros((frame.shape[:-1]))
                cv2.rectangle(img=rect_person, pt1=(x_rp, y_rp), pt2=(x_rp + w_rp, y_rp + h_rp),
                              color=(255, 255, 255), thickness=cv2.FILLED)
                text = 'Person not facial'
                color = COLOR_GREEN
                # rect_person = frame[y_rp: y_rp + h_rp, x_rp: x_rp + w_rp, :]
                for x_pp, y_pp, w_pp, h_pp in bounding_boxes_pp:
                    rect_painting = np.zeros((frame.shape[:-1]))
                    cv2.rectangle(img=rect_painting, pt1=(x_pp, y_pp), pt2=(x_pp + w_pp, y_pp + h_pp),
                                  color=(255, 255, 255), thickness=cv2.FILLED)
                    rect_intersection = np.multiply(rect_person, rect_painting)
                    area_intersection = np.sum(rect_intersection)
                    if area_intersection > 0:
                        color = COLOR_RED
                cv2.rectangle(img=frame, pt1=(x_rp, y_rp), pt2=(x_rp + w_rp, y_rp + h_rp),
                              color=color, thickness=3)
        else:
            for x_rp, y_rp, w_rp, h_rp in bounding_boxes_rp:
                rect_person = np.zeros((frame.shape[:-1]))
                cv2.rectangle(img=rect_person, pt1=(x_rp, y_rp), pt2=(x_rp + w_rp, y_rp + h_rp),
                              color=COLOR_GREEN, thickness=cv2.FILLED)
    frames_result.append(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    print(f"Elaborate frame.")

store_video(name=filename + '.avi', frames=frames_result, path=DESTINATION_PEOPLE_FACIAL)
print("End.")
