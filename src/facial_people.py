import cv2
import sys
import skvideo.io

import numpy as np
import argparse

# Custom importing
from read_write import read_video, read_pickle_file


def get_args():
    parser = argparse.ArgumentParser('Test your image or video by trained model.')
    parser.add_argument('-video', type=str, default="../data/videos/002/20180206_114604.mp4",
                        help='Path of cfg file', dest='video')
    args = parser.parse_args()
    return args


args = get_args()
video_name = args.video

if not args.video:
    sys.exit("[ERROR]")

filename = video_name.split('/')[-1].split('.')[-2]
bbox = read_pickle_file(filename)
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

writer = skvideo.io.FFmpegWriter('../output/' + video_name.split('/')[-1][:-4] + '.avi')
print('Start elaboration ...')

for frame, (num_frames, classes_founded) in zip(frames, bbox.items()):
    if 'real person' in classes_founded and 'painted person' in classes_founded:
        bounding_boxes_rp = classes_founded['real person']
        bounding_boxes_pp = classes_founded['painted person']
        for x_rp, y_rp, w_rp, h_rp in bounding_boxes_rp:
            rect_person = np.zeros((frame.shape[:-1]))
            cv2.rectangle(img=rect_person, pt1=(x_rp, y_rp), pt2=(x_rp + w_rp, y_rp + h_rp),
                          color=(255, 255, 255), thickness=cv2.FILLED)
            text = 'Person not facial'
            # rect_person = frame[y_rp: y_rp + h_rp, x_rp: x_rp + w_rp, :]
            for x_pp, y_pp, w_pp, h_pp in bounding_boxes_pp:
                rect_painting = np.zeros((frame.shape[:-1]))
                cv2.rectangle(img=rect_painting, pt1=(x_pp, y_pp), pt2=(x_pp + w_pp, y_pp + h_pp),
                              color=(255, 255, 255), thickness=cv2.FILLED)
                rect_intersection = np.multiply(rect_person, rect_painting)
                area_intersection = np.sum(rect_intersection)
                if area_intersection > 0:
                    # print(area_intersection)
                    # plt.imshow(rect_intersection, cmap='gray')
                    # plt.show()
                    text = 'Person facial'
            cv2.rectangle(img=frame, pt1=(x_rp, y_rp), pt2=(x_rp + w_rp, y_rp + h_rp),
                          color=(0, 255, 0), thickness=3)
            cv2.putText(img=frame, text=text, org=(x_rp, y_rp),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.75, color=(0, 255, 0),
                        thickness=3)
    writer.writeFrame(frame)

writer.close()
print("End.")
