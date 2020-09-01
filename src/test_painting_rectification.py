import random
import argparse
import cv2
import os
import matplotlib.pyplot as plt

from read_write import read_video, save_paintings, read_pickle_file
from painting_rectification import rectification, search_unique_bounding_boxes
from parameters import DESTINATION_PAINTINGS_DETECTED, DESTINATION_PAINTINGS_RECTIFIED


def arg_parse():
    """
        Parse arguments to the Painting Detection

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", dest='num_example', help='The number of video which do you want detect',
                        default=-1, type=int)
    parser.add_argument("--show", dest='show_images', help='If True you can see the results of frame',
                        default=False, type=bool)
    parser.add_argument("--source", dest='source_folder',
                        help='The source folder of painting detected',
                        default=DESTINATION_PAINTINGS_DETECTED, type=str)
    parser.add_argument("--destination", dest='destination_folder',
                        help='The folder where the list of bounding boxes will be saved',
                        default=DESTINATION_PAINTINGS_RECTIFIED, type=str)
    return parser.parse_args()


args = arg_parse()
num_example = args.num_example
show_images = args.show_images
source_folder = args.source_folder
destination_folder = args.destination_folder

print("Start Processing Painting Rectification ...")

file_name = None
path_video = None
height, width = (0, 0)
total_frame = 0
bounding_boxes_dict = dict()

pickles = []
print(f"[INFO] Source folder is {source_folder}")
for root, _, file_names in os.walk(source_folder):
    for filename in file_names:
        if filename.lower().endswith('.pck'):
            pickles.append(os.path.join(root, filename))

pickles = random.choices(pickles, k=num_example if num_example > 0 else len(pickles))

print("[INFO] Number of video which will be elaborated: {}".format(len(pickles)))

try:
    while len(pickles) > 0:
        path_pickle_file = random.choice(pickles)
        pickles.remove(path_pickle_file)
        file_name = os.path.split(path_pickle_file)[1]
        file_name = file_name.split('.')[0]
        pickle_file = read_pickle_file(file_name, path=source_folder)
        if len(pickle_file.items()) == 0:
            continue
        if 'Name file' in pickle_file:
            print('[INFO] Elaborated file {}'.format(pickle_file['Name file']))
        if 'Path video' in pickle_file:
            path_video = pickle_file['Path video']
            print('[INFO] Path: {}'.format(path_video))
        if 'Total frame' in pickle_file:
            print('[INFO] Total Number frame: {}'.format(pickle_file['Total frame']))
        if 'Resolution frame' in pickle_file:
            height, width = pickle_file['Resolution frame']
            print('[INFO] Frame Resolution: {}'.format((height, width)))
        if 'Bounding boxes' in pickle_file:
            bounding_boxes_dict = pickle_file['Bounding boxes']
        frames = read_video(video_path=path_video)
        titles = []
        for frame, (num_frame, bounding_boxes_frame) in zip(frames, bounding_boxes_dict.items()):
            paintings_rectified = dict()
            for num, (upper_left, upper_right, down_left, down_right) in enumerate(bounding_boxes_frame):
                height_frame, width_frame = frame.shape[:2]
                scale = round(height_frame / height)
                upper_left = (upper_left[0] * scale, upper_left[1] * scale)
                upper_right = (upper_right[0] * scale, upper_right[1] * scale)
                down_left = (down_left[0] * scale, down_left[1] * scale)
                down_right = (down_right[0] * scale, down_right[1] * scale)
                area = (down_right[0] - upper_left[0]) * (down_right[1] - upper_left[1])
                painting = rectification(frame, (upper_left, upper_right, down_left, down_right))
                painting = cv2.cvtColor(painting, cv2.COLOR_BGR2RGB)
                name = "{}_frame{}_painting{}".format(file_name, num_frame, num)
                paintings_rectified[name] = painting
            save_paintings(paintings_rectified, folder=True, filename=file_name)

except KeyboardInterrupt:
    print('Stop processing')
    pass

print("End process.")
