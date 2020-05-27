import os
import random
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Custom importing
from read_write import read_video, write_video
from detection import edge_detection, get_bounding_boxes, elaborate_edge_detection
from plotting import plt_images, draw_paintings
from rectification import rectification
from paiting_retrieval import match_paitings

from parameters import *

if __name__ == '__main__':
    #  read_single_image('../data/test.jpg')
    video_list = []
    root_path = '../data/videos'
    path_video = '../output/videos/{}'

    for forder in os.listdir(root_path):
        if forder == '.DS_Store':
            continue
        path = os.path.join(root_path, forder)
        for file in os.listdir(path):
            if file.endswith(".mp4") or file.endswith(".MP4") or file.endswith(".MOV"):
                video = os.path.join(path, file)
                video_list.append(video)

    # video_list = []
    # Take a random video and remove it from list
    # video_list = []
    # video_list.append('../data/videos/002/20180206_113059.mp4')
    # capVideo('../data/videos/001/GOPR5832.MP4',
    #          'GOPR5832.MP4')
    video_list = ['../data/videos/001/GOPR5828.MP4']
    # capVideo('../data/videos/010/VID_20180529_112440.mp4',
    #  'VID_20180529_112440.mp4')
    while len(video_list) > 0:
        path_painting = "../output/paintings/{}_#{}.jpg"
        video = random.choice(video_list)
        video_list.remove(video)
        file_name = video.split('/')[-1]

        # Edge Detection
        video_results = list()
        frames = read_video(video, file_name)

        # Utils for choice a one frame because it is too slow
        frames = random.choices(frames, k=8)

        for frame in frames:
            list_boundings = elaborate_edge_detection(frame, show_images=False)
            result = draw_paintings(frame, list_boundings)
            video_results.append(result)

            paintings = []
            titles = []
            paintings.append(result)
            titles.append("Detection Frame")
            for bounding, num in zip(list_boundings, range(len(list_boundings))):
                # Affine Transformation for painting
                painting = rectification(frame, bounding)
                paintings.append(painting)
                titles.append("Painting #ID: {}".format(num))
                # cv2.imwrite(path_painting.format(file_name, num),
                #             cv2.cvtColor(painting, cv2.COLOR_BGR2RGB))
            plt_images(paintings, titles)

        # write_video(path_video.format(file_name), video_results)
    # # Paiting Retrival
    # path = '../output/paintings/IMG_4083_#3.jpg'
    # painting = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    # list_retrieval = match_paitings(painting)
    # if list_retrieval is not None and len(list_retrieval) > 0:
    #     best_match, similarity = list_retrieval[0]
    #     retrieval = cv2.imread(PATH + best_match, cv2.IMREAD_GRAYSCALE)
    #     # TO DO: Improve Visualization
    #     HP, WP = painting.shape
    #     HB, WB = retrieval.shape
    #     result = np.empty((max(HP, HB), WP + WB), np.uint8)
    #     result[:HP, :WP] = painting
    #     result[:HB, WP:WP + WB] = retrieval
    #     plt.imshow(result, cmap='gray')
    #     plt.show()
    # else:
    #     print("Nothing match found")
