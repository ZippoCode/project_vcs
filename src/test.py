import os
import random
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Custom importing
from read_write import capVideo, read_single_image
from paiting_retrieval import match_paitings
from parameters import *

# CONSTANTS

if __name__ == '__main__':
    #  read_single_image('../data/test.jpg')
    video_list = []
    root_path = '../data/videos'
    for forder in os.listdir(root_path):
        if forder == '.DS_Store':
            continue
        path = os.path.join(root_path, forder)
        for file in os.listdir(path):
            if file.endswith(".mp4") or file.endswith(".MP4") or file.endswith(".MOV"):
                video = os.path.join(path, file)
                video_list.append(video)

    #   video_list = []
    # Take a random video and remove it from list
    while len(video_list) > 0:
        video = random.choice(video_list)
        video_list.remove(video)
        file_name = video.split('/')[-1]
        capVideo(video, file_name)

    # Paiting Retrival
    path = '../output/paintings/IMG_4083_#3.jpg'
    painting = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    list_retrieval = match_paitings(painting)
    if list_retrieval is not None and len(list_retrieval) > 0:
        best_match, similarity = list_retrieval[0]
        retrieval = cv2.imread(PATH + best_match, cv2.IMREAD_GRAYSCALE)
        # TO DO: Improve Visualization
        HP, WP = painting.shape
        HB, WB = retrieval.shape
        result = np.empty((max(HP, HB), WP + WB), np.uint8)
        result[:HP, :WP] = painting
        result[:HB, WP:WP + WB] = retrieval
        plt.imshow(result, cmap='gray')
        plt.show()
    else:
        print("Nothing match found")
