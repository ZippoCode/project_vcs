import os
import random
import cv2
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path

# Custom importing
from read_write import read_video, write_video
from detection import elaborate_edge_detection
from plotting import draw_paintings
from rectification import rectification
from paiting_retrieval import match_paitings
from plotting import plt_images

from parameters import *


def paiting_detection(num_example=1):
    """
        Execute the task one and two of project.

        :param num_example: a number of example which you want to do. All if it is negative number
    :return:
    """
    path_videos = list()
    ROOT_PATH = '../data/videos/'
    for folder in os.listdir(ROOT_PATH):
        if folder == '.DS_Store':
            continue
        path = os.path.join(ROOT_PATH, folder)
        for file in os.listdir(path):
            if file.endswith(".mp4") or file.endswith(".MP4") or file.endswith(".MOV"):
                video = os.path.join(path, file)
                path_videos.append(video)

    path_videos = random.choices(
        path_videos, k=num_example if num_example > 0 else len(path_videos))
    # path_videos = ['../data/videos/010/VID_20180529_112722.mp4']

    while len(path_videos) > 0:
        path_video = random.choice(path_videos)
        path_videos.remove(path_video)

        #   folder = path_video.split('/')[-2]
        file_name = path_video.split('/')[-1]
        # path_output_paitings = "../output/paintings/{}/{}/".format(folder, file_name)
        path_output_paitings = "../output/paintings/{}_#{}.jpg"
        path_output_video = '../output/videos/{}'

        # If you want to save in specific folder
        # if not os.path.exists(path_output_paitings):
        #     os.makedirs(path_output_paitings)

        video_results = list()
        frames = read_video(path_video, file_name)

        # Utils for choice a TOT frame because it is too slow
        frames = random.choices(frames, k=10)

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
                painting = cv2.cvtColor(painting, cv2.COLOR_BGR2RGB)
                # output_name = path_output_paitings + "Painting #{}.jpg".format(num)
                output_name = path_output_paitings.format(file_name, num)
                cv2.imwrite(output_name, painting)
            plt_images(paintings, titles)

        write_video(path_output_video.format(file_name), video_results)
    return


def painting_retrieval(num_example=1):
    ROOT_PATH = "../output/paintings/"
    path_paitings = [file for file in os.listdir(
        ROOT_PATH) if file.endswith('.jpg')]

    paiting_choices = random.choices(path_paitings, k=num_example)
    for name_paiting in paiting_choices:
        painting = cv2.imread(ROOT_PATH + name_paiting, cv2.IMREAD_GRAYSCALE)
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


if __name__ == '__main__':
    paiting_detection(num_example=3)
    # painting_retrieval(num_example=4)
