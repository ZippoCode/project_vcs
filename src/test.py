import os
import random
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Custom importing
from read_write import get_videos, write_video, read_video, save_paitings
from detection import elaborate_edge_detection
from plotting import draw_paintings, plt_images
from rectification import rectification, is_painting
from painting_retrieval import match_paitings
from parameters import *
from people_localization import match_keypoint, draw_matches


def paiting_detection(num_example=1):
    """
        Execute the task one and two of project.

        :param num_example: a number of example which you want to do. All if it is negative number
    :return:
    """
    path_videos = get_videos()
    path_videos = random.choices(
        path_videos, k=num_example if num_example > 0 else len(path_videos))
    #path_videos = ['../data/videos/003/GOPR1923.MP4']

    while len(path_videos) > 0:
        path_video = random.choice(path_videos)
        path_videos.remove(path_video)

        video_results = list()
        frames = read_video(path_video)
        # Choice a TOT frame because it is too slow
        frames = random.choices(frames, k=1)

        for frame in frames:
            list_boundings = elaborate_edge_detection(frame, show_images=True)
            good_boundings = list()

            paintings = []
            titles = []
            detected_paiting = dict()
            for bounding, num in zip(list_boundings, range(len(list_boundings))):
                # Affine Transformation for painting
                painting = rectification(frame, bounding)
                if is_painting(painting):
                    paintings.append(painting)
                    titles.append("Painting #{}".format(num))
                    painting = cv2.cvtColor(painting, cv2.COLOR_BGR2RGB)
                    detected_paiting["Painting #{}".format(num)] = painting
                    good_boundings.append(bounding)

            result = draw_paintings(frame, good_boundings)
            video_results.append(result)
            titles.append("Detection Frame")
            paintings.append(result)

            save_paitings(detected_paiting, path_video, folders=False)
            #plt_images(paintings, titles)

        file_name = path_video.split('/')[-1]
        path_output_video = '../output/videos/{}'
        write_video(path_output_video.format(file_name), video_results)
    return


def painting_retrieval(num_example=1):
    path_paitings = [file for file in os.listdir(
        ROOT_PATH_DETECTED) if file.endswith('.jpg')]
    paiting_choices = random.choices(
        path_paitings, k=num_example if num_example > 0 else len(path_paitings))
    for name_paiting in paiting_choices:
        painting = cv2.imread(ROOT_PATH_DETECTED +
                              name_paiting, cv2.IMREAD_COLOR)

        list_retrieval = match_paitings(painting)
        if list_retrieval is not None and len(list_retrieval) > 0:
            best_match, similarity = list_retrieval[0]
            retrieval = cv2.imread(PATH + best_match, cv2.IMREAD_COLOR)
            HP, WP, CP = painting.shape
            HB, WB, CB = retrieval.shape
            result = np.empty((max(HP, HB), WP + WB, CP), np.uint8)
            result[:HP, :WP, :] = painting
            result[:HB, WP:WP + WB, :] = retrieval
            result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            plt.imshow(result)
            plt.show()
        else:
            print("Nothing match found")


def localization(num_example=1):
    img1 = cv2.imread(
        '../data/video_painting/20180206_113600_11.jpg', cv2.IMREAD_COLOR)

    list_boundings = elaborate_edge_detection(img1, show_images=False)
    paintings = []
    titles = []
    detected_paiting = dict()
    for bounding, num in zip(list_boundings, range(len(list_boundings))):
                # Affine Transformation for painting
        painting = rectification(img1, bounding)
        if is_painting(painting):
            paintings.append(painting)
            titles.append("Painting #{}".format(num))
            result = match_keypoint()

            plt.imshow(result)
            plt.show()

    # list_boundings = elaborate_edge_detection(frame, show_images=False)
    # # good_boundings = list()

    # paintings = []
    # titles = []
    # detected_paiting = dict()
    # for bounding, num in zip(list_boundings, range(len(list_boundings))):
    #             # Affine Transformation for painting
    #     painting = rectification(frame, bounding)
    #     if is_painting(painting):
    #         paintings.append(painting)
    #         titles.append("Painting #{}".format(num))
    #         painting_gray = cv2.cvtColor(painting, cv2.COLOR_BGR2GRAY)
    #         detected_paiting["Painting #{}".format(num)] = painting
    #         # good_boundings.append(bounding)
    #         list_retrieval = match_paitings(painting_gray)
    #         if list_retrieval is not None and len(list_retrieval) > 0:
    #             best_match, similarity = list_retrieval[0]
    #             print('best_match', best_match)
    #             retrieval = cv2.imread(
    #                 PATH + best_match, cv2.IMREAD_COLOR)
    #             HP, WP, CP = painting.shape
    #             HB, WB, CB = retrieval.shape
    #             result = np.empty((max(HP, HB), WP + WB, CP), np.uint8)
    #             result[:HP, :WP, :] = painting
    #             result[:HB, WP:WP + WB, :] = retrieval
    #             result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    #             plt.imshow(result)
    #             plt.show()
    #         else:
    #             print("Nothing match found")
    # # return


if __name__ == '__main__':
    # paiting_detection(num_example=3)
    # painting_retrieval(num_example=1)
    localization(num_example=1)
