import os
import random
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Custom importing
from read_write import get_videos, write_video, read_video, save_paitings
from painting_detection import elaborate_edge_detection
from plotting import draw_paintings, plt_images
from painting_rectification import rectification, is_painting
from painting_retrieval import match_paitings
from parameters import *
from people_localization import people_localization


def paiting_detection(num_example=1):
    """
        Execute the task one and two of project.

        :param num_example: a number of example which you want to do. All if it is negative number
    :return:
    """
    path_videos = get_videos()
    path_videos = random.choices(path_videos, k=num_example if num_example > 0 else len(path_videos))
    # path_videos = ['../data/videos/003/GOPR1923.MP4']

    while len(path_videos) > 0:
        path_video = random.choice(path_videos)
        path_videos.remove(path_video)

        video_results = list()
        frames = read_video(path_video)
        try:
            for num_frame, frame in enumerate(frames):
                # frame = cv2.imread('../data/video_painting/20180206_113800_34.jpg', cv2.IMREAD_COLOR)
                list_boundings = elaborate_edge_detection(frame, show_images=False)
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
                titles.append("Detection Frame")
                paintings.append(result)

                save_paitings(detected_paiting, path_video, folders=True)
                print("Elaborate {} of {} frames".format(num_frame, len(frames)))
                # plt_images(paintings, titles)

        except KeyboardInterrupt:
            print('Stop processing')
            pass

        file_name = path_video.split('/')[-1]
        write_video(file_name, video_results, path=PATH_DESTINATION_PAINTING_DETECTED)
    return


def painting_retrieval(num_example=1):
    path_paitings = [file for file in os.listdir(
        ROOT_PATH_DETECTED) if file.endswith('.jpg')]
    paiting_choices = random.choices(
        path_paitings, k=num_example if num_example > 0 else len(path_paitings))
    for name_paiting in paiting_choices:
        painting = cv2.imread(ROOT_PATH_DETECTED + name_paiting, cv2.IMREAD_COLOR)
        list_retrieval = match_paitings(painting)
        if list_retrieval is not None and len(list_retrieval) > 0:
            best_match, similarity = list_retrieval[0]
            retrieval = cv2.imread(PATH_PAINTINGS_DB + best_match, cv2.IMREAD_COLOR)
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


if __name__ == '__main__':
    # paiting_detection(num_example=1)
    # painting_retrieval(num_example=1)
    #   localization(num_example=1)
    people_localization('../output/person_detected/20180206_113059.avi')
    #people_localization("../output/person_detected/GOPR1940.avi")
