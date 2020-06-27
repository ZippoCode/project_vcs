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
from people_localization import room_dict, roi_labeling, people_localization


def paiting_detection(num_example=1):
    """
        Execute the task one and two of project.

        :param num_example: a number of example which you want to do. All if it is negative number
    :return:
    """
    path_videos = get_videos()
    path_videos = random.choices(path_videos, k=num_example if num_example > 0 else len(path_videos))
    path_videos = ['../data/videos/006/IMG_9622.MOV']

    while len(path_videos) > 0:
        path_video = random.choice(path_videos)
        path_videos.remove(path_video)

        video_results = list()
        frames = read_video(path_video)
        try:
<<<<<<< HEAD
            for num_frame, frame in enumarate(frames):
                frame = cv2.imread('../data/video_painting/20180206_113800_34.jpg', cv2.IMREAD_COLOR)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
=======
            for num_frame, frame in enumerate(frames):
>>>>>>> 2f17b00ebd4ead94678850d33ab17f81bba0c32e
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
                video_results.append(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
                save_paitings(detected_paiting, path_video, folders=True)
                print("[INFO] Elaborate {} of {} frames".format(num_frame + 1, len(frames)))

        except KeyboardInterrupt:
            print('Stop processing')
            pass

        file_name = path_video.split('/')[-1]
        file_name = file_name.split('.')[0]
        write_video(file_name + '.avi', video_results, path=PATH_DESTINATION_PAINTING_DETECTED)
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


def localization(num_example=1):
    frame = cv2.imread(
        # '../data/video_painting/20180206_113800_34.jpg', cv2.IMREAD_COLOR)
        '../data/video_painting/00frame.jpg', cv2.IMREAD_COLOR)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    list_boundings = elaborate_edge_detection(frame, show_images=True)

    paintings = []
    titles = []
    detected_paiting = dict()
    for bounding, num in zip(list_boundings, range(len(list_boundings))):
        # Affine Transformation for painting
        painting = rectification(frame, bounding)
        if is_painting(painting):
            painting_gray = cv2.cvtColor(painting, cv2.COLOR_BGR2GRAY)
            list_retrieval = match_paitings(painting_gray)
            if list_retrieval is not None and len(list_retrieval) > 0:
                best_match, similarity = list_retrieval[0]
                if similarity < 11:
                    frame = roi_labeling(num, frame, bounding)
                    print("Nothing match found")
                else:
                    location = room_dict(best_match)
                    frame = roi_labeling(num, frame, bounding, best_match)
                    plt.imshow(location)
                    plt.show()
            else:
                print("Nothing match found")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    plt.imshow(frame)
    plt.show()

    return


if __name__ == '__main__':
    paiting_detection(num_example=1)
    # painting_retrieval(num_example=1)
<<<<<<< HEAD
    #   localization(num_example=1)
    people_localization('/data/videos/002/20180206_113059.mp4')
=======
    # localization(num_example=1)
    # people_localization('../output/person_detected/20180206_114306.avi')
    # people_localization("../output/person_detected/GOPR1940.avi")
>>>>>>> 2f17b00ebd4ead94678850d33ab17f81bba0c32e
