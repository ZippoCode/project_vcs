import random
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Custom importing
from painting_rectification import rectification, is_painting
from painting_retrieval import match_paitings
from parameters import *
from people_localization import room_dict, roi_labeling



def painting_retrieval(num_example=1):
    path_paitings = [file for file in os.listdir(ROOT_PATH_DETECTED) if file.endswith('.jpg')]
    paiting_choices = random.choices(path_paitings, k=num_example if num_example > 0 else len(path_paitings))
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


# def localization(image_name):
#     frame = cv2.imread(image_name, cv2.IMREAD_COLOR)
#     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     list_boundings = elaborate_edge_detection(frame, show_images=True)
#
#     paintings = []
#     titles = []
#     detected_paiting = dict()
#     for bounding, num in zip(list_boundings, range(len(list_boundings))):
#         # Affine Transformation for painting
#         painting = rectification(frame, bounding)
#         if is_painting(painting):
#             painting_gray = cv2.cvtColor(painting, cv2.COLOR_BGR2GRAY)
#             list_retrieval = match_paitings(painting_gray)
#             if list_retrieval is not None and len(list_retrieval) > 0:
#                 best_match, similarity = list_retrieval[0]
#                 if similarity < 11:
#                     frame = roi_labeling(num, frame, bounding)
#                     print("Nothing match found")
#                 else:
#                     location = room_dict(best_match)
#                     frame = roi_labeling(num, frame, bounding, best_match)
#                     plt.imshow(location)
#                     plt.show()
#             else:
#                 print("Nothing match found")
#     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     plt.imshow(frame)
#     plt.show()
#
#     return


if __name__ == '__main__':
    painting_retrieval(num_example=3)
    # localization('../data/video_painting/20180206_113800_34.jpg')
