import numpy as np
import cv2
import math


def check_same_bounding_boxes(first_box, second_box):
    ul1, ur1, dl1, dr1 = first_box
    ul2, ur2, dl2, dr2 = second_box
    x1, y1, w1, r1 = ul1[0], ur1[0], (dr1[0] - ul1[0]), (dr1[1] - ul1[1])
    x2, y2, w2, r2 = ul2[0], ur2[0], (dr2[0] - ul2[0]), (dr2[1] - ul2[1])
    if math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2) > 100:
        return False
    x1_centre, y1_centre = round((ul1[0] + dr1[0]) / 2), round((ul1[1] + dr1[1]) / 2)
    x2_centre, y2_centre = round((ul2[0] + dr2[0]) / 2), round((ul2[1] + dr2[1]) / 2)
    if math.sqrt((x1_centre - x2_centre) ** 2 + (y1_centre - y2_centre) ** 2) > 100:
        return False
    # print(f"{first_box} and {second_box} are the same boxes")
    return True


def search_best_area(first_box, second_box):
    """
    Given two bounding boxes return which has a

    :param first_box:
    :param second_box:
    :return:
    """
    ul1, ur1, dl1, dr1 = first_box
    ul2, ur2, dl2, dr2 = second_box
    x1, y1, w1, h1 = ul1[0], ur1[0], (dr1[0] - ul1[0]), (dr1[1] - ul1[1])
    x2, y2, w2, h2 = ul2[0], ur2[0], (dr2[0] - ul2[0]), (dr2[1] - ul2[1])
    area_one = 2 * (w1 + h1)
    area_two = 2 * (w2 + h2)
    if area_one > area_two:
        return first_box
    return second_box


def search_unique_bounding_boxes(current_bounding_boxes, previous_bounding_boxed):
    """
        Return a list of unique bounding boxes given to match between a current bounding boxes and
        a list which will be for searching.

    :param current_bounding_boxes:
    :param list_bounding_boxes:
    :return:
    """
    if len(previous_bounding_boxed) == 0:
        return current_bounding_boxes
    best_bounding_boxes = []
    for search_box in current_bounding_boxes:
        for current_box in previous_bounding_boxed:
            if check_same_bounding_boxes(current_box, search_box):
                # Same Box. Add the best of these two
                best = search_best_area(search_box, current_box)
                return best_bounding_boxes.append(best)
            else:
                # This bounding boxes is not present in the previous frame
                best_bounding_boxes.append(search_box)

    return best_bounding_boxes


def rectification(im, coordinate):
    upper_left, upper_right, down_left, down_right = coordinate

    rows = int(math.sqrt(math.pow((down_left[0] - upper_left[0]), 2) + math.pow((down_left[1] - upper_left[1]), 2)))
    cols = int(math.sqrt(math.pow((upper_right[0] - upper_left[0]), 2) + math.pow((upper_right[1] - upper_left[1]), 2)))
    src_points = np.float32([upper_left, down_left, upper_right, down_right])
    dst_points = np.float32([[0, 0], [0, rows - 1], [cols - 1, 0], [cols - 1, rows - 1]])
    projective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    img_output = cv2.warpPerspective(im, projective_matrix, (cols, rows))

    return img_output
