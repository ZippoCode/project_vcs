from distutils.command.upload import upload

import cv2
import numpy as np

import matplotlib.pyplot as plt
from setuptools.command.upload import upload


def convert_bounding_boxes(list_bounding_boxes):
    """
        Give a list of bounding boxes in format point (upper_left, upper_right, down_left, down_right)
        return a list of bounding boxes in format (X, Y, W, H ) where (x,y) is a coordinate of upper left point,
        W is the weight and  H is the height of rectangle

    :param list_bounding_boxes: list
    :return:
    """
    bounding_boxes = []
    for upper_left, upper_right, down_left, down_right in list_bounding_boxes:
        x, y = upper_left
        x1, y1 = down_right
        bounding_boxes.append((x, y, (x1 - x), (y1, y)))
        print(upper_left, upper_right, down_left, down_right)
        print((x, y, (x1 - x), (y1 - y)))
    return bounding_boxes


def sorted_points(contour):
    """
        Given a contour with shape (4, 1, 2) and return the sorted points
        Upper Left, Upper Right, Down Left, Down Right.

    :param contour:
    :return:
    """
    middle_x, middle_y = 0, 0
    upper_left, upper_right, down_left, down_right = (0, 0), (0, 0), (0, 0), (0, 0)
    for point in range(contour.shape[0]):
        middle_x += contour[point, 0, 1]
        middle_y += contour[point, 0, 0]
    middle_x /= 4
    middle_y /= 4
    for point in range(contour.shape[0]):
        if contour[point, 0, 1] < middle_x and contour[point, 0, 0] < middle_y:
            upper_left = (contour[point, 0, 0], contour[point, 0, 1])
        elif contour[point, 0, 1] < middle_x and contour[point, 0, 0] > middle_y:
            upper_right = (contour[point, 0, 0], contour[point, 0, 1])
        elif contour[point, 0, 1] > middle_x and contour[point, 0, 0] < middle_y:
            down_left = (contour[point, 0, 0], contour[point, 0, 1])
        elif contour[point, 0, 1] > middle_x and contour[point, 0, 0] > middle_y:
            down_right = (contour[point, 0, 0], contour[point, 0, 1])
        else:
            return
    return upper_left, upper_right, down_left, down_right


def get_bounding_boxes(original_image, edge_detection_image):
    """
        Given an image it looks for the paintings and returns a list of bounding boxes

    :param image:
    :return: list of bounding boxes (x, y, w, h)
    """

    contours, hierarchy = cv2.findContours(edge_detection_image, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return []

    list_bounding_boxes = []
    threshold = sum([cv2.contourArea(c) for c in contours]) / len(contours)

    # Find rectangle
    used_contours = []
    for precision in np.arange(0.01, 0.1, 0.02):
        for index, contour in enumerate(contours):
            epsilon = cv2.arcLength(contour, True) * precision
            approx = cv2.approxPolyDP(contour, epsilon=epsilon, closed=True)
            # cv2.drawContours(original_image, [approx], -1, (255, 0, 0), 3)
            if len(approx) == 4 and cv2.contourArea(approx) > threshold and index is not used_contours:
                used_contours.append(contour)
                sorted_approx = sorted_points(approx)
                if sorted_approx is not None and not (0, 0) in sorted_approx:
                    list_bounding_boxes.append(sorted_approx)

        # elif len(approx) >= 10 and cv2.contourArea(approx) > threshold:
        #     x, y, w, h = cv2.boundingRect(approx)
        #     list_bounding_boxes.append(((x, y), (x + w, y), (x, y + h), (x + w, y + h)))
    # plt.imshow(edge_detection_image)
    # plt.show()
    return list_bounding_boxes
