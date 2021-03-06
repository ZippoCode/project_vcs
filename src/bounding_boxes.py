import cv2
import numpy as np
import math
import matplotlib.pyplot as plt


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
        bounding_boxes.append((x, y, (x1 - x), (y1 - y)))
    return bounding_boxes


def sorted_points(contour):
    """
        Given a contour with shape (4, 1, 2) and return the sorted points
        Upper Left, Upper Right, Down Left, Down Right.

    :param contour:
    :return:
    """
    if len(contour) == 8:
        points = contour[:, 0, :]
        upper_left = np.min(points[:, 0]), np.min(points[:, 1])
        upper_right = np.min(points[:, 0]), np.max(points[:, 1])
        down_left = np.max(points[:, 0]), np.min(points[:, 1])
        down_right = np.max(points[:, 0]), np.max(points[:, 1])
    else:
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
                return None
    return upper_left, upper_right, down_left, down_right


def check_bounding_boxes(list_bounding_boxes, bounding_boxes):
    """
        Check if a bounding boxes is contained into a list

    :param list_bounding_boxes:
    :param bounding_boxes:
    :return:
    """
    for upper_left, upper_right, down_left, down_right in list_bounding_boxes:
        ul, ur, dl, dr = bounding_boxes
        if upper_left[0] == ul[0] and upper_left[1] == ul[1]:
            return False
        if upper_right[0] == ur[0] and upper_right[1] == ur[1]:
            return False
        if down_left[0] == dl[0] and down_left[1] == dl[1]:
            return False
        if down_right[0] == dr[0] and down_right[1] == dr[1]:
            return False
    return True


def check_paintings(bounding_boxes, previous_rectangles, image):
    """
        Check if a rectangle of image is a painting

    :param bounding_boxes:
    :param previous_rectangles:
    :param image
    :return:
    """
    if bounding_boxes is None:
        return False
    if (0, 0) in bounding_boxes:
        return False
    if not check_bounding_boxes(previous_rectangles, bounding_boxes):
        return False

    # upper_left, upper_right, down_left, down_right = bounding_boxes
    # x, y = upper_left
    # x1, y1 = down_right
    # x, y, w, h = (x, y, (x1 - x), (y1 - y))

    # rgb_image = cv2.split(image[y:y + h, x:x + w, :])
    # intensity = image[y:y + h, x:x + w, :].sum(axis=2)
    # red_fraction = np.sum(image[y:y + h, x:x + w, 0]) / np.sum(intensity)
    # green_fraction = np.sum(image[y:y + h, x:x + w, 1]) / np.sum(intensity)
    # blue_fraction = np.sum(image[y:y + h, x:x + w, 2]) / np.sum(intensity)
    # print(red_fraction, green_fraction, blue_fraction)

    # r_hist = cv2.calcHist(rgb_image, [0], None, [256], (0, 256), accumulate=False)
    # g_hist = cv2.calcHist(rgb_image, [1], None, [256], (0, 256), accumulate=False)
    # b_hist = cv2.calcHist(rgb_image, [2], None, [256], (0, 256), accumulate=False)
    # count_red, count_green, count_green = r_hist.sum(), g_hist.sum(), b_hist.sum()
    # sum_red = np.sum(r_hist * np.c_[0:256])
    # sum_green = np.sum(g_hist * np.c_[0:256])
    # sum_blue = np.sum(b_hist * np.c_[0:256])
    # avg_red, avg_green, avg_blue = sum_red / count_red, sum_green / count_green, sum_blue / count_green
    # print(avg_red, avg_green, avg_blue)

    # plt.imshow(rgb_image[0], cmap='gray')
    # plt.show()
    # plt.imshow(rgb_image[1], cmap='gray')
    # plt.show()
    # plt.imshow(rgb_image[2], cmap='gray')
    # plt.show()
    return True


def find_paintings(contour):
    for precision in np.arange(0.03, 0.3, 0.02):
        epsilon = cv2.arcLength(contour, True) * precision
        approx = cv2.approxPolyDP(contour, epsilon=epsilon, closed=True)
        if len(approx) == 8:
            sorted_approx = sorted_points(approx)
            if sorted_approx is None:
                continue
            return sorted_approx
        if len(approx) == 4:
            sorted_approx = sorted_points(approx)
            if sorted_approx is None:
                continue
            return sorted_approx
    return None


def get_bounding_boxes(original_frame, edge_detection_image):
    """
        Given an image it looks for the paintings and returns a list of bounding boxes.
        Each Bounding Boxes contains four points: upper_left, upper_right, down_left, down_right

    :param original_frame: ndarray (H, W, C)
    :param edge_detection_image: ndarray (H, W)
    :return: list of bounding boxes
    """
    list_bounding_boxes = []
    padding_image = cv2.copyMakeBorder(edge_detection_image, 1, 1, 1, 1, cv2.BORDER_CONSTANT, None, 0)
    ret_val, labels, stats, centroids = cv2.connectedComponentsWithStatsWithAlgorithm(padding_image,
                                                                                      connectivity=8,
                                                                                      ltype=cv2.CV_16U,
                                                                                      ccltype=cv2.CCL_GRANA)
    # Remove more small
    num_paintings = 1
    for num_label in range(1, stats.shape[0]):
        if stats[num_label, cv2.CC_STAT_AREA] < np.mean(stats[1:, cv2.CC_STAT_AREA]):
            labels[labels == num_label] = 0
        else:
            labels[labels == num_label] = num_paintings
            num_paintings += 1
    for num_label in range(1, num_paintings):  # Label equals zero is the background
        label_image = (labels == num_label) * 255
        label_image = label_image.astype('uint8')
        contours, hierarchy = cv2.findContours(label_image, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(label_image, contours, -1, color=(0, 0, 0), thickness=10)
        sorted_approx = find_paintings(contours[0])
        if sorted_approx is not None and not (0, 0) in sorted_approx:
            # upper_left, upper_right, lower_left, lower_right = sorted_approx
            # upper_width = abs(upper_right[1] - upper_left[1])
            # lower_width = abs(lower_right[1] - lower_left[1])
            # right_height = abs(lower_left[0] - upper_left[0])
            # left_height = abs(lower_right[0] - upper_right[0])
            # print(upper_width, lower_width, right_height, left_height)
            list_bounding_boxes.append(sorted_approx)
    return list_bounding_boxes
