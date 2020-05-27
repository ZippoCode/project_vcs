import numpy as np
import cv2
import math

def rectification(im, coordinate):
    upper_left, upper_right, down_left, down_right = coordinate

    rows = math.sqrt(math.pow(
        (down_left[0] - upper_left[0]), 2) + math.pow((down_left[1] - upper_left[1]), 2))
    cols = math.sqrt(math.pow(
        (upper_right[0] - upper_left[0]), 2) + math.pow((upper_right[1] - upper_left[1]), 2))
    rows = int(rows)
    cols = int(cols)
    src_points = np.float32([upper_left, down_left, upper_right, down_right])
    dst_points = np.float32(
        [[0, 0], [0, rows - 1], [cols - 1, 0], [cols - 1, rows - 1]])
    projective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    img_output = cv2.warpPerspective(im, projective_matrix, (cols, rows))

    return img_output
