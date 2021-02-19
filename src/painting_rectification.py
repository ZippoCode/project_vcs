import numpy as np
import cv2
import math
import matplotlib.pyplot as plt


# def get_calibration(image, bounding_boxes):
#     height, width, channels = image.shape
#     dist_coefficients = np.zeros((4, 1), np.float64)
#     k1, k2, p1, p2 = -1.0e-5, 0.0, 0.0, 0.0
#
#     dist_coefficients[0, 0] = k1
#     dist_coefficients[1, 0] = k2
#     dist_coefficients[2, 0] = p1
#     dist_coefficients[3, 0] = p2
#
#     cam = np.eye(3, dtype=np.float32)
#     cam[0, 2] = width / 2.0  # define center x
#     cam[1, 2] = height / 2.0  # define center y
#     cam[0, 0] = 10.  # define focal length x
#     cam[1, 1] = 10.  # define focal length y
#
#     new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(cam, dist_coefficients, (width, height), 1, (width, height))
#     output = cv2.undistort(image, cam, dist_coefficients, None, new_camera_matrix)
#     plt.imshow(output)
#     plt.show()
#     bounding_boxes = np.swapaxes(np.expand_dims(np.array(bounding_boxes, dtype=np.float32), axis=0), 1, 0)
#     new_bounding_boxes = cv2.undistortPoints(bounding_boxes, cam, dist_coefficients, None, new_camera_matrix)
#     new_bounding_boxes = np.swapaxes(new_bounding_boxes, 0, 1).squeeze().tolist()
#     output_bounding_boxes = list()
#     for x, y in new_bounding_boxes:
#         output_bounding_boxes.append((int(x), int(y)))
#
#     return output, output_bounding_boxes


# def get_new_coordinates(bounding_boxes, x, y, height, width):
#     upper_leftX = bounding_boxes[0][0]
#     upper_leftY = bounding_boxes[0][1]
#     lower_rightX = bounding_boxes[3][0]
#     lower_rightY = bounding_boxes[3][1]
#
#     sizeX = lower_rightX - upper_leftX
#     sizeY = lower_rightY - upper_leftY
#     size_max = max(sizeX, sizeY)
#
#     centerX = (lower_rightX + upper_leftX) / 2
#     centerY = (lower_rightY + upper_leftY) / 2
#
#     offesetX = (centerX - size_max / 2) * height / size_max
#     offesetY = (centerY - size_max / 2) * width / size_max
#
#     x = x * height / size_max - offesetX
#     y = y * width / size_max - offesetY
#
#     return (x, y)


def rectification(image, coordinate):
    upper_left, upper_right, down_left, down_right = coordinate

    rows = int(math.sqrt(math.pow((down_left[0] - upper_left[0]), 2) + math.pow((down_left[1] - upper_left[1]), 2)))
    cols = int(math.sqrt(math.pow((upper_right[0] - upper_left[0]), 2) + math.pow((upper_right[1] - upper_left[1]), 2)))
    src_points = np.float32([upper_left, down_left, upper_right, down_right])
    dst_points = np.float32([[0, 0], [0, rows - 1], [cols - 1, 0], [cols - 1, rows - 1]])
    projective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    img_output = cv2.warpPerspective(image, projective_matrix, (cols, rows))

    return img_output
