import numpy as np
import cv2
import math

from detection import edge_detection, get_bounding_boxes
from parameters import *
from plotting import plt_images
import matplotlib.pyplot as plt


def recfitication(im):
    images = []
    titles = []

    img = im.copy()
    images.append(img)
    titles.append('Original Image')

    # Convert to gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Smooth
    gray = cv2.GaussianBlur(gray, (3, 3), 15)
    images.append(gray)
    titles.append('Gray')

    # Adaptive Threshold
    gray = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # Edge_detection
    canny = cv2.Canny(img, 50, 240)

    images.append(gray)
    titles.append('Adaptive Theshold')

    gray = cv2.medianBlur(gray, 3)
    images.append(gray)
    titles.append('MedianBlur')

    KERNEL_HIGH_PASS_FILTER = np.asarray(
        [[0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0]], np.uint8)
    gray = cv2.erode(gray, KERNEL_HIGH_PASS_FILTER)
    gray = cv2.dilate(gray, np.ones(DILATE_KERNEL_SIZE,
                                    dtype=np.uint8), iterations=DILATE_ITERATIONS)
    gray = cv2.erode(gray, KERNEL_HIGH_PASS_FILTER,
                     iterations=EROSION_ITERATIONS)

    # Connected components
    _, labeled_img = cv2.connectedComponentsWithAlgorithm(
        gray, 8, cv2.CV_32S, cv2.CCL_GRANA)
    labels = np.unique(labeled_img)
    labels = labels[labels != 0]
    im = np.zeros_like(labeled_img, dtype=np.uint8)
    global_mask = np.zeros_like(labeled_img, dtype=np.uint8)
    for label in labels:
        mask = np.zeros_like(labeled_img, dtype=np.uint8)
        mask[labeled_img == label] = 255

        # Compute the convex hull
        contours, _ = cv2.findContours(
            mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        hull = []
        for cnt in contours:
            hull.append(cv2.convexHull(cnt, False))
        hull_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
        for i in range(len(contours)):
            hull_mask = cv2.drawContours(hull_mask, hull, i, 255, -1, 8)

        global_mask = np.clip(
            global_mask + hull_mask, 0, 255)

    # dst = cv2.cornerHarris(global_mask, 2, 3, 0.04)
    # # result is dilated for marking the corners, not important
    # dst = cv2.dilate(dst, None)
    # # Threshold for an optimal value, it may vary depending on the image.
    # img[dst > 0.01*dst.max()] = [0, 0, 255]
    # images.append(img)
    # titles.append('Detect corner')

    # detect corners with the goodFeaturesToTrack function.
    corners = cv2.goodFeaturesToTrack(global_mask, 27, 0.01, 10)
    corners = np.int0(corners)

    # we iterate through each corner,
    # making a circle at each point that we think is a corner.
    for i in corners:
        x, y = i.ravel()
        # print(x, y)
        cv2.circle(img, (x, y), 3, 255, -1)
    images.append(img)
    titles.append('Detect corner')

    # plt_images(images, titles)
    return gray

# recfitication('../data/Amalia_image.jpeg')


def affine_transformation(im, coordinate):
    upper_left, upper_right, down_left, down_right = coordinate

    rows = math.sqrt(math.pow(
        (upper_right[0] - upper_left[0]), 2) + math.pow((upper_right[1] - upper_left[1]), 2))
    cols = math.sqrt(math.pow(
        (down_left[0] - upper_left[0]), 2) + math.pow((down_left[1] - upper_left[1]), 2))
    rows = int(rows)
    cols = int(cols)
    src_points = np.float32([upper_left, down_left, upper_right, down_right])
    dst_points = np.float32([[0, 0], [0, rows-1], [cols, 0], [cols, rows-1]])
    projective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    img_output = cv2.warpPerspective(im, projective_matrix, (cols, rows))

    return img_output


im = cv2.imread('../data/100.jpg')
images, titles = edge_detection(im)
list_painting = get_bounding_boxes(images[-1])
for painting in range(len(list_painting)):
    new_im = affine_transformation(im, list_painting[painting])
# #     img_crop = im[y:y+h, x:x+w, :]
#     cv2.imwrite('../output/100_'+str(painting)+'.jpg', new_im)

# cv2.release()
# cv2.destroyAllWindows()
