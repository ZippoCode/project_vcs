import numpy as np
import cv2
import math

from detection import edge_detection
from parameters import *
from plotting import plt_images


def recfitication(im):
    images = []
    titles = []

    img = im.copy()
    images.append(img)
    titles.append('Original Image')

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 15)
    images.append(gray)
    titles.append('Gray')

    # Adaptive Threshold
    gray = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    images.append(gray)
    titles.append('Adaptive Theshold')

    gray = cv2.medianBlur(gray, 3)
    images.append(gray)
    titles.append('MedianBlur')

    # Connected Component
    KERNEL_HIGH_PASS_FILTER = np.asarray(
        [[0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0]], np.uint8)
    gray = cv2.erode(gray, KERNEL_HIGH_PASS_FILTER)
    gray = cv2.dilate(gray, np.ones(DILATE_KERNEL_SIZE,
                                    dtype=np.uint8), iterations=DILATE_ITERATIONS)
    gray = cv2.erode(gray, KERNEL_HIGH_PASS_FILTER,
                     iterations=EROSION_ITERATIONS)
    images.append(gray)
    titles.append('Erode Dilate')

    # Connected components
    _, labeled_img = cv2.connectedComponentsWithAlgorithm(
        gray, 8, cv2.CV_32S, cv2.CCL_GRANA)
    images.append(labeled_img)
    titles.append('Labeled Img 01')

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
    images.append(global_mask)
    titles.append('Connected components')

    # detect corners with the goodFeaturesToTrack function.
    corners = cv2.goodFeaturesToTrack(global_mask, 27, 0.01, 10)
    corners = np.int0(corners)

    # we iterate through each corner,
    # making a circle at each point that we think is a corner.
    for i in corners:
        x, y = i.ravel()
        cv2.circle(img, (x, y), 3, 255, -1)
    images.append(img)
    titles.append('Detect corner')

    plt_images(images, titles)
    return gray

# recfitication('../data/Amalia_image.jpeg')


im = cv2.imread('../data/100.jpg')
list_painting = edge_detection(im)
print(list_painting)
drawing_frame = im.copy()
for painting in range(len(list_painting)):
    x, y, w, h = list_painting[painting]
    img_crop = im[y:y+h, x:x+w, :]
    gray = recfitication(img_crop)
    # cv2.imwrite('../output/100_'+str(painting)+'.jpg', img_crop)
    # drawing_frame = cv2.rectangle(
    #     im, (x, y), (x + w, y + h), (0, 255, 0), 3)
    # cv2.imshow('crop_img', img_crop)
    # cv2.waitKey()

# cv2.imshow('Edge Detection', drawing_frame)
# cv2.waitKey()
