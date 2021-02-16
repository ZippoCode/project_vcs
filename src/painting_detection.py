import cv2
import math
import matplotlib.pyplot as plt

from improve_quality import multiscale_retinex
from parameters import *


def edge_detection(image):
    """
        Takes an image RGB and return a two lists.
        The first list contains edited images while the second contains
        a name of algorithms which used

    :param image: original image

    :return:
        - a list containing the images of the operations carried out
        - a list containing the names of the changes applied
    """
    images = []
    titles = []

    # Multi-Scale Retinex
    retinex_image = multiscale_retinex(image)
    images.append(retinex_image)
    titles.append('Multi-scale Retinex')

    # Bilateral Filter
    bilateral_image = cv2.bilateralFilter(retinex_image, d=9, sigmaColor=75, sigmaSpace=75)
    images.append(bilateral_image)
    titles.append('Bilateral Image')

    # Apply Canny on V-Channel
    h_space, s_space, v_space = cv2.split(cv2.cvtColor(bilateral_image, cv2.COLOR_RGB2HSV))
    image = cv2.Canny(v_space, threshold1=70, threshold2=150)
    images.append(image)
    titles.append('Canny on S-channel in HSV space')

    # Dilate and Closure the edge
    dilate_image = cv2.dilate(image, kernel=KERNEL_3x3, iterations=NUM_ITERATIONS_DILATE)
    dilate_image = cv2.morphologyEx(dilate_image, cv2.MORPH_CLOSE, kernel=np.ones((3, 3), np.uint8))
    images.append(dilate_image)
    titles.append('Background')

    # Fill the paintings
    hull_image = dilate_image.copy()
    contours, hierarchy = cv2.findContours(hull_image, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=lambda x: cv2.contourArea(x))
    hull_list = [cv2.convexHull(contour) for contour in contours]
    for hull_contour in hull_list:
        epsilon = cv2.arcLength(hull_contour, True) * 0.03
        approx = cv2.approxPolyDP(hull_contour, epsilon=epsilon, closed=True)
        cv2.fillConvexPoly(hull_image, approx, color=COLOR_WHITE)
    images.append(hull_image)
    titles.append('Fill paintings')

    eroded_image = cv2.morphologyEx(hull_image, cv2.MORPH_OPEN, kernel=np.ones((5, 5), np.uint8))
    images.append(eroded_image)
    titles.append('Eroded paintings')

    return images, titles
