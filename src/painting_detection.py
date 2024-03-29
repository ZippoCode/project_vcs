import cv2
import math
import matplotlib.pyplot as plt

from improve_quality import multiscale_retinex
from parameters import *


def edge_detection(original_image: np.ndarray):
    """
        Takes an image RGB and return a two lists.
        The first list contains edited images while the second contains
        a name of algorithms which used

    :param original_image: np.ndarray

    :return:
        - a list containing the images of the operations carried out
        - a list containing the names of the changes applied
    """
    images = []
    titles = []

    # image = cv2.copyMakeBorder(original_image, top=VP, bottom=VP, left=VP, right=VP, borderType=cv2.BORDER_CONSTANT)
    # images.append(image)
    # titles.append("Border image")

    # Multi-Scale Retinex
    retinex_image = multiscale_retinex(original_image)
    images.append(retinex_image)
    titles.append('Multi-scale Retinex')
    # cv2.imwrite('../image_results/multiscale-retinex.png', cv2.cvtColor(retinex_image, cv2.COLOR_BGR2RGB))

    # Apply Canny on V-Channel
    h_space, s_space, v_space = cv2.split(cv2.cvtColor(retinex_image, cv2.COLOR_RGB2HSV))
    image = cv2.Canny(v_space, threshold1=80, threshold2=110, L2gradient=True)
    images.append(image)
    titles.append('Canny on S-channel in HSV space')

    # Dilate and Closure the edge
    dilate_image = cv2.dilate(image, kernel=KERNEL_3x3, iterations=NUM_ITERATIONS_DILATE)
    dilate_image = cv2.morphologyEx(dilate_image, cv2.MORPH_CLOSE, kernel=np.ones((3, 3), np.uint8))
    images.append(dilate_image)
    titles.append('Background')
    # cv2.imwrite('../image_results/edge_detection_1.jpg.png', dilate_image)

    # FIND FORMS
    # Rectangle
    linesP = cv2.HoughLinesP(dilate_image, rho=15, theta=np.pi / 180, threshold=150, lines=np.array([]),
                             minLineLength=10,
                             maxLineGap=10)
    lines_image = np.zeros(image.shape, dtype=np.uint8)
    if linesP is not None:
        for i in range(0, len(linesP)):
            cv2.line(lines_image, (linesP[i][0][0], linesP[i][0][1]), (linesP[i][0][2], linesP[i][0][3]), 255, 5,
                     cv2.LINE_AA)
    images.append(lines_image)
    titles.append('Line')

    # Fill the paintings
    hull_image = np.copy(lines_image)
    contours, hierarchy = cv2.findContours(hull_image, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv2.contourArea(x))
    hull_list = [cv2.convexHull(contour) for contour in contours]
    for hull in hull_list:
        epsilon = cv2.arcLength(hull, True) * 0.03
        approx = cv2.approxPolyDP(hull, epsilon=epsilon, closed=True)
        cv2.fillConvexPoly(hull_image, approx, color=COLOR_WHITE)
    images.append(hull_image)
    titles.append('Hull paintings')

    eroded_image = cv2.morphologyEx(hull_image, cv2.MORPH_OPEN, kernel=np.ones((7, 7), np.uint8), iterations=3)
    images.append(eroded_image)
    titles.append('Eroded paintings')

    # Remove border add
    # result_image = eroded_image[VP:original_image.shape[0] - VP, VP: original_image.shape[1] - VP]
    # images.append(result_image)
    # titles.append("Final")

    # cv2.imwrite('../image_results/edge_detection_1.jpg.jpg.png', eroded_image)
    return images, titles
