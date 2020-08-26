import cv2

from improve_quality import multiscale_retinex
from parameters import *


def edge_detection(im):
    """
        Takes an image RGB and return a two lists.
        The first list contains edited images while the second contains
        a name of algorithms which used

    :param im: original image

    :return:
        - a list containing the images of the operations carried out
        - a list containing the names of the changes applied
    """
    images = []
    titles = []

    # Multi-Scale Retinex
    frame_retinex = multiscale_retinex(im)
    images.append(frame_retinex)
    titles.append('Multi-scale Retinex')

    # PYR MEAN SHIFT FILTERING
    msf_image = cv2.pyrMeanShiftFiltering(frame_retinex, sp=SPATIAL_WINDOW_RADIUS, sr=COLOR_WINDOW_RADIUS, maxLevel=3)
    images.append(msf_image)
    titles.append("Mean Shift Filtering")

    # Apply Canny on S-Channel
    hsv = cv2.cvtColor(msf_image, cv2.COLOR_RGB2HSV)
    h_space, s_space, v_space = cv2.split(hsv)
    threshold = int(min(255, 1.5 * np.median(s_space)))
    # print(f"[INFO] Threshold Min: {threshold_min} - Threshold  {threshold_max}")
    # s_space_threshold = cv2.Canny(s_space, threshold1=threshold_min, threshold2=threshold_max)
    s_space_threshold = cv2.adaptiveThreshold(v_space, maxValue=threshold,
                                              adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                              thresholdType=cv2.THRESH_BINARY_INV, blockSize=11, C=2)
    images.append(s_space_threshold)
    titles.append('Adaptive Threshold on S-channel in HSV space')

    # Sure background area
    sure_bg = cv2.dilate(s_space_threshold, kernel=KERNEL_3x3, iterations=NUM_ITERATIONS_DILATE)
    images.append(sure_bg)
    titles.append('Background')

    # Fill the contours with white color
    contours, _ = cv2.findContours(sure_bg, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
    img_fill = sure_bg.copy()
    for contour in contours:
        cv2.fillPoly(img_fill, pts=[contour], color=COLOR_WHITE)
    images.append(img_fill)
    titles.append('Fill Image')

    # Erode Fill Countours
    fill_erode = cv2.erode(img_fill, kernel=KERNEL_3x3, iterations=NUM_ITERATIONS_ERODE)
    images.append(fill_erode)
    titles.append("Erode Fill Image")

    return images, titles
