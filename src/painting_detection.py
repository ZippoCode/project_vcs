import numpy as np
import cv2


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

    # # PYR MEAN SHIFT FILTERING
    msf_image = cv2.pyrMeanShiftFiltering(im, sp=8, sr=8, maxLevel=3)
    images.append(msf_image)
    titles.append("Mean Shift Filtering")

    # Apply Threshold on S-Channel
    hsv = cv2.cvtColor(msf_image, cv2.COLOR_RGB2HSV)
    h_space, s_space, v_space = cv2.split(hsv)
    s_space_threshold = cv2.adaptiveThreshold(v_space, maxValue=np.max(s_space),
                                              adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                              thresholdType=cv2.THRESH_BINARY_INV, blockSize=11, C=2)
    images.append(s_space_threshold)
    titles.append('Threshold S-channel in HSV space')

    # # Erosion and dilation
    img_dilate = cv2.dilate(s_space_threshold, np.ones((3, 3), dtype=np.uint8), iterations=1)
    img_erose = cv2.dilate(img_dilate, np.ones((3, 3), dtype=np.uint8), iterations=1)
    images.append(img_erose)
    titles.append('Erosion and dilation')

    # Distance Transform
    image_dist = cv2.distanceTransform(img_erose, distanceType=cv2.DIST_C, maskSize=3)
    cv2.normalize(image_dist, image_dist, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX)
    _, image_dist = cv2.threshold(image_dist, 0.2, 1.0, cv2.THRESH_BINARY)
    image_dist = cv2.dilate(image_dist, kernel=np.ones((3, 3), dtype=np.uint8))
    image_dist = image_dist.astype('uint8')
    images.append(image_dist)
    titles.append('Image Transform')

    # Apply Canny on V-Channel Space
    # img_canny = cv2.Canny(img_erose, threshold1=np.mean(msf_image), threshold2=np.max(msf_image))
    # images.append(img_canny)
    # titles.append('Canny Edge Detection')

    # Connected components
    im_ccs = connected_components_segmentation(image_dist)
    images.append(im_ccs)


    titles.append('Connected components Image')

    return images, titles


def connected_components_segmentation(im):
    """

    :param im:
    :return:
    """
    _, labeled_img = cv2.connectedComponentsWithAlgorithm(im, 8, cv2.CV_32S, cv2.CCL_GRANA)
    labels = np.unique(labeled_img)
    labels = labels[labels != 0]
    im = np.zeros_like(labeled_img, dtype=np.uint8)
    for label in labels:
        mask = np.zeros_like(labeled_img, dtype=np.uint8)
        mask[labeled_img == label] = 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hull = []
        for cnt in contours:
            hull.append(cv2.convexHull(cnt, False))
        hull_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
        for i in range(len(contours)):
            hull_mask = cv2.drawContours(hull_mask, hull, i, 255, -1, 8)
        im = np.clip(im + hull_mask, 0, 255)
    return im
