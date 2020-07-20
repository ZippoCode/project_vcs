import numpy as np
import cv2
import sys

# Custom importing
from parameters import PATH_DESTINATION_PAINTING_DETECTED
from improve_quality import multiscale_retinex
from plotting import plt_images, draw_paintings
from read_write import read_video, store_video, save_paitings


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
    # images.append(msf_image)
    # titles.append('Mean Shift Filtering')

    hsv = cv2.cvtColor(msf_image, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(msf_image, cv2.COLOR_BGR2GRAY)

    H, S, V = np.arange(3)
    gray = cv2.addWeighted(hsv[:, :, V], 0.35, gray, 0.65, 0)
    # thresh, _ = cv2.threshold(gray, 120, 255, cv2.THRESH_OTSU)
    # mask = (gray < thresh).astype(np.uint8) * 255

    average_mean_V = int(np.average(gray))
    ret, mask = cv2.threshold(gray, average_mean_V, 255,
                              cv2.ADAPTIVE_THRESH_GAUSSIAN_C + cv2.THRESH_MASK)
    images.append(mask)
    titles.append('threshold')

    gray = cv2.Canny(mask, 50, 150)
    images.append(gray)
    titles.append('Canny')

    # Erosion and dilation
    gray = cv2.dilate(gray, np.ones((5, 5), dtype=np.uint8), iterations=2)
    gray = cv2.erode(gray, np.ones((3, 3), dtype=np.uint8), iterations=3)
    images.append(gray)
    titles.append('Erosion and dilation')

    # Connected components
    im = connected_components_segmentation(gray)

    # Erosion and dilation
    im = cv2.dilate(im, np.ones((5, 5), dtype=np.uint8), iterations=2)
    im = cv2.erode(im, np.ones((3, 3),
                               dtype=np.uint8), iterations=3)
    images.append(gray)
    titles.append('Erosion and dilation')

    # Connected components
    im = connected_components_segmentation(im)
    images.append(im)
    titles.append('Connected components Image')

    return images, titles


def connected_components_segmentation(im):
    # Connected components
    _, labeled_img = cv2.connectedComponentsWithAlgorithm(
        im, 8, cv2.CV_32S, cv2.CCL_GRANA)
    labels = np.unique(labeled_img)
    labels = labels[labels != 0]
    im = np.zeros_like(labeled_img, dtype=np.uint8)
    for label in labels:
        mask = np.zeros_like(labeled_img, dtype=np.uint8)
        mask[labeled_img == label] = 255
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hull = []
        for cnt in contours:
            hull.append(cv2.convexHull(cnt, False))
        hull_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
        for i in range(len(contours)):
            hull_mask = cv2.drawContours(hull_mask, hull, i, 255, -1, 8)
        im = np.clip(im + hull_mask, 0, 255)
    return im


def sorted_points(contour):
    """
        Given a contour with shape (4, 1, 2) and return the sorted points
        Upper Left, Upper Right, Down Left, Down Right.
    :param contour:
    :return:
    """
    middle_x, middle_y = 0, 0
    upper_left, upper_right, down_left, down_right = (
                                                         0, 0), (0, 0), (0, 0), (0, 0)
    for point in range(contour.shape[0]):
        #   print("X: {}, Y : {}".format(contour[point, 0, 1], contour[point, 0, 0]))
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
    if (upper_right[0] - upper_left[0]) < 75 or (down_left[1] - upper_left[1]) < 75:
        return
    if (down_right[0] - down_left[0]) < 75 or (down_right[1] - upper_right[1]) < 75:
        return
    return upper_left, upper_right, down_left, down_right


def get_bounding_boxes(image):
    """
        Given an image it looks for the paintings and returns a list of bounding boxes
    :param image:
    :return: list of bounding boxes (x, y, w, h)
    """
    list_bounding_boxes = []
    contours, hierarchy = cv2.findContours(
        image, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_TC89_L1)
    for contour in contours:
        epsilon = cv2.arcLength(contour, True) * 0.06
        approx = cv2.approxPolyDP(contour, epsilon=epsilon, closed=True)
        if len(approx) == 4 and cv2.contourArea(contour) > 5000:
            sorted_approx = sorted_points(approx)
            if sorted_approx is not None and not (0, 0) in sorted_approx:
                list_bounding_boxes.append(sorted_approx)

    return list_bounding_boxes


def elaborate_edge_detection(frames, show_images=False):
    """
        Elaborate an video with Edge Detection
    :param path_video: path of original video
    :return:
        - A list of bounding boxes (x, y, w, h)
    """
    if frames is None or len(frames) == 0:
        print("[ERROR] Frames ...")
    print("Elaborating Edge Detection ...")
    frame_results = []
    try:
        for num_frame, frame in enumerate(frames):
            frame_retinex = multiscale_retinex(frame)
            edit_images, edit_titles = edge_detection(frame_retinex)
            list_bounding = get_bounding_boxes(edit_images[-1])
            paintings = []
            titles = []
            result = draw_paintings(frame, list_bounding)
            if show_images:
                images = []
                titles = []
                images.append(frame)
                titles.append("Original Frame")
                images.append(frame_retinex)
                titles.append('Multiscale retinex')
                for image, title in zip(edit_images, edit_titles):
                    images.append(image)
                    titles.append(title)
                result = draw_paintings(frame, list_bounding)
                images.append(result)
                titles.append('Final result')
                plt_images(images, titles)
            titles.append("Detection Frame")
            paintings.append(result)
            frame_results.append(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
            print("[INFO] Elaborate {} of {} frames".format(num_frame + 1, len(frames)))

    except KeyboardInterrupt:
        print('Stop processing')
        pass


    return frame_results
