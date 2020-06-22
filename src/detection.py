import numpy as np
import cv2

# Custom importing
from improve_quality import multiscale_retinex
from plotting import plt_images, draw_paintings


# def eight_directional_sobel_filter(image, stride=1):
#     """
#         Run a Multi-direction Sobel Operator
#
#     :param image:
#     :param stride:
#     :return:
#     """
#     height, width = image.shape
#     image = cv2.resize(image, None, fx=0.1, fy=0.1,
#                        interpolation=cv2.INTER_CUBIC)
#     S_h = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
#     S_v = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
#     S_dl = np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]])
#     S_dr = np.array([[2, 1, 0], [1, 0, -1], [0, -1, -2]])
#
#     kH, kW = S_h.shape
#
#     oH = int((image.shape[0] - (kH - 1) - 1) / stride) + 1
#     oW = int((image.shape[1] - (kW - 1) - 1) / stride) + 1
#
#     out = np.zeros((oH, oW), )
#     Hor = np.zeros((oH, oW), )
#     Ver = np.zeros((oH, oW), )
#
#     for col in range(oH):
#         for row in range(oW):
#             Gx = np.sum(image[col * stride: col * stride + kH,
#                         row * stride: row * stride + kW] * S_h)
#             Gy = np.sum(image[col * stride: col * stride + kH,
#                         row * stride: row * stride + kW] * S_v)
#             G_dl = np.sum(image[col * stride: col * stride +
#                                               kH, row * stride: row * stride + kW] * S_dl)
#             G_dr = np.sum(image[col * stride: col * stride +
#                                               kH, row * stride: row * stride + kW] * S_dr)
#             M = np.sqrt(Gx ** 2 + Gy ** 2)
#
#             Hor[col, row] = Gx
#             Ver[col, row] = Gy
#             out[col, row] = M
#
#     # Normalize Magnitude and Direction
#     out = cv2.normalize(out, None, 0, 255, cv2.NORM_MINMAX)
#     out = cv2.resize(out, (width, height), interpolation=cv2.INTER_CUBIC)
#     return out.astype(np.uint8)


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
    titles.append('Mean Shift Filtering')

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
    if (upper_right[0] - upper_left[0]) < 150 or (down_left[1] - upper_left[1]) < 150:
        return
    if (down_right[0] - down_left[0]) < 150 or (down_right[1] - upper_right[1]) < 150:
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


def elaborate_edge_detection(frame, show_images=False):
    """
        Elaborate an frame with Edge Detection and Rectification
    :param frame: numpy.ndarray with shape (H, W, C)
    :return:
        - A list of bounding boxes (x, y, w, h)
    """
    frame_retinex = multiscale_retinex(frame)
    edit_images, edit_titles = edge_detection(frame_retinex)
    # plt_images(edit_images, edit_titles)
    list_bounding = get_bounding_boxes(edit_images[-1])

    if show_images:
        images = []
        titles = []
        # Append original frame
        images.append(frame)
        titles.append("Original Frame")
        # Append frame with Retinex elaboration
        images.append(frame_retinex)
        titles.append('Multiscale retinex')
        # Append all images from elaboration
        for image in edit_images:
            images.append(image)
        for title in edit_titles:
            titles.append(title)
        # Drawing image with the rectangle, points and line
        result = draw_paintings(frame, list_bounding)
        images.append(result)
        titles.append('Final result')
        # Show the steps of image elaboration
        plt_images(images, titles)

    return list_bounding
