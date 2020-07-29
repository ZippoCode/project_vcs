import cv2


def sorted_points(contour):
    """
        Given a contour with shape (4, 1, 2) and return the sorted points
        Upper Left, Upper Right, Down Left, Down Right.

    :param contour:
    :return:
    """
    middle_x, middle_y = 0, 0
    upper_left, upper_right, down_left, down_right = (0, 0), (0, 0), (0, 0), (0, 0)
    for point in range(contour.shape[0]):
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
    # if (upper_right[0] - upper_left[0]) < 75 or (down_left[1] - upper_left[1]) < 75:
    #     return
    # if (down_right[0] - down_left[0]) < 75 or (down_right[1] - upper_right[1]) < 75:
    #     return
    return upper_left, upper_right, down_left, down_right


def get_bounding_boxes(original_image, edge_detection_image):
    """
        Given an image it looks for the paintings and returns a list of bounding boxes

    :param image:
    :return: list of bounding boxes (x, y, w, h)
    """

    contours, hierarchy = cv2.findContours(edge_detection_image, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return []

    list_bounding_boxes = []
    threshold = sum([cv2.contourArea(c) for c in contours]) / len(contours)

    for contour in contours:
        epsilon = cv2.arcLength(contour, True) * 0.1
        approx = cv2.approxPolyDP(contour, epsilon=epsilon, closed=True)
        if len(approx) == 4 and cv2.contourArea(approx) > threshold:     # Find Rectangle
            sorted_approx = sorted_points(approx)
            if sorted_approx is not None and not (0, 0) in sorted_approx:
                list_bounding_boxes.append(sorted_approx)
            # cv2.drawContours(original_image, [approx], 0, (255, 255, 255), 2)

    # elif len(approx) > 8 and cv2.contourArea(contour) > 5000:
    #     (x, y), radius = cv2.minEnclosingCircle(contour)
    #     (x, y), radius = (int(x), int(y)), int(radius)
    #     top_left_x = x - radius if x - radius >= 0 else 0
    #     top_left_y = y - radius if y - radius >= 0 else 0
    #     bottom_right_x = x + radius if x + radius <= image.shape[1] else image.shape[1]
    #     bottom_right_y = y + radius if y + radius <= image.shape[0] else image.shape[0]
    #
    #     upper_left = (top_left_x, top_left_y)
    #     upper_right = (top_left_x, bottom_right_y)
    #     down_left = (bottom_right_x, top_left_y)
    #     down_right = (bottom_right_x, bottom_right_y)
    #     list_bounding_boxes.append((upper_left, upper_right, down_left, down_right))

    # plt.imshow(original_image)
    # plt.show()
    return list_bounding_boxes
