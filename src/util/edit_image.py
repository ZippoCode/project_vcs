import cv2

from constants.colors import *


def reduce_size(image):
    """
        Reduce size of shape from image

    :param image:
    :return:
    """
    if image is None:
        print(f"{FAIL}[ERROR] Image is None.")
    if len(image.shape) < 2:
        print(f"{FAIL}[ERROR] Shape of image is wrong.")
        return
    h, w = int(image.shape[0]), int(image.shape[1])
    thr_w, thr_h = 500, 500
    if h > thr_h or w > thr_w:
        h_ratio = thr_h / h
        w_ratio = thr_w / w
        w = int(image.shape[1] * min(h_ratio, w_ratio))
        h = int(image.shape[0] * min(h_ratio, w_ratio))
        frame = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
    return frame


def draw_paintings(image, list_painting):
    """
        Given a images with shape (H, W, C) and a list of painting
        this method draws the lines, the points and a rectangles on each paintings
    :param:
        - image: numpy.ndarray with shape (H, W, C)
        - list_paintings: a list of pointing.
    :return:
        - image: numpy.ndarray
    """
    if image is None or len(image.shape) != 3:
        print(f"{FAIL}[ERROR] Shape is wrong")
        return image

    image_painting = image.copy()
    for upper_left, upper_right, down_left, down_right in list_painting:
        cv2.line(img=image_painting, pt1=upper_left, pt2=upper_right, color=color_green)
        cv2.line(img=image_painting, pt1=upper_left, pt2=down_left, color=color_green)
        cv2.line(img=image_painting, pt1=down_left, pt2=down_right, color=color_green)
        cv2.line(img=image_painting, pt1=upper_right, pt2=down_right, color=color_green)

        cv2.circle(image_painting, upper_left, radius=2, color=color_green)
        cv2.circle(image_painting, down_left, radius=2, color=color_red)
        cv2.circle(image_painting, upper_right, radius=2, color=color_blue)
        cv2.circle(image_painting, down_right, radius=2, color=color_yellow)

    return image_painting
