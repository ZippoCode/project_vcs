import numpy as np
import cv2
import argparse
import imutils

from skimage import exposure
from matplotlib import pyplot as plt
# from pyimagesearch import imutils


def plt_images(images, titles):
    """
    Take a vector of image and a vector of title and show these on the screen

    :param images
    :param titles
    :return:
    """
    if len(images) != len(titles) or len(images) > 20:
        return
    fig = plt.figure(figsize=(150, 200))
    nrows = 3
    ncols = 3
    for img in range(len(images)):
        fig.add_subplot(nrows, ncols, img + 1)
        if (len(images[img].shape) < 3):
            plt.imshow(images[img], cmap='gray')
        else:
            plt.imshow(images[img])
        plt.title(titles[img])
        plt.xticks([])
        plt.yticks([])

    plt.show()


images = []
titles = []

img = cv2.imread('../data/100.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# gray = cv2.GaussianBlur(gray, (3, 3), 15)
# in the image
gray = cv2.bilateralFilter(gray, 11, 17, 17)
images.append(gray)
titles.append('Bilateral Filter')

edged = cv2.Canny(gray, 30, 200)
images.append(edged)
titles.append('Canny')

cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]
screenCnt = None

# loop over our contours
for c in cnts:
        # approximate the contour
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.015 * peri, True)
    # if our approximated contour has four points, then
    # we can assume that we have found our screen
    if len(approx) == 4:
        screenCnt = approx
        break
cv2.drawContours(img, [screenCnt], -1, (0, 255, 0), 3)
images.append(img)
titles.append('Detect corner')

plt_images(images, titles)


def crop_rotated_rectangle(self, img, coords):
    # find rotated rectangle
    rect = cv2.minAreaRect(coords.reshape(4, 1, 2).astype(np.float32))
    rbox = self.order_points(cv2.boxPoints(rect))
    # get width and height of the detected rectangle
    # output of minAreaRect is unreliable for already axis aligned rectangles!!
    width = np.linalg.norm([rbox[0, 0] - rbox[1, 0], rbox[0, 1] - rbox[1, 1]])
    height = np.linalg.norm(
        [rbox[0, 0] - rbox[-1, 0], rbox[0, 1] - rbox[-1, 1]])
    src_pts = rbox.astype(np.float32)
    # coordinate of the points in box points after the rectangle has been straightened
    # this step needs order_points to be called on src
    dst_pts = np.array([[0, 0],
                        [width - 1, 0],
                        [width - 1, height - 1],
                        [0, height - 1]], dtype="float32")
    # the perspective transformation matrix
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    # directly warp the rotated rectangle to get the straightened rectangle
    warped = cv2.warpPerspective(img, M, (width, height), None, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT,
                                 (255, 255, 255))
    return warped
