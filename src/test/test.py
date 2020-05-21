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

# cv2.imshow('Image', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
