import numpy as np
import cv2

from matplotlib import pyplot as plt


def show_image(images=[], titles=[]):
    fig = plt.figure(figsize=(150, 200))
    nrows = 1
    ncols = 6
    for img in range(len(images)):
        fig.add_subplot(nrows, ncols, img + 1)
        if (len(images[img].shape) < 3):
            plt.imshow(images[img], cmap='gray')
        else:
            plt.imshow(images[img])
        plt.title(titles[img])
        plt.xticks([])
        plt.yticks([])

    # plt.show()


def edge_detection(im):
    images = []
    titles = []

    # Original Image
    im_original = im.copy()
    images.append(im_original)
    titles.append('Original Image')

    # PYR MEAN SHIFT FILTERING
    im = cv2.pyrMeanShiftFiltering(im, sp=2, sr=8, maxLevel=3)
    images.append(im)
    titles.append('Mean Shift Filtering')
    cv2.imshow('Mean Shift Filtering', im)
    cv2.waitKey(0)

    # Apply threshould
    hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
    im = cv2.inRange(hsv, (0, 100, 100), (255, 200, 200))
    # im = cv2.cvtColor(im_threshould, cv2.COLOR_HSV2RGB)
    images.append(im)
    titles.append("Threshould Image")
    cv2.imshow('Threshould Image', im)
    cv2.waitKey(0)

    """
    # Try to search pixel's values which correspond at the wall and I try to remove these
    im_HSV = cv2.cvtColor(im, cv2.COLOR_RGB2HLS)
    L = 2
    im_HSV[:, :, L] = np.where(im_HSV[:, :, L] < np.average(im_HSV[:, :, L]), 255, im_HSV[:, :, L])
    im = cv2.cvtColor(im_HSV, cv2.COLOR_HSV2BGR)
    images.append(im);
    titles.append('TRY')
    """

    # Canny
    # im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    im = cv2.Canny(im, threshold1=50, threshold2=150,
                   apertureSize=3, L2gradient=True)
    images.append(im)
    titles.append('Canny Filtering Image')
    cv2.imshow('Canny', im)
    cv2.waitKey(0)

    # Lines
    im_lines = np.copy(im_original)
    minLineLength = 50
    maxLineGap = 5
    lines = cv2.HoughLinesP(im, 1, np.pi / 180, 10, minLineLength, maxLineGap)
    if len(lines) > 0:
        for x in range(0, len(lines)):
            for x1, y1, x2, y2 in lines[x]:
                cv2.line(im_lines, (x1, y1), (x2, y2), (255, 0, 0), 3)
    images.append(im_lines)
    titles.append('Image with Line')

    #show_image(images, titles)
    return im_lines
