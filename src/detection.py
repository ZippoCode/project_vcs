import numpy as np
import cv2
import math

from plotting import plt_images


def eight_directional_sobel_filter(image, stride=1):
    """
        Run a Multi-direction Sobel Operator

    :param image:
    :param stride:
    :return:
    """
    height, width = image.shape
    image = cv2.resize(image, None, fx=0.1, fy=0.1,
                       interpolation=cv2.INTER_CUBIC)
    S_h = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    S_v = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    S_dl = np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]])
    S_dr = np.array([[2, 1, 0], [1, 0, -1], [0, -1, -2]])

    kH, kW = S_h.shape

    oH = int((image.shape[0] - (kH - 1) - 1) / stride) + 1
    oW = int((image.shape[1] - (kW - 1) - 1) / stride) + 1

    out = np.zeros((oH, oW), )
    Hor = np.zeros((oH, oW), )
    Ver = np.zeros((oH, oW), )

    for col in range(oH):
        for row in range(oW):
            Gx = np.sum(image[col * stride: col * stride + kH,
                              row * stride: row * stride + kW] * S_h)
            Gy = np.sum(image[col * stride: col * stride + kH,
                              row * stride: row * stride + kW] * S_v)
            G_dl = np.sum(image[col * stride: col * stride +
                                kH, row * stride: row * stride + kW] * S_dl)
            G_dr = np.sum(image[col * stride: col * stride +
                                kH, row * stride: row * stride + kW] * S_dr)
            M = np.sqrt(Gx ** 2 + Gy ** 2)

            Hor[col, row] = Gx
            Ver[col, row] = Gy
            out[col, row] = M

    # Normalize Magnitude and Direction
    out = cv2.normalize(out, None, 0, 255, cv2.NORM_MINMAX)
    out = cv2.resize(out, (width, height), interpolation=cv2.INTER_CUBIC)
    return out.astype(np.uint8)


def edge_detection(im):
    """
        Takes an image RGB and return a two lists.
        The first list contains edited images while the second contains
        a name of algorithms which used

    :param im:
    :return:
    """
    list_painting = []

    images = []
    titles = []

    # Original Image
    im_original = im.copy()
    images.append(im_original)
    titles.append('Original Image')

    # PYR MEAN SHIFT FILTERING
    im = cv2.pyrMeanShiftFiltering(im, sp=8, sr=8, maxLevel=3)
    images.append(im)
    titles.append('Mean Shift Filtering')

    # Erosion and dilation

    erosion_size = 20
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erosion_size + 1, 2 * erosion_size + 1),
                                        (erosion_size, erosion_size))
    im = cv2.erode(src=im, kernel=element)
    images.append(im)
    titles.append("Erosed Image")

    # Apply difference Threshold
    hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
    H, S, V = np.arange(3)
    ret, hsv[:, :, V] = cv2.threshold(
        hsv[:, :, V], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    scale = 1
    delta = 0
    ddepth = cv2.CV_16S
    grad_x = cv2.Sobel(hsv[:, :, V], ddepth, 1, 0, ksize=3,
                       scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    grad_y = cv2.Sobel(hsv[:, :, V], ddepth, 0, 1, ksize=3,
                       scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    hsv[:, :, V] = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    im = hsv[:, :, V].astype(np.uint8)
    # Apply to threshould
    images.append(im)
    titles.append("Threshold and Sobel image")

    # Lines
    im_lines = np.copy(im_original)
    contours, _ = cv2.findContours(
        im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        epsilon = cv2.arcLength(contour, True) * 0.06
        approx = cv2.approxPolyDP(contour, epsilon=epsilon, closed=True)
        if len(approx) == 4 and cv2.contourArea(contour) > 5000:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(im_lines, (x, y), (x + w, y + h), (0, 255, 0), 3)
            list_painting.append((x, y, w, h))

    images.append(im_lines)
    titles.append('Image with Line')
    #   plt_images(images, titles)

    return list_painting
