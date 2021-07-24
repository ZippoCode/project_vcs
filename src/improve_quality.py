import numpy as np
import cv2

# Custom importing
from parameters import SIGMA, LOW_CLIP, HIGH_CLIP


def multiscale_retinex(image: np.ndarray):
    """
        Using the Multiscale Retinex with Color Restoration for image enhancement
    :param image: np.ndarray
        Original Image with shape (H, W, C)
    :return:
    """
    if len(image.shape) != 3:
        print('Image need have shape (H, W, C)')
        return
    h, w = image.shape[:2]
    h_ratio = 300 / h
    w_ratio = 300 / w
    new_w = int(image.shape[1] * min(h_ratio, w_ratio))
    new_h = int(image.shape[0] * min(h_ratio, w_ratio))
    image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    image = image.astype(np.float64) + 1.0

    intensity = np.sum(image, axis=2) / image.shape[2]

    # Multi-scale Retinex
    retinex = np.zeros_like(intensity)
    for s in SIGMA:
        retinex += np.log10(intensity) - np.log10(cv2.GaussianBlur(intensity, (0, 0), s))
    retinex /= len(SIGMA)

    intensity = np.expand_dims(intensity, 2)
    retinex = np.expand_dims(retinex, 2)

    # Simplest color balance
    total = retinex.shape[0] * retinex.shape[1]
    for c in range(retinex.shape[2]):
        unique, counts = np.unique(retinex[:, :, c], return_counts=True)
        current, high_val, low_val = 0, 0, 0
        for u, count in zip(unique, counts):
            if float(current) / total < LOW_CLIP:
                low_val = u
            if float(current) / total < HIGH_CLIP:
                high_val = u
            current += count
        retinex[:, :, c] = np.maximum(np.minimum(retinex[:, :, c], high_val), low_val)

    retinex = (retinex - np.min(retinex)) / (np.max(retinex) - np.min(retinex)) * 255.0 + 1.0
    out = np.zeros_like(image)
    for y in range(out.shape[0]):
        for x in range(out.shape[1]):
            B = np.max(image[y, x])
            A = np.minimum(256.0 / B, retinex[y, x, 0] / intensity[y, x, 0])
            out[y, x, 0] = A * image[y, x, 0]
            out[y, x, 1] = A * image[y, x, 1]
            out[y, x, 2] = A * image[y, x, 2]

    return cv2.resize(np.uint8(out - 1.0), (w, h), interpolation=cv2.INTER_AREA)
