import numpy as np
import cv2
import math

from parameters import *
from plotting import plt_images
from detection import edge_detection


def rectification(im, coordinate):
    upper_left, upper_right, down_left, down_right = coordinate

    rows = math.sqrt(math.pow(
        (down_left[0] - upper_left[0]), 2) + math.pow((down_left[1] - upper_left[1]), 2))
    cols = math.sqrt(math.pow(
        (upper_right[0] - upper_left[0]), 2) + math.pow((upper_right[1] - upper_left[1]), 2))
    rows = int(rows)
    cols = int(cols)
    src_points = np.float32([upper_left, down_left, upper_right, down_right])
    dst_points = np.float32(
        [[0, 0], [0, rows - 1], [cols-1, 0], [cols-1, rows - 1]])
    projective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    img_output = cv2.warpPerspective(im, projective_matrix, (cols, rows))

    return img_output


# im = cv2.imread('../data/100.jpg')
# images, titles = edge_detection(im)
# list_painting = get_bounding_boxes(images[-1])
# for painting in range(len(list_painting)):
#     new_im = affine_transformation(im, list_painting[painting])
# #     img_crop = im[y:y+h, x:x+w, :]
#     cv2.imwrite('../output/100_'+str(painting)+'.jpg', new_im)

# cv2.release()
# cv2.destroyAllWindows()
