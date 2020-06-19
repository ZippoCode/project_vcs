import cv2
import numpy as np


def yoloFormattocv(x1, y1, x2, y2, H, W):
    bbox_width = x2 * W
    bbox_height = y2 * H
    center_x = x1 * W
    center_y = y1 * H

    voc = []
    voc.append(center_x - (bbox_width / 2))
    voc.append(center_y - (bbox_height / 2))
    voc.append(center_x + (bbox_width / 2))
    voc.append(center_y + (bbox_height / 2))

    return [int(v) for v in voc]


def cvFormattoYolo(corner, H, W):
    bbox_W = corner[3] - corner[1]
    bbox_H = corner[4] - corner[2]

    center_bbox_x = (corner[1] + corner[3]) / 2
    center_bbox_y = (corner[2] + corner[4]) / 2

    return corner[0], round(center_bbox_x / W, 6), round(center_bbox_y / H, 6), round(bbox_W / W, 6), round(bbox_H / H,
                                                                                                            6)


def rotate_image(image, angle):
    height, width = image.shape[:2]

    image_center = (width / 2, height / 2)
    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)
    abs_cos = abs(rotation_mat[0, 0])
    abs_sin = abs(rotation_mat[0, 1])
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)
    rotation_mat[0, 2] += bound_w / 2 - image_center[0]
    rotation_mat[1, 2] += bound_h / 2 - image_center[1]
    return cv2.warpAffine(image, rotation_mat, (bound_w, bound_h))


def rotate_yolo_bbox(image, angle, rotation_angle, text):
    rot_matrix = np.array([[np.cos(rotation_angle), -np.sin(rotation_angle)],
                           [np.sin(rotation_angle), np.cos(rotation_angle)]])
    new_height, new_width = rotate_image(image, angle).shape[:2]
    new_bbox = []
    H, W = image.shape[:2]
    lines = [line for line in text.split('\n') if len(line) != 0]
    for line in lines:
        bbox = line.strip('\n').split(' ')
        (center_x, center_y, bbox_width, bbox_height) = yoloFormattocv(float(bbox[1]), float(bbox[2]),
                                                                       float(bbox[3]), float(bbox[4]), H, W)
        upper_left_corner_shift = (center_x - W / 2, -H / 2 + center_y)
        upper_right_corner_shift = (bbox_width - W / 2, -H / 2 + center_y)
        lower_left_corner_shift = (center_x - W / 2, -H / 2 + bbox_height)
        lower_right_corner_shift = (bbox_width - W / 2, -H / 2 + bbox_height)

        new_lower_right_corner = [-1, -1]
        new_upper_left_corner = []

        for i in (upper_left_corner_shift, upper_right_corner_shift,
                  lower_left_corner_shift, lower_right_corner_shift):
            new_coords = np.matmul(rot_matrix, np.array((i[0], -i[1])))
            x_prime, y_prime = new_width / 2 + new_coords[0], new_height / 2 - new_coords[1]
            if new_lower_right_corner[0] < x_prime:
                new_lower_right_corner[0] = x_prime
            if new_lower_right_corner[1] < y_prime:
                new_lower_right_corner[1] = y_prime

            if len(new_upper_left_corner) > 0:
                if new_upper_left_corner[0] > x_prime:
                    new_upper_left_corner[0] = x_prime
                if new_upper_left_corner[1] > y_prime:
                    new_upper_left_corner[1] = y_prime
            else:
                new_upper_left_corner.append(x_prime)
                new_upper_left_corner.append(y_prime)
        new_bbox.append([bbox[0], new_upper_left_corner[0], new_upper_left_corner[1],
                         new_lower_right_corner[0], new_lower_right_corner[1]])

    return new_bbox
