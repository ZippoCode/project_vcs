import cv2, numpy as np, os, sys

from parameters import PATH_ORIGINAL, PATH_EDIT
from rotate_image import rotate_image, rotate_yolo_bbox, cvFormattoYolo


def write_output(image, text, name_image, edit_name):
    try:
        if not os.path.exists(PATH_EDIT):
            os.mkdir(PATH_EDIT)
        path_jpg = PATH_EDIT + name_image + edit_name + '.jpg'
        path_txt = PATH_EDIT + name_image + edit_name + '.txt'
        cv2.imwrite(path_jpg, image)
        with open(path_txt, 'w') as file:
            file.write(text)
            file.close()
    except OSError:
        sys.exit('Creation of the directory {} failed'.format(PATH_EDIT))


def rotating_and_saving(image, angle, text, path):
    rotation_angle = angle * np.pi / 180
    rot_image = rotate_image(image, angle)
    bbox = rotate_yolo_bbox(image, angle, rotation_angle, text)
    path_jpg, path_txt = path + '.jpg', path + '.txt'
    cv2.imwrite(path_jpg, rot_image)
    if os.path.exists(path_txt):
        os.remove(path_txt)
    for i in bbox:
        with open(path_txt, 'a') as fout:
            fout.writelines(' '.join(map(str, cvFormattoYolo(i, rot_image.shape[0], rot_image.shape[1]))) + '\n')
