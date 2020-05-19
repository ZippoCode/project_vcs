import numpy as np
import cv2
import math

from detection import edge_detection


def recfitication(im):
    filename = im
    img, gray = read_undistorted_image_color_grayscale(filename)
    # if DEBUG is True:
    #     show(img, 'original')
    gray = cv2.GaussianBlur(
        gray, BLURRING_GAUSSIAN_KERNEL_SIZE, BLURRING_GAUSSIAN_SIGMA)
    components, gray = image_segmentation(gray)
    global_mask = np.zeros_like(gray, dtype=np.uint8)

    img_segm = np.zeros_like(img)
    img_segm[:, :] = SEGMENTATION_COLOR_BG
    i = 0
    for component in components:
        is_contained, global_mask = component.check_if_contained_in_another_component(
            global_mask)

        # if DEBUG is True:
        #     show(component.mask, 'mask component')

        if is_contained is True:
            continue
        if check_if_picture(img, gray, component.mask) is False:
            continue
        else:
            global_mask[component.mask == 255] = 255

        image_vertices, real_vertices = component.get_vertices(gray)
        if image_vertices is None:
            continue

        if len(image_vertices) == 4:
            sorted_vertices = sort_corners(image_vertices)
            # if DEBUG is True:
            #     show_vertices(img, sorted_vertices,
            #                   'sorted_vertices', with_order=True)

            if DEBUG is True:
                show_rectangle(img, sorted_vertices)


# recfitication('../data/Amalia_image.jpeg')

im = cv2.imread('../data/Amalia_image.jpeg')
list_painting = edge_detection(im)
print(list_painting)
for painting in list_painting:
    x, y, w, h = painting
    drawing_frame = cv2.rectangle(
        im, (x, y), (x + w, y + h), (0, 255, 0), 3)
    cv2.imshow('Edge Detection', im)
    cv2.waitKey()
