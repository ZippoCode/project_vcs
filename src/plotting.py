import cv2
from matplotlib import pyplot as plt


def plt_images(images, titles):
    """
        Take a vector of image and a vector of title and show these on the screen

    :param images
    :param titles
    :return:
    """
    if len(images) != len(titles) or len(images) > 20:
        print("Too images. Edit this file")
        return
    rows = int(len(images) / 3)
    cols = int(len(images) / rows) if len(images) % 3 == 0 else int(len(images) / rows) + 1
    fig, axes = plt.subplots(rows, cols)
    ax = axes.ravel()
    for img in range(len(images)):
        if len(images[img].shape) < 3:
            ax[img].imshow(images[img], cmap='gray')
        else:
            ax[img].imshow(images[img])
        #ax[img].set_title(titles[img])
        plt.tight_layout()
    plt.show()


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
    if len(image.shape) != 3:
        return image

    image_painting = image.copy()
    color_green = (0, 255, 0)
    color_red = (255, 0, 0)
    color_blue = (0, 0, 255)
    color_yellow = (0, 255, 255)

    for painting in list_painting:
        upper_left, upper_right, down_left, down_right = painting
        cv2.line(image_painting, upper_left, upper_right, color_green, 3)
        cv2.line(image_painting, upper_left, down_left, color_green, 3)
        cv2.line(image_painting, down_left, down_right, color_green, 3)
        cv2.line(image_painting, upper_right, down_right, color_green, 3)

        # Write rectangle -> Useless
        # x, y = upper_left
        # w = upper_right[0] - x
        # h = down_left[1] - y
        # cv2.rectangle(image_painting, (x, y), (x + w, y + h), color_red, 3)

        cv2.circle(image_painting, upper_left, 10, color_green, -1)
        cv2.circle(image_painting, down_left, 10, color_red, -1)
        cv2.circle(image_painting, upper_right, 10, color_blue, -1)
        cv2.circle(image_painting, down_right, 10, color_yellow, -1)

    return image_painting
