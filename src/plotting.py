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
    rows = int(len(images) / 2)
    cols = int(len(images) / rows) if len(images) % 2 == 0 else int(len(images) / rows) + 1
    fig, axes = plt.subplots(rows, cols)
    ax = axes.ravel()
    for img in range(len(images)):
        if len(images[img].shape) < 3:
            ax[img].imshow(images[img], cmap='gray')
        else:
            ax[img].imshow(images[img])
        # ax[img].set_title(titles[img])
        plt.tight_layout()
    plt.show()
