from matplotlib import pyplot as plt


def plt_images(images, titles):
    """
    Take a vector of image and a vector of title and show these on the screen

    :param images
    :param titles
    :return:
    """
    fig = plt.figure(figsize=(150, 200))
    nrows = 3
    ncols = 2
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
