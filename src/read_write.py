import cv2
from matplotlib import pyplot as plt

# Custom importing
from detection import edge_detection


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

def capVideo(video_path, name_video):
    """

    :param video_path:
    :param name_video:
    :return:
    """
    print("Reading file: {}".format((video_path)))
    cap = cv2.VideoCapture(video_path)

    width = int(cap.get(3))  # float
    height = int(cap.get(4))  # float
    # it gives me 0.0 :/

    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(name_video, fourcc,  fps, (width, height))
    if(not out.isOpened()):
        print("Error Output Video")
        return
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            images, titles = edge_detection(frame)
            out.write(cv2.cvtColor(images[-1], cv2.COLOR_BGR2RGB))
        else:
            break

    plt_images(images, titles)

    cap.release()
    print("Saving file: {}".format(name_video))
    out.release()
    cv2.destroyAllWindows()