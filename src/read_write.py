import cv2

# Custom importing
from detection import edge_detection


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
    out = cv2.VideoWriter(name_video, fourcc, fps, (width, height))
    if (not out.isOpened()):
        print("Error Output Video")
        return
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            list_painting = edge_detection(frame)
            for painting in list_painting:
                x, y, w, h = painting
                drawing_frame = cv2.rectangle(
                    frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                out.write(cv2.cvtColor(drawing_frame, cv2.COLOR_BGR2RGB))
        else:
            break

        # Show only first image from video
        break

    cap.release()
    print("Saving file: {}".format(name_video))
    out.release()
    cv2.destroyAllWindows()
