import cv2

# Custom importing
from detection import edge_detection, get_bounding_boxes
from plotting import plt_images, draw_paintings
from improve_quality import multiscale_retinex
from rectification import affine_transformation


def read_single_image(input_filename, output_filename='../output/output.jpg'):
    image = cv2.imread(input_filename, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if image is None:
        print("Error reading image")
        return
    frame_retinex = multiscale_retinex(image)
    images, titles = edge_detection(frame_retinex)
    list_paintings = get_bounding_boxes(images[-1])
    out = draw_paintings(image, list_paintings)
    cv2.imwrite(output_filename, cv2.cvtColor(out, cv2.COLOR_BGR2RGB))


def elaborate_edge_detection(frame, show_images=False):
    """
        Elaborate an frame with Edge Detection and Rectification

    :param frame: numpy.ndarray with shape (H, W, C)

    :return:
        - A list of bounding boxes (x, y, w, h)
    """
    frame_retinex = multiscale_retinex(frame)
    edit_images, edit_titles = edge_detection(frame_retinex)
    list_bounding = get_bounding_boxes(edit_images[-1])

    if show_images:
        images = []
        titles = []
        # Append original frame
        images.append(frame)
        titles.append("Original Frame")
        # Append frame with Retinex elaboration
        images.append(frame_retinex)
        titles.append('Multiscale retinex')
        # Append all images from elaboration
        for image in edit_images:
            images.append(image)
        for title in edit_titles:
            titles.append(title)
        # Drawing image with the rectangle, points and line
        result = draw_paintings(frame, list_bounding)
        images.append(result)
        titles.append('Final result')
        # Show the steps of image elaboration
        plt_images(images, titles)

    return list_bounding


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
            list_boundings = elaborate_edge_detection(frame, show_images=False)
            result = draw_paintings(frame, list_boundings)
            out.write(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))

            # Step Two
            painting = []
            titles = []
            painting.append(result)
            titles.append("Detection Frame")
            for bounding, num in zip(list_boundings, range(len(list_boundings))):
                # Affine Transformation for painting
                image = affine_transformation(frame, bounding)
                painting.append(image)
                titles.append("Paiting #ID: {}".format(num))
            plt_images(painting, titles)
                #cv2.imwrite('../output/' + name_video + '.jpg', new_image)

        else:
            break

        # Show only first image from video
        break

    cap.release()
    print("Saving file: {}".format(name_video))
    out.release()
    cv2.destroyAllWindows()
