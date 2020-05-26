import cv2

<<<<<<< HEAD
# Custom importing
from detection import edge_detection, get_bounding_boxes
from plotting import plt_images, draw_paintings
from improve_quality import multiscale_retinex
from rectification import rectification
=======
>>>>>>> e2d1a793d9754470e13078d66525579f39bd19cf

# def read_single_image(input_filename, output_filename='../output/output.jpg'):
#     image = cv2.imread(input_filename, cv2.IMREAD_COLOR)
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     if image is None:
#         print("Error reading image")
#         return
#     frame_retinex = multiscale_retinex(image)
#     images, titles = edge_detection(frame_retinex)
#     list_paintings = get_bounding_boxes(images[-1])
#     out = draw_paintings(image, list_paintings)
#     cv2.imwrite(output_filename, cv2.cvtColor(out, cv2.COLOR_BGR2RGB))


def read_video(video_path, name_video):
    """
        Given a path of video return a list of frame. One Frame each second.
        Each Frame is a image into RGB
    :param
        video_path: string
    :param
    :return:
            name_video: list<numpy.ndarray>
    """
<<<<<<< HEAD
    frame_retinex = multiscale_retinex(frame)
    # plt.imshow(frame_retinex)
    # plt.show()
    # plt_images([frame, frame_retinex], ['frame', 'frame_retinex'])
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
=======
    video = cv2.VideoCapture(video_path)
    if video is None or not video.isOpened():
        print('Error reading file {}'.format(video_path))
    print("Reading file: {}".format(video_path))
    fps = int(video.get(cv2.CAP_PROP_FPS)) / 1             # Reduce the number of frames captured
    count = 0
    video_frames = list()
    while video.isOpened():
        ret, frame = video.read()
        count += fps
        video.set(1, count)
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_frames.append(frame)
        else:
            break
    video.release()
    print("End. Read {} frames".format(len(video_frames)))
    return video_frames
>>>>>>> e2d1a793d9754470e13078d66525579f39bd19cf


def write_video(name, frames):
    """
        Store the video on the file system

    :param name: string - The name of video
    :param frames: list<numpy.ndarray> - The list of frame
    :return:
    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    height, width, channel = frames[0].shape
    output = cv2.VideoWriter(name, fourcc, 15.0, (width, height))
    print('Storage video {}'.format(name))
    if not output.isOpened():
        print("Error Output Video")
        return
<<<<<<< HEAD
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            list_boundings = elaborate_edge_detection(frame, show_images=False)
            result = draw_paintings(frame, list_boundings)
            out.write(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))

            # Step Two
            paintings = []
            titles = []
            paintings.append(result)
            titles.append("Detection Frame")
            for bounding, num in zip(list_boundings, range(len(list_boundings))):
                # Affine Transformation for painting
                painting = rectification(frame, bounding)
                paintings.append(painting)
                titles.append("Painting #ID: {}".format(num))
                # Take the only name of file
                name = name_video.split('.')[0]
                # The swapaxes is utile for swap R and B
                cv2.imwrite(path_painting.format(name, num),
                            cv2.cvtColor(painting, cv2.COLOR_BGR2RGB))
            plt_images(paintings, titles)

        else:
            break

        # Show only first image from video
        break

    cap.release()
    print("Saving file: {}".format(path_video))
    out.release()
=======
    for frame in frames:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        output.write(frame)
    output.release()
    print('Ending storage.')
>>>>>>> e2d1a793d9754470e13078d66525579f39bd19cf
