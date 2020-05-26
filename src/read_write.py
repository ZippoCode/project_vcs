import cv2


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
    for frame in frames:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        output.write(frame)
    output.release()
    print('Ending storage.')
