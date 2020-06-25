import cv2
import sys
import _pickle as pickle

# Custom importing
from parameters import *


def save_bounding_boxes(bounding_boxes_dict, name_file, path=PATH_OUTPUT_DETECTED_BBOX):
    if not os.path.exists(path):
        print('\t> Creating folder ...')
        Path(path).mkdir(parents=True, exist_ok=True)
    with open(PATH_OUTPUT_DETECTED_BBOX + name_file + '.pck', 'wb') as file:
        pickle.dump(bounding_boxes_dict, file)
    print('\t> File storage ...')


def read_bounding_boxes(filename, path=PATH_OUTPUT_DETECTED_BBOX):
    complete_path = path + filename + '.pck'
    if not os.path.exists(complete_path):
        sys.exit('File not found')
    print("\t> Reading file {}".format(filename))
    with open(complete_path, 'rb') as file:
        database = pickle.load(file)
    bounding_boxes = dict()
    for frame in database:
        bounding_boxes[frame] = database[frame]
    return bounding_boxes


def get_videos(folder_video=ROOT_PATH_VIDEOS):
    """
        Look up all the video in the ROOT FOLDER and return a list of these
    :return:
    """
    path_videos = list()
    try:
        for folder in os.listdir(folder_video):
            if folder == '.DS_Store':
                continue
            path = os.path.join(folder_video, folder)
            for file in os.listdir(path):
                if file.endswith(".mp4") or file.endswith(".MP4") or file.endswith(".MOV") or file.endswith(".jpg"):
                    path_video = os.path.join(path, file)
                    path_videos.append(path_video)
    except FileNotFoundError:
        print('File not found into {}\nExit ...'.format(folder_video))
        sys.exit()
    print("Found {} videos".format(len(path_videos)))
    return path_videos


def save_paitings(dict_image, origin_path, folders=False):
    file_name = origin_path.split('/')[-1]
    file_name = file_name.split('.')[0]
    folder = origin_path.split('/')[-2]

    if folders:
        output_path = ROOT_PATH_DETECTED + '{}/{}/'.format(folder, file_name)
    else:
        output_path = ROOT_PATH_DETECTED

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for title, image in dict_image.items():
        if folders:
            path = output_path + "{}.jpg".format(title)
        else:
            path = output_path + \
                   '{}_{}_{}.jpg'.format(folder, file_name, title)
        cv2.imwrite(path, image)


def read_video(video_path, reduce_size=True, path=ROOT_PATH_VIDEOS):
    """
        Given a path of video return a list of frame. One Frame each second.
        Each Frame is a image into RGB
    :param video_path: string - Path of video
    :param reduce_size: bool - If it is True the algorithm reduce the size of frame
    :return:
            name_video: list<numpy.ndarray>
    """
    if not os.path.exists(video_path):
        sys.exit('Error in reading file {}'.format(video_path))
    video = cv2.VideoCapture(video_path)
    print("Reading file: {}".format(video_path))
    video_frames = list()
    try:
        while video.isOpened():
            ret, frame = video.read()
            if ret:
                # cv2.imwrite('../output/frames/' + video_path.split("/")[-1] + "_{}.jpg".format(count), frame)
                if reduce_size:
                    h, w = int(frame.shape[0]), int(frame.shape[1])
                    thr_w, thr_h = 500, 500
                    if h > thr_h or w > thr_w:
                        h_ratio = thr_h / h
                        w_ratio = thr_w / w
                        w = int(frame.shape[1] * min(h_ratio, w_ratio))
                        h = int(frame.shape[0] * min(h_ratio, w_ratio))
                        frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
                video_frames.append(frame)
            else:
                break
    except KeyboardInterrupt:
        print('Exception Keyboard interrupt ...\n')

    video.release()
    print("End. Read {} frames".format(len(video_frames)))
    return video_frames


def write_video(name, frames, fps=30, fourcc_name='mp4v', path=PATH_OUTPUT):
    """
        Store the video on the file system

    :param name: string - The name of video
    :param frames: list<numpy.ndarray> - The list of frame
    :param fps: int - Frame per Seconds
    :param fourcc_name: string - The name of fourcc codec
    :param path: string - Destination of saving
    :return:
    """
    if frames is None or len(frames) == 0:
        print("Frames is not found. Return")
        return
    if not os.path.exists(path):
        Path(path).mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[0], frames[0].shape[1]
    codec = cv2.VideoWriter_fourcc(*fourcc_name)
    output = cv2.VideoWriter(path + name, codec, fps, (width, height))
    print('Storage video {} into folder {}'.format(name, path))
    if not output.isOpened():
        print("Error Output Video")
        return
    for frame in frames:
        output.write(frame)
    output.release()
    print('Ending storage.')
