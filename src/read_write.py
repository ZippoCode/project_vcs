import cv2
import sys
import _pickle as pickle
import skvideo.io

# Custom importing
from parameters import *


def save_bounding_boxes(bounding_boxes_dict, name_file, path=PATH_OUTPUT_DETECTED_BBOX):
    if not os.path.exists(path):
        print('[INFO] Creating folder ...')
        Path(path).mkdir(parents=True, exist_ok=True)
    with open(path + name_file + '.pck', 'wb') as file:
        pickle.dump(bounding_boxes_dict, file)
    print('File storage ...')


def read_bounding_boxes(filename, path=PATH_OUTPUT_DETECTED_BBOX):
    complete_path = path + filename + '.pck'
    bounding_boxes = dict()
    if not os.path.exists(complete_path):
        print('[ERROR] File not found')
        return bounding_boxes
    with open(complete_path, 'rb') as file:
        database = pickle.load(file)
    for frame in database:
        bounding_boxes[frame] = database[frame]
    return bounding_boxes


ext = (".mp4", ".mov", ".avi")


def get_videos(folder_video=ROOT_PATH_VIDEOS):
    """
        Look up all the video in the ROOT FOLDER and return a list of these
    :return:
    """
    path_videos = list()
    try:
        for folder, subfolder, filenames in os.walk(folder_video):
            for file in filenames:
                if file.lower().endswith(tuple(ext)):
                    path_video = os.path.join(folder, file)
                    path_videos.append(path_video)
    except FileNotFoundError:
        print('File not found into {}\nExit ...'.format(folder_video))
        sys.exit()
    print("Found {} videos".format(len(path_videos)))
    return path_videos


def save_paitings(dict_image, destination_path=DESTINATION_PAINTINGS_RECTIFIED, folder=False, filename=None):
    """

    :param dict_image:
    :param destination_path:
    :param folder:
    :param filename:
    :return:
    """
    if folder and filename is None:
        print("If you want save a specific format you need pass the filename")
        return False
    if folder:
        destination_path = destination_path + '{}/'.format(filename)
    if not os.path.exists(destination_path):
        os.makedirs(destination_path)
    for title, image in dict_image.items():
        path = destination_path + '{}.jpg'.format(title)
        cv2.imwrite(path, image)
    return True


def read_video(video_path):
    """
        Given a path of video return a list of frame. One Frame each second.
        Each Frame is a image into RGB
    :param video_path: string - Path of video
    :param reduce_size: bool - If it is True the algorithm reduce the size of frame

    :return:
            name_video: list<numpy.ndarray>
    """
    if not os.path.exists(video_path):
        print('[ERROR] File {} not found'.format(video_path))
        return []
    videodata = skvideo.io.vreader(video_path)
    print('Video readed.')
    return videodata


def store_video(name, frames, fps=30, fourcc_name='MJPG', path=PATH_OUTPUT):
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
    print("[INFO] Storage video {} into folder {}".format(name, path))
    print("[INFO] Codec: {} - Size: {}".format(fourcc_name, (height, width)))
    if not output.isOpened():
        print("Error Output Video")
        return
    for frame in frames:
        output.write(frame)
    output.release()
    print('Ending storage.')
