import cv2
import sys
import os
import _pickle as pickle
import skvideo

if os.name == 'nt':
    skvideo.setFFmpegPath("../ffmpeg/bin")

import skvideo.io

# Custom importing
from constants.parameters import *


def save_pickle_file(dictionary, name_file, path):
    """
        Save a dictionary with some information as pickle file

    :param dictionary: The file will be to stored
    :param name_file: The name of file
    :param path: The destination where it will save the file
    :return:
    """
    if not os.path.exists(path):
        print('[INFO] Folder Not found. Creating folder ...')
        Path(path).mkdir(parents=True, exist_ok=True)
    with open(path + name_file + '.pck', 'wb') as file:
        pickle.dump(dictionary, file)
    print(f'File {name_file} stored into folder {path}.')


def read_pickle_file(filename, path):
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


def get_videos(folder_video):
    """
        Look up all the video in the ROOT FOLDER and return a list of these
    :return:
    """
    path_videos = list()
    if not os.path.exists(folder_video):
        return path_videos
    print("[INFO] Find videos into folder: {}".format(folder_video))
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
        Given a path of video it returns a generator which is a ndarrays with shape (M, N, C).
        Where M is frame height, N is frame width and C is number of channel per pixel

    :param video_path: string - Path of video
    :return: video_data: generator
    """
    if not os.path.exists(video_path):
        print('[ERROR] File {} not found'.format(video_path))
        return []
    video_data = skvideo.io.vreader(fname=video_path)
    metadata_video = skvideo.io.ffprobe(video_path)['video']
    if '@width' in metadata_video and '@height' in metadata_video:
        resolution = (int(metadata_video['@width']), int(metadata_video['@height']))
    print('Video readed.')
    return video_data


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
