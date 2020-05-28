import cv2
import os
from pathlib import Path

# Custom importing
from parameters import *


def get_videos():
    """
        Look up all the video in the ROOT FOLDER and return a list of these
    :return:
    """
    path_videos = list()
    for folder in os.listdir(ROOT_PATH_VIDEOS):
        if folder == '.DS_Store':
            continue
        path = os.path.join(ROOT_PATH_VIDEOS, folder)
        for file in os.listdir(path):
            if file.endswith(".mp4") or file.endswith(".MP4") or file.endswith(".MOV"):
                video = os.path.join(path, file)
                path_videos.append(video)

    return path_videos


def save_paitings(dict_image, origin_path, folders=False):
    file_name = origin_path.split('/')[-1]
    file_name = file_name.split('.')[0]
    if folders:
        folder = origin_path.split('/')[-2]
        output_path = ROOT_PATH_DETECTED + '{}/{}/'.format(folder, file_name)
    else:
        output_path = ROOT_PATH_DETECTED + file_name + '_'

    for title, image in dict_image.items():
        path = output_path + "{}.jpg".format(title)
        cv2.imwrite(path, image)

def read_video(video_path):
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
    # Reduce the number of frames captured
    fps = int(video.get(cv2.CAP_PROP_FPS)) / 1
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
