import cv2
import os
import sys
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
            if file.endswith(".mp4") or file.endswith(".MP4") or file.endswith(".MOV") or file.endswith(".jpg"):
                video = os.path.join(path, file)
                path_videos.append(video)

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


def resize_when_too_big(img):
    h = int(img.shape[0])
    w = int(img.shape[1])
    thr_w, thr_h = 1080, 1080
    if h > thr_h or w > thr_w:
        h_ratio = thr_h / h
        w_ratio = thr_w / w
        ratio = min(h_ratio, w_ratio)
        img = resize_to_ratio(img, ratio)
    return img


def resize_to_ratio(img, ratio):
    """
        Resize an image according to the given ration
    :param img: Image to be resized
    :param ratio: ratio used to resize the image
    :return: Image resized
    """
    assert ratio > 0, 'ratio_percent must be > 0'
    w = int(img.shape[1] * ratio)
    h = int(img.shape[0] * ratio)
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


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
    if not os.path.exists(video_path):
        sys.exit('Error in reading file {}'.format(video_path))
    video = cv2.VideoCapture(video_path)
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
            # cv2.imwrite('../output/frames/' + video_path.split("/")[-1] + "_{}.jpg".format(count), frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = resize_when_too_big(frame)
            video_frames.append(frame)
        else:
            break
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
    if frames is None:
        print("Frames is None. Return")
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
