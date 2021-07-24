import cv2
import sys
import os
import _pickle as pickle
import math
import skvideo
import numpy as np
from typing import List

if os.name == 'nt':
    skvideo.setFFmpegPath("../ffmpeg/bin")

import skvideo.io

# Custom importing
from parameters import *
from rotate_image import rotate_image, rotate_yolo_bbox, cvFormattoYolo


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
        for folder, _, filenames in os.walk(folder_video):
            for file in filenames:
                if file.lower().endswith(tuple(ext)):
                    path_video = os.path.join(folder, file)
                    path_videos.append(path_video)
    except FileNotFoundError:
        print('File not found into {}\nExit ...'.format(folder_video))
        sys.exit()
    print("Found {} videos".format(len(path_videos)))
    return path_videos


ext_image = ('.jpg', '.bmp', '.jpeg', '.png')


def find_image(folder: str):
    image_paths = list()
    file = None
    if not os.path.exists(folder):
        return image_paths
    try:
        for folder, _, filenames in os.walk(folder):
            for file in filenames:
                if file.lower().endswith(ext_image):
                    image_paths.append(os.path.join(folder, file))
    except FileNotFoundError:
        print(f"File {file} not found into {folder}")
        sys.exit()
    return image_paths


def save_paintings(dict_image, destination_path=DESTINATION_PAINTINGS_RECTIFIED, folder=False, filename=None):
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
        print(f'[ERROR] File {video_path} not found!')
        return []
    video_data = skvideo.io.vreader(fname=video_path)
    return video_data


def crawl_frame():
    path_videos = get_videos(folder_video=SOURCE_PATH_VIDEOS)
    for path_video in path_videos:
        name_video = os.path.split(path_video)[1]
        destination_folder = "../output/frames/"
        cap = cv2.VideoCapture(path_video)
        frame_rate = cap.get(5)  # frame rate
        count = 0
        print(f"Elaborate {name_video}")
        while cap.isOpened():
            frame_id = cap.get(1)  # current frame number
            ret, frame = cap.read()
            if not ret:
                break
            if frame_id % math.floor(frame_rate) == 0:
                filename = destination_folder + '/' + str(name_video[:-4]) + '_' + str(count) + ".jpg"
                count = count + 1
                cv2.imwrite(filename, frame)
        cap.release()
        print("Done!")


def write_output(image, text, name_image, edit_name):
    try:
        if not os.path.exists(PATH_EDIT):
            os.mkdir(PATH_EDIT)
        path_jpg = PATH_EDIT + name_image + edit_name + '.jpg'
        path_txt = PATH_EDIT + name_image + edit_name + '.txt'
        cv2.imwrite(path_jpg, image)
        with open(path_txt, 'w') as file:
            file.write(text)
            file.close()
    except OSError:
        sys.exit('Creation of the directory {} failed'.format(PATH_EDIT))


def save_detected_paintings(frames: List[np.ndarray], bounding_boxes: dict, filename: str, folder: str):
    if len(frames) == 0:
        print("Frames is empty")
        return False

    if not os.path.exists(folder):  # Create folder if it is not exist
        print(f"[INFO] Folder not found! Creating folder: {folder}")
        Path(folder).mkdir(parents=True, exist_ok=True)
    if not folder.endswith("/"):  # Check last char
        folder += "/"
    (height, width) = frames[0].shape[:2]
    name_video = os.path.splitext(os.path.basename(filename))[0]

    # Store Video
    codec = cv2.VideoWriter_fourcc(*FOURCC_NAME)
    video_writer = cv2.VideoWriter(f"{folder}{name_video}.avi", codec, FPS, (width, height))
    if video_writer.isOpened():
        for frame in frames:
            video_writer.write(frame)
    else:
        print("Error during saving the file!")
        return False
    video_writer.release()
    print(f"Video {name_video} stored! Codec: {FOURCC_NAME} - Size: {(height, width)}")

    # Save pickle file
    result_dict = dict()
    result_dict['Name file'] = name_video
    result_dict['Path video'] = filename
    result_dict['Total frame'] = len(frames)
    result_dict['Elaborated frame'] = len(bounding_boxes.items())
    result_dict['Resolution frame'] = (height, width)
    result_dict['Bounding boxes'] = bounding_boxes
    with open(f"{folder}{name_video}.pck", 'wb') as file:
        pickle.dump(result_dict, file)
    print(f"Pickle file {name_video} stored!")
    return True


def rotating_and_saving(image, angle, text, path):
    rotation_angle = angle * np.pi / 180
    rot_image = rotate_image(image, angle)
    bbox = rotate_yolo_bbox(image, angle, rotation_angle, text)
    path_jpg, path_txt = path + '.jpg', path + '.txt'
    cv2.imwrite(path_jpg, rot_image)
    if os.path.exists(path_txt):
        os.remove(path_txt)
    for i in bbox:
        with open(path_txt, 'a') as fout:
            fout.writelines(' '.join(map(str, cvFormattoYolo(i, rot_image.shape[0], rot_image.shape[1]))) + '\n')
