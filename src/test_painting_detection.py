import random
import sys
import cv2
import argparse
import os
import matplotlib.pyplot as plt

from parameters import SOURCE_PATH_VIDEOS, DESTINATION_PAINTINGS_DETECTED, FAIL, ENDC
from painting_detection import edge_detection
from bounding_boxes import get_bounding_boxes, convert_bounding_boxes
from plotting import plt_images
from editing_image import reduce_size, draw_paintings
from read_write import get_videos, read_video, store_video, save_pickle_file


def arg_parser():
    """
        Parse arguments to the Painting Detection
    :return:
    """
    parser = argparse.ArgumentParser(description="Process Painting Detection")
    parser.add_argument("--num", dest='num_example',
                        help='The number of videos. With -1 will be process all videos. (Default: 1)',
                        default=-1, type=int)
    parser.add_argument("--show", dest='show_images',
                        help='If True you can see the results for each frame (Default: False)',
                        default=True, type=bool)
    parser.add_argument("--save", dest='save_video',
                        help='If True it saves the video with painting detected (Default: False)',
                        default=True, type=bool)
    parser.add_argument("--resize", dest='resize_frame',
                        help='If True the algorithm reduces the size of frames (Default: True)',
                        default=False, type=bool)
    parser.add_argument("--source", dest='source_folder',
                        help=f'The source folder of painting detected (Default: {SOURCE_PATH_VIDEOS})',
                        default=SOURCE_PATH_VIDEOS, type=str)
    parser.add_argument("--destination", dest='destination_folder',
                        help=f"The folder where it will save the results (Default: {DESTINATION_PAINTINGS_DETECTED})",
                        default=DESTINATION_PAINTINGS_DETECTED, type=str)
    return parser.parse_args()


args = arg_parser()
num_example = args.num_example
save_flag = args.save_video
show_images = args.show_images
resize_frame = args.resize_frame
source_folder = args.source_folder
destination = args.destination_folder

print("Start Processing Painting Detection ...")
print("Press CTRL+C for stopping the process.")

path_videos = get_videos(folder_video=source_folder)
if len(path_videos) == 0:
    print(f'{FAIL}[ERROR] Folder not found!{ENDC}')
    sys.exit(0)
path_videos = random.sample(path_videos, k=num_example if num_example > 0 else len(path_videos))

print(f"[INFO] Number of videos which will be elaborated: {len(path_videos)}")
print(f"[INFO] Save Video: {save_flag}")
print(f"[INFO] Show frame elaboration: {show_images}")
print(f"[INFO] Reduce size of image: {resize_frame}")
print(f"[INFO] Folder where will store the videos: {destination}")

while len(path_videos) > 0:
    frame_results = []
    result_dict = dict()
    bounding_boxes_dict = dict()
    correct_format_boxes = dict()
    num_frame = 0
    h, w = (0, 0)

    path_video = random.choice(path_videos)  # Choice random video
    path_videos.remove(path_video)

    print(f"Elaborating Edge Detection for {path_video.split('/')[-1]}")
    frames = read_video(path_video)
    if frames is None:
        print(f'{FAIL}[ERROR] Frames don\'t found{ENDC}')
        continue
    try:
        for num_frame, frame in enumerate(frames):
            h, w = frame.shape[:-1]
            edit_images, edit_titles = edge_detection(frame)
            list_bounding = get_bounding_boxes(frame, edit_images[-1])
            print(f"Elaborate {num_frame + 1} frame. Found {len(list_bounding)} paintings")
            bounding_boxes_dict[num_frame] = list_bounding
            correct_format_boxes[num_frame] = convert_bounding_boxes(list_bounding)
            result = draw_paintings(frame, list_bounding)
            frame_results.append(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
            if show_images and num_frame % 50 == 0:
                images, titles = [], []
                images.append(frame)
                titles.append("Original Frame")
                for image, title in zip(edit_images, edit_titles):
                    images.append(image)
                    titles.append(title)
                images.append(result)
                titles.append('Final result')
                plt_images(images, titles)
    except KeyboardInterrupt:
        print(f'{FAIL}Stop processing{ENDC}')
        pass

    # Print Bounding Boxes
    for num_frame, list_bboxes in correct_format_boxes.items():
        print(f" Found in frame {num_frame}: {list_bboxes}")

    file_name_with_ext = os.path.split(path_video)[1]
    file_name = file_name_with_ext.split('.')[0]
    if save_flag:
        store_video(file_name + '.avi', frame_results, path=destination)

    result_dict['Name file'] = file_name_with_ext
    result_dict['Path video'] = path_video
    result_dict['Total frame'] = num_frame
    result_dict['Elaborated frame'] = len(bounding_boxes_dict.items())
    result_dict['Resolution frame'] = (h, w)
    result_dict['Bounding boxes'] = bounding_boxes_dict
    result_dict['Correct Bounding Boxes'] = correct_format_boxes
    save_pickle_file(result_dict, file_name, path=destination)

print("End process.")
