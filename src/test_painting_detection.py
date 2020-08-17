import random, argparse, cv2, sys

from constants.parameters import SOURCE_PATH_VIDEOS, DESTINATION_PAINTINGS_DETECTED
from constants.colors import FAIL, ENDC
from painting_detection.improve_quality import multiscale_retinex
from painting_detection.painting_detection import edge_detection
from util.bounding_boxes import get_bounding_boxes
from util.plotting import plt_images
from util.edit_image import reduce_size, draw_paintings
from util.read_write import get_videos, read_video, store_video, save_pickle_file


def arg_parse():
    """
        Parse arguments to the Painting Detection
    :return:
    """
    parser = argparse.ArgumentParser(description="Process Painting Detection")
    parser.add_argument("--num", dest='num_example',
                        help='The number of videos. With -1 will be process all videos. (Default: 1)',
                        default=1, type=int)
    parser.add_argument("--show", dest='show_images',
                        help='If True you can see the results of frame (Default: False)',
                        default=False, type=bool)
    parser.add_argument("--save", dest='save_video',
                        help='If True it saves the video with painting detected (Default: False)',
                        default=False, type=bool)
    parser.add_argument("--resize", dest='resize_frame',
                        help='If True the algorithm reduces the size of frames (Default: True)',
                        default=True, type=bool)
    parser.add_argument("--source", dest='source_folder',
                        help=f'The source folder of painting detected (Default: {SOURCE_PATH_VIDEOS})',
                        default=SOURCE_PATH_VIDEOS, type=str)
    parser.add_argument("--destination", dest='destination_folder',
                        help=f"The folder where it will save the results (Default: {DESTINATION_PAINTINGS_DETECTED})",
                        default=DESTINATION_PAINTINGS_DETECTED, type=str)
    return parser.parse_args()


args = arg_parse()
num_example = args.num_example
save_flag = args.save_video
show_images = args.show_images
resize_frame = args.resize_frame
source_folder = args.source_folder
destination = args.destination_folder

print("Start Processing PAINTING DETECTION ...")

path_videos = get_videos(folder_video=source_folder)
if len(path_videos) == 0:
    print(f'{FAIL}[ERROR] Folder not found!{ENDC}')
    sys.exit(0)
path_videos = random.choices(path_videos, k=num_example if num_example > 0 else len(path_videos))
# path_videos = ['../data/videos/010/VID_20180529_112614.mp4']

print(f"[INFO] Number of videos which will be elaborated: {len(path_videos)}")
print(f"[INFO] Save Video: {save_flag}")
print(f"[INFO] Show frame elaboration: {show_images}")
print(f"[INFO] Reduce size of image: {resize_frame}")
print(f"[INFO] Folder where will store the videos: {destination}")

path_video = ''
result_dict = dict()
bounding_boxes_dict = dict()
num_frame = 0
h, w = (0, 0)

while len(path_videos) > 0:
    frame_results = []
    path_video = random.choice(path_videos)
    path_videos.remove(path_video)

    print(f"Elaborating Edge Detection for {path_video.split('/')[-1]}")
    frames = read_video(path_video)
    if frames is None:
        print(f'{FAIL}[ERROR] Frames don\'t found{ENDC}')
        continue
    try:
        for num_frame, frame in enumerate(frames):
            if resize_frame:
                frame = reduce_size(frame)
            frame_retinex = multiscale_retinex(frame)
            edit_images, edit_titles = edge_detection(frame_retinex)
            list_bounding = get_bounding_boxes(frame, edit_images[-1])
            bounding_boxes_dict[num_frame] = list_bounding
            result = draw_paintings(frame, list_bounding)
            frame_results.append(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
            if show_images:
                images = []
                titles = []
                images.append(frame)
                titles.append("Original Frame")
                images.append(frame_retinex)
                titles.append('Multi-scale Retinex')
                for image, title in zip(edit_images, edit_titles):
                    images.append(image)
                    titles.append(title)
                images.append(result)
                titles.append('Final result')
                plt_images(images, titles)
            print(f"Elaborate {num_frame + 1} frame. Found {len(list_bounding)} paintings")
    except KeyboardInterrupt:
        print(f'{FAIL}Stop processing{ENDC}')
        pass

    file_name_with_ext = path_video.split('/')[-1]
    file_name = file_name_with_ext.split('.')[0]
    if save_flag:
        store_video(file_name + '.avi', frame_results, path=destination)

    result_dict['Name file'] = file_name_with_ext
    result_dict['Path video'] = path_video
    result_dict['Total frame'] = num_frame
    result_dict['Elaborated frame'] = len(bounding_boxes_dict.items())
    result_dict['Resolution frame'] = (h, w)
    result_dict['Bounding boxes'] = bounding_boxes_dict
    save_pickle_file(result_dict, file_name, path=destination)

print("End process.")
