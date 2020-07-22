import random, argparse, cv2

import matplotlib.pyplot as plt

from read_write import get_videos, read_video, save_paitings, read_bounding_boxes
from painting_rectification import rectification
from parameters import DESTINATION_PAINTING_BBOX, PATH_DESTINATION_PAINTING_DETECTED


def arg_parse():
    """
        Parse arguments to the Painting Detection

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", dest='num_example', help='The number of video which do you want detect',
                        default=1, type=int)
    parser.add_argument("--show", dest='show_images', help='If True you can see the results of frame',
                        default=False, type=bool)
    parser.add_argument("--resize", dest='resize_frame', help='If True the algorithm reduce the size of frames',
                        default=True, type=bool)
    parser.add_argument("--folder", dest='folder', help='The folder where the list of bounding boxes will be saved',
                        default=DESTINATION_PAINTING_BBOX, type=str)
    return parser.parse_args()


args = arg_parse()
num_example = args.num_example
show_images = args.show_images
resize_frame = args.resize_frame
folder = args.folder

path_videos = get_videos()
path_videos = random.choices(path_videos, k=num_example if num_example > 0 else len(path_videos))

path_videos = ['../data/videos/003/GOPR1930.MP4']

print("Start Processing ...")
print("[INFO] Number of video which will be elaborated: {}".format(len(path_videos)))
print("[INFO] Show frame elaboration: {}".format(show_images))
print("[INFO] Reduce size of image: {}".format(resize_frame))

try:
    while len(path_videos) > 0:
        path_video = random.choice(path_videos)
        path_videos.remove(path_video)
        file_name = path_video.split('/')[-1]
        file_name = file_name.split('.')[0]
        bounding_boxes_dict = read_bounding_boxes(file_name, path=folder)
        frames = read_video(path_video, reduce_size=resize_frame)
        titles = []
        paintings_rectified = dict()
        for frame, (num_frame, bounding_boxes_frame) in zip(frames, bounding_boxes_dict.items()):
            for num, bounding_boxes in enumerate(bounding_boxes_frame):
                painting = rectification(frame, bounding_boxes)
                painting = cv2.cvtColor(painting, cv2.COLOR_BGR2RGB)
                name = "Painting #{} Frame{}".format(num, num_frame)
                paintings_rectified[name] = painting
        print("Saving painting rectified")
        save_paitings(paintings_rectified, path_video, folders=True)

except KeyboardInterrupt:
    print('Stop processing')
    pass

print("End process.")
