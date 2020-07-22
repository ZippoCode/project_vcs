import random, argparse, cv2

from improve_quality import multiscale_retinex
from painting_detection import edge_detection, get_bounding_boxes
from plotting import draw_paintings, plt_images
from read_write import get_videos, read_video, store_video, save_bounding_boxes
from parameters import PATH_DESTINATION_PAINTING_DETECTED, DESTINATION_PAINTING_BBOX


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
    parser.add_argument("--dest", dest='destination', help='The folder where the list of bounding boxes will be saved',
                        default=DESTINATION_PAINTING_BBOX, type=str)
    return parser.parse_args()


args = arg_parse()
num_example = args.num_example
show_images = args.show_images
resize_frame = args.resize_frame
destination = args.destination

path_videos = get_videos()
path_videos = random.choices(path_videos, k=num_example if num_example > 0 else len(path_videos))

# path_videos = ['../data/videos/009/IMG_2655.MOV']

print("Start Processing ...")
print("[INFO] Number of video which will be elaborated: {}".format(len(path_videos)))
print("[INFO] Show frame elaboration: {}".format(show_images))
print("[INFO] Reduce size of image: {}".format(resize_frame))

path_video = ''
frame_results = []
bounding_boxes_dict = dict()

while len(path_videos) > 0:
    path_video = random.choice(path_videos)
    path_videos.remove(path_video)
    frames = read_video(path_video, reduce_size=resize_frame)
    if frames is None or len(frames) == 0:
        print("[ERROR] Frames don't found ...")
    print("Elaborating Edge Detection ...")
    try:
        for num_frame, frame in enumerate(frames):
            frame_retinex = multiscale_retinex(frame)
            edit_images, edit_titles = edge_detection(frame_retinex)
            list_bounding = get_bounding_boxes(edit_images[-1])
            bounding_boxes_dict[num_frame] = list_bounding
            result = draw_paintings(frame, list_bounding)
            frame_results.append(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
            if show_images:
                images = []
                titles = []
                images.append(frame)
                titles.append("Original Frame")
                images.append(frame_retinex)
                titles.append('Multiscale retinex')
                for image, title in zip(edit_images, edit_titles):
                    images.append(image)
                    titles.append(title)
                images.append(result)
                titles.append('Final result')

                plt_images(images, titles)
            print("[INFO] Elaborate {} of {} frames".format(num_frame + 1, len(frames)))
    except KeyboardInterrupt:
        print('Stop processing')
    pass

    file_name = path_video.split('/')[-1]
    file_name = file_name.split('.')[0]
    store_video(file_name + '.avi', frame_results, path=PATH_DESTINATION_PAINTING_DETECTED)
    save_bounding_boxes(bounding_boxes_dict, file_name, path=destination)
