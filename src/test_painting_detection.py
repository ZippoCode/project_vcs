import random, argparse, cv2

from improve_quality import multiscale_retinex
from painting_detection import edge_detection
from bounding_boxes import get_bounding_boxes
from plotting import draw_paintings, plt_images
from read_write import get_videos, read_video, store_video, save_bounding_boxes
from parameters import DESTINATION_VIDEOS_PAINTINGS_DETECTED, DESTINATION_PAINTING_BBOX


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

# path_videos = ['../data/videos/010/VID_20180529_112614.mp4']

print("Start Processing ...")
print("[INFO] Number of video which will be elaborated: {}".format(len(path_videos)))
print("[INFO] Show frame elaboration: {}".format(show_images))
print("[INFO] Reduce size of image: {}".format(resize_frame))

path_video = ''
result_dict = dict()
bounding_boxes_dict = dict()
num_frame = 0
h, w = (0, 0)

while len(path_videos) > 0:
    frame_results = []
    path_video = random.choice(path_videos)
    path_videos.remove(path_video)

    print("Elaborating Edge Detection for {}".format(path_video))
    frames = read_video(path_video)
    if frames is None:
        print("[ERROR] Frames don't found ...")
        continue
    try:
        for num_frame, frame in enumerate(frames):
            if resize_frame:
                h, w = int(frame.shape[0]), int(frame.shape[1])
                thr_w, thr_h = 500, 500
                if h > thr_h or w > thr_w:
                    h_ratio = thr_h / h
                    w_ratio = thr_w / w
                    w = int(frame.shape[1] * min(h_ratio, w_ratio))
                    h = int(frame.shape[0] * min(h_ratio, w_ratio))
                    frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
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
                titles.append('Multiscale retinex')
                for image, title in zip(edit_images, edit_titles):
                    images.append(image)
                    titles.append(title)
                images.append(result)
                titles.append('Final result')
                plt_images(images, titles)
            print("[INFO] Elaborate {} frame. Found {} paintings".format(num_frame + 1, len(list_bounding)))
    except KeyboardInterrupt:
        print('Stop processing')
    pass

    file_name_with_ext = path_video.split('/')[-1]
    file_name = file_name_with_ext.split('.')[0]
    store_video(file_name + '.avi', frame_results, path=DESTINATION_VIDEOS_PAINTINGS_DETECTED)

    result_dict['Name file'] = file_name_with_ext
    result_dict['Path video'] = path_video
    result_dict['Total frame'] = num_frame
    result_dict['Elaborated frame'] = len(bounding_boxes_dict.items())
    result_dict['Resolution frame'] = (h, w)
    result_dict['Bounding boxes'] = bounding_boxes_dict
    save_bounding_boxes(result_dict, file_name, path=destination)
