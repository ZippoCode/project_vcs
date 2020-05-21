import os
import random

# Custom importing
from read_write import capVideo, read_single_image

if __name__ == '__main__':
    #  read_single_image('../data/test.jpg')
    video_list = []
    root_path = '../data/videos'
    for forder in os.listdir(root_path):
        if forder == '.DS_Store':
            continue
        path = os.path.join(root_path, forder)
        for file in os.listdir(path):
            if file.endswith(".mp4") or file.endswith(".MP4") or file.endswith(".MOV"):
                video = os.path.join(path, file)
                video_list.append(video)

    # Take a random video and remove it from list
    # video_list = []
    # video_list.append('../data/videos/002/20180206_113059.mp4')
    while len(video_list) > 0:
        video = random.choice(video_list)
        video_list.remove(video)
        file_name = video.split('/')[-1]
        capVideo(video, '../output/' + file_name)





# if __name__ == '__main__':
#     capVideo('../data/videos/000/VIRB0406.MP4', '../output/VIRB0406.MP4')
