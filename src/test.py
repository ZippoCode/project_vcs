import os
import random
from read_write import capVideo

# if __name__ == '__main__':
#     video_list = []
#     root_path = '../data/videos'
#     for forder in os.listdir(root_path):
#         if forder == '.DS_Store':
#             continue
#         path = os.path.join(root_path, forder)
#         for file in os.listdir(path):
#             if file.endswith(".mp4") or file.endswith(".MP4") or file.endswith(".MOV"):
#                 video = os.path.join(path, file)
#                 video_list.append(video)
#                 # capVideo(video, '../output/' + file)
#                 # print('Finish ...')
#     video = random.choice(video_list)
#     file_name = video.split('/')[-1]
#     print(video)
#     capVideo(video, '../output/' + file_name)

if __name__ == '__main__':
    capVideo('../data/videos/000/VIRB0406.MP4', '../output/VIRB0406.MP4')
