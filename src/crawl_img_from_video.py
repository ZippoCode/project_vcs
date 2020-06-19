import cv2
import math
import os

# # Custom importing
from read_write import get_videos

path_videos = get_videos()
for path_video in path_videos:
    path_video_arr = path_video.split('/')
    imagesFolder = "../data/video_painting/"
    cap = cv2.VideoCapture(path_video)
    frameRate = cap.get(5)  # frame rate
    count = 0
    print(path_video_arr[3] + '/' + path_video_arr[-1])
    while(cap.isOpened()):
        frameId = cap.get(1)  # current frame number
        ret, frame = cap.read()
        if (ret != True):
            break
        if (frameId % math.floor(frameRate) == 0):
            filename = imagesFolder + '/' + \
                path_video_arr[-1][:-4] + '_' + str(count) + ".jpg"
            count = count + 1
            cv2.imwrite(filename, frame)
    cap.release()
    print("Done!")


# Delete image
# count = 1
# for count_video, filename_video in enumerate(os.listdir("../data/video_painting/")):
#     for count_delete, filename_delete in enumerate(os.listdir("../data/back_up/")):
#         if filename_video == filename_delete:
#             print(count)
#             os.remove("../data/video_painting/" + filename_video)
#             count = count+1


# Rename Image
# for count, filename_video in enumerate(os.listdir("../data/video_painting/")):
#     dst = str(count+1) + ".jpg"
#     src = '../data/video_painting/' + filename_video
#     dst = '../data/video_painting/' + dst
#     os.rename(src, dst)
