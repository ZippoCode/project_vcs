import os
import random

from read_write import capVideo

if __name__ == '__main__':
    root_path = '../data/videos/'
    output_path = '../output/'
    videos = []
    # Generate an arrays with all videos so we can choice a random of this then
    for forder in os.listdir(root_path):
        if forder == '.DS_Store':
            continue
        path = os.path.join(root_path, forder)
        for file in os.listdir(path):
            if file.lower().endswith(".mp4") or file.lower().endswith(".mov"):
                video = os.path.join(path, file)
                videos.append(video)

    # Choices a random video
    while len(videos) > 0:
        video = random.choice(videos)
        videos.remove(video)
        paths = video.split('/')
        capVideo(video, output_path + paths[-1])
