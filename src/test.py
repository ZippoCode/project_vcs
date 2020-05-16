import os

from read_write import capVideo

if __name__ == '__main__':
    root_path = '../data/videos'
    for forder in os.listdir(root_path):
        if forder == '.DS_Store':
            continue
        path = os.path.join(root_path, forder)
        for file in os.listdir(path):
            if file.endswith(".MP4") or file.endswith(".MP4"):
                video = os.path.join(path, file)
                capVideo(video, '../output/' + file)
                print('Finish ...')
