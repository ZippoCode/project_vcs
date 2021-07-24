import random
import argparse
import cv2
import os
import sys

from painting_retrieval import PaintingRetrieval
from plotting import plt_images
from parameters import *


def arg_parse():
    """
        Parse arguments to the Painting Detection

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", dest='num_example', help='The number of painting which do you want retrieval',
                        default=1, type=int)
    parser.add_argument('--painting', dest='name_painting',
                        help="The path of painting which you want retrieval",
                        default=None, type=str)
    parser.add_argument('--folder', dest='paintings_folder',
                        help="The path of folder which contains painting which you want retrieval",
                        default=PATH_TEST_DATASET, type=str)
    parser.add_argument('--source_db', dest='source_folder', help="The path of paintings database folder",
                        default=SOURCE_PAINTINGS_DB, type=str)
    return parser.parse_args()


args = arg_parse()
num_example = args.num_example
name_painting = args.name_painting
folder = args.paintings_folder
source_folder = args.source_folder

path_paintings = []
if args.name_painting is not None:
    path_paintings = [name_painting]
else:
    if not os.path.isdir(folder):
        print(f"[ERROR] Folder not found")
        sys.exit(-1)
    print(f"[INFO] The source folder of paintings: {folder}")
    for root, _, file_names in os.walk(folder):
        for filename in file_names:
            if filename.lower().endswith('.jpg') or filename.lower().endswith('png'):
                path_paintings.append(os.path.join(root, filename))
    path_paintings = random.sample(path_paintings, k=num_example if num_example > 0 else len(path_paintings))
    if len(path_paintings) == 0:
        sys.exit("[ERROR] Paintings not found.")

print("Start Processing Painting Retrieval ...")
print(f"[INFO] Source folder database: {source_folder}")
print(f"[INFO] Number of paintings which be elaborate: {len(path_paintings)}")

painting_retrieval = PaintingRetrieval(folder_database=SOURCE_PAINTINGS_DB)

try:
    for name_painting in path_paintings:
        if not os.path.isfile(name_painting):
            print('Rectified image not found')
            continue
        print(f"[INFO] Elaborate {os.path.split(name_painting)[1]} painting rectified")
        list_retrieval = painting_retrieval.match_painting(name_painting)
        for best_match, similarity in list_retrieval:
            print(f"Match Name {best_match} - Similarity {similarity}")
        if list_retrieval is not None and len(list_retrieval) > 0:
            best_match, similarity = list_retrieval[0]
            titles = ['Painting Rectified', 'Painting Best Match']
            painting = cv2.imread(name_painting, cv2.IMREAD_COLOR)
            retrieval = cv2.imread(f"{SOURCE_PAINTINGS_DB}{best_match}.png", cv2.IMREAD_COLOR)
            images = [cv2.cvtColor(painting, cv2.COLOR_BGR2RGB), cv2.cvtColor(retrieval, cv2.COLOR_BGR2RGB)]
            plt_images(images=images, titles=titles)
        else:
            print("Nothing match found")

except KeyboardInterrupt:
    print('Stop processing')
    pass
