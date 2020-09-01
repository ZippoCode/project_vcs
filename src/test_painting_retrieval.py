import random
import argparse
import cv2
import os
import sys

from painting_retrieval import match_paitings
from plotting import plt_images
from parameters import DESTINATION_PAINTINGS_RECTIFIED, SOURCE_PAINTINGS_DB


def arg_parse():
    """
        Parse arguments to the Painting Detection

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", dest='num_example', help='The number of video which do you want detect',
                        default=1, type=int)
    parser.add_argument('--name', dest='name_painting', help="The path of painting rectified which you want retrieval",
                        default=None, type=str)
    parser.add_argument('--folder', dest='paintings_folder',
                        help="The path of folder which contains painting rectified which you want retrieval",
                        default=DESTINATION_PAINTINGS_RECTIFIED, type=str)
    parser.add_argument('--source_db', dest='source_folder', help="The path of paintings database folder",
                        default=SOURCE_PAINTINGS_DB, type=str)
    return parser.parse_args()


args = arg_parse()
num_example = args.num_example
name_painting = args.name_painting
folder = args.paintings_folder
source_folder = args.source_folder

path_paintings = []
if name_painting:
    path_paintings.append(name_painting)
else:
    print(f"[INFO] The source folder of paintings: {folder}")
    for root, _, file_names in os.walk(folder):
        for filename in file_names:
            if filename.lower().endswith('.jpg'):
                path_paintings.append(os.path.join(root, filename))

if len(path_paintings) == 0:
    sys.exit("[ERROR] Paintings not found.")

painting_choices = random.choices(path_paintings, k=num_example if num_example > 0 else len(path_paintings))
print("Start Processing Painting Retrieval ...")
print(f"[INFO] Source folder database: {source_folder}")
print(f"[INFO] Number of paintings which be elaborate: {len(painting_choices)}")

try:
    for name_painting in painting_choices:
        if not os.path.isfile(name_painting):
            print('Rectified image not found')
            continue
        print(f"[INFO] Elaborate {os.path.split(name_painting)[1]} painting rectified")
        painting = cv2.imread(name_painting, cv2.IMREAD_COLOR)
        list_retrieval = match_paitings(painting, folder_database=source_folder)
        for best_match, similarity in list_retrieval:
            print(f"Match Name {best_match} - Similarity {similarity}")
        if list_retrieval is not None and len(list_retrieval) > 0:
            best_match, similarity = list_retrieval[0]
            retrieval = cv2.imread(SOURCE_PAINTINGS_DB + best_match, cv2.IMREAD_COLOR)
            titles = ['Painting Rectified', 'Painting Best Match']
            images = [cv2.cvtColor(painting, cv2.COLOR_BGR2RGB), cv2.cvtColor(retrieval, cv2.COLOR_BGR2RGB)]
            plt_images(images=images, titles=titles)
        else:
            print("Nothing match found")

except KeyboardInterrupt:
    print('Stop processing')
    pass
