import random, argparse, cv2, os, numpy as np, sys

import matplotlib.pyplot as plt

from painting_retrieval import match_paitings
from parameters import ROOT_PATH_DETECTED, PATH_PAINTINGS_DB


def arg_parse():
    """
        Parse arguments to the Painting Detection

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", dest='num_example', help='The number of video which do you want detect',
                        default=4, type=int)
    return parser.parse_args()


args = arg_parse()
num_example = args.num_example

path_paitings =[]
for root, dirnames, filenames in os.walk(ROOT_PATH_DETECTED):
    for filename in filenames:
        if filename.endswith('.jpg'):
            path_paitings.append(os.path.join(root, filename))

if len(path_paitings) == 0:
    sys.exit("[ERROR] Paintings not found.")

paiting_choices = random.choices(path_paitings, k=num_example if num_example > 0 else len(path_paitings))

print("Start Processing ...")

try:
    for name_paiting in paiting_choices:
        painting = cv2.imread(name_paiting, cv2.IMREAD_COLOR)
        list_retrieval = match_paitings(painting)
        if list_retrieval is not None and len(list_retrieval) > 0:
            best_match, similarity = list_retrieval[0]
            retrieval = cv2.imread(PATH_PAINTINGS_DB + best_match, cv2.IMREAD_COLOR)
            HP, WP, CP = painting.shape
            HB, WB, CB = retrieval.shape
            result = np.empty((max(HP, HB), WP + WB, CP), np.uint8)
            result[:HP, :WP, :] = painting
            result[:HB, WP:WP + WB, :] = retrieval
            result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            plt.imshow(result)
            plt.show()
        else:
            print("Nothing match found")

except KeyboardInterrupt:
    print('Stop processing')
    pass
