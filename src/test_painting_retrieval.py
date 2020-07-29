import random, argparse, cv2, os, sys

from painting_retrieval import match_paitings
from plotting import plt_images
from constants.parameters import DESTINATION_PAINTINGS_RECTIFIED, PATH_PAINTINGS_DB


def arg_parse():
    """
        Parse arguments to the Painting Detection

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", dest='num_example', help='The number of video which do you want detect',
                        default=1, type=int)
    parser.add_argument('--name', dest='name_painting', help="The path of painting rectified if you want retrieval",
                        default=None, type=str)
    return parser.parse_args()


args = arg_parse()
num_example = args.num_example
name_painting = args.name_painting

path_paintings = []

if name_painting is None:
    for root, _, file_names in os.walk(DESTINATION_PAINTINGS_RECTIFIED):
        for filename in file_names:
            if filename.lower().endswith('.jpg'):
                path_paintings.append(os.path.join(root, filename))
else:
    path_paintings.append(name_painting)

if len(path_paintings) == 0:
    sys.exit("[ERROR] Paintings not found.")

painting_choices = random.choices(path_paintings, k=num_example if num_example > 0 else len(path_paintings))

print("Start Processing ...")

try:
    for name_painting in painting_choices:
        if not os.path.isfile(name_painting):
            print('Rectified image not found')
            continue
        print("[INFO] Elaborate {} painting rectified".format(name_painting.split('/')[-1]))
        painting = cv2.imread(name_painting, cv2.IMREAD_COLOR)
        list_retrieval = match_paitings(painting)
        for best_match, similarity in list_retrieval:
            print("Match Name {} - Similarity {}".format(best_match, similarity))
        if list_retrieval is not None and len(list_retrieval) > 0:
            best_match, similarity = list_retrieval[0]
            retrieval = cv2.imread(PATH_PAINTINGS_DB + best_match, cv2.IMREAD_COLOR)
            titles = ['Painting Rectified', 'Painting Best Match']
            images = [cv2.cvtColor(painting, cv2.COLOR_BGR2RGB), cv2.cvtColor(retrieval, cv2.COLOR_BGR2RGB)]
            plt_images(images=images, titles=titles)
        else:
            print("Nothing match found")

except KeyboardInterrupt:
    print('Stop processing')
    pass
