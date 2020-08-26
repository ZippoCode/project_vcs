import warnings
import sys
from os.path import isfile, join
from editing_image import *

warnings.simplefilter(action='ignore', category=FutureWarning)


def create_images(images, labels):
    for path_image, path_labels in zip(images, labels):
        name_image = path_image.split('.')[0]
        name_bbox = path_labels.split('.')[0]
        if name_image != name_bbox:
            print('Error file: {} does not match with {}'.format(name_image, name_bbox))
            continue

        image = cv2.imread(PATH_ORIGINAL + path_image, cv2.IMREAD_COLOR)
        with open(PATH_ORIGINAL + path_labels, 'r') as file:
            readed = file.read()

        save_edit_value_pixel(image, readed, name_image)
        save_noised_image(image, readed, name_image)
        save_filter(image, readed, name_image)
        save_rotated_image(image, readed, name_image)
        save_resized_image(image, readed, name_image)

        # OBJECT-AWAY RANDOM ERASING
        for num in range(1, 4):
            object_away_random_erasing(image, readed, name_image, '_ore{}'.format(num))

        # EOH - EDGE OBJECT HISTOGRAM
        edge_object_histogram(image, readed, name_image)

        print('Finish {}'.format(name_image))


PATH_TRAIN = '/content/gdrive/My Drive/VCS Project/Darknet/custom_data/'
PATH = '../../yolo/obj/'


def check_database():
    path_images = []
    for path, sub_dirs, files in os.walk(PATH):
        for name in files:
            if name.split('.')[-1] == 'jpg':
                path_images.append(join(path, name.split('.')[0]))
    path_images.sort()

    path_labels = []
    for path, sub_dirs, files in os.walk(PATH):
        for name in files:
            if name.split('.')[-1] == 'txt' and name.split('.')[0] != 'classes':
                path_labels.append(join(path, name.split('.')[0]))
    path_labels.sort()

    num_images = len(path_images)
    num_labels = len(path_labels)
    if num_images != num_labels:
        sys.exit('Nums don\'t match')
    print('Lengths: {} File JPG {} File TXT'.format(num_images, num_labels))

    for pi, pl in zip(path_images, path_labels):
        # print('Image {} - Labels {}'.format(pi, pl))
        if pi != pl:
            print(pi, pl)

    print("Saving train file ...")
    train = open('../../yolo/train.txt', 'w')
    for imgs in path_images:
        train.write(PATH_TRAIN + imgs + '.jpg\n')
    train.close()
    print('End.')


if __name__ == '__main__':
    images = [file for file in os.listdir(PATH_ORIGINAL) if
              isfile(join(PATH_ORIGINAL, file)) and file.split('.')[-1] == 'jpg']
    labels = [file for file in os.listdir(PATH_ORIGINAL) if
              isfile(join(PATH_ORIGINAL, file)) and file.split('.')[-1] == 'txt' and file.split('.')[0] != 'classes']
    if len(images) != len(labels):
        sys.exit('Error. Number of images {} and number of labels are different'.format(len(images), len(labels)))

    images.sort()
    labels.sort()
    create_images(images, labels)

    # Create training file
    check_database()
