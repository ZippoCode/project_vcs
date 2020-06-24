import os, sys, getopt
from os.path import join

PATH_TRAIN = '/content/gdrive/My Drive/VCS Project/Darknet/custom_data/'

if __name__ == '__main__':
    PATH = 'obj/'

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
    print('Lenghts: {} File JPG {} File TXT'.format(num_images, num_labels))

    for pi, pl in zip(path_images, path_labels):
        # print('Image {} - Labels {}'.format(pi, pl))
        if pi != pl:
            print(pi, pl)

    print("Saving train file ...")
    train = open('train.txt', 'w')
    for imgs in path_images:
        train.write(PATH_TRAIN + imgs + '.jpg\n')
    train.close()
    print('End.')


# def create_database():
#     images = []
#     for root, dirnames, files in os.walk('imgs/'):
#         for file in files:
#             if file.endswith('.png') or file.endswith('.jpg'):
#                 images.append(os.path.join(root, file))
#     images.sort()
#
#     for num, p in enumerate(images):
#         imgs = cv2.imread(p, cv2.IMREAD_COLOR)
#         name = 'negative samples/negative example_' + str(num)
#         cv2.imwrite(name + '.jpg', imgs)
#         open(name + '.txt', 'w').close()
#         print(name)
