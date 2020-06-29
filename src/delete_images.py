import cv2
import os
import sys

ROOT_PATH_IMAGES = '../data/obj/with_statues/'

path_images = []
path_text = []
for folder in os.listdir(ROOT_PATH_IMAGES):
    if folder == '.DS_Store':
        continue
    # path = os.path.join(ROOT_PATH_IMAGES, folder)
    # for file in os.listdir(path):
    if folder.endswith(".jpg"):
        image = os.path.join(ROOT_PATH_IMAGES, folder)
        path_images.append(image)
    if folder.endswith(".txt"):
        text = os.path.join(ROOT_PATH_IMAGES, folder)
        path_text.append(text)

images = path_images
texts = []
count = 0
for image in path_images:
    for text in path_text:
        if image.split('/')[-1][:-4] == text.split('/')[-1][:-4]:
            count = count + 1
            texts.append(text)
            print(count)

for text in texts:
    path_text.remove(text)
print(path_text)
