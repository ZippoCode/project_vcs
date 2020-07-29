import glob2
import math
import os
import numpy as np

PATH_DATA = 'data/paintings/'

files = []
for ext in ["*.png", "*.jpeg", "*.jpg"]:
    image_files = glob2.glob(os.path.join(PATH_DATA, ext))
    files += image_files

nb_val = math.floor(len(files) * 0.2)
rand_idx = np.random.randint(0, len(files), nb_val)

# Create train.txt file
with open("train.txt", "w") as f:
    for idx in np.arange(len(files)):
        if (os.path.exists(files[idx][:-3] + "txt")):
            f.write(files[idx] + '\n')

# Create vali.txt file
with open("val.txt", "w") as f:
    for idx in np.arange(len(files)):
        if (idx in rand_idx) and (os.path.exists(files[idx][:-3] + "txt")):
            f.write(files[idx] + '\n')
