from __future__ import print_function

import matplotlib.image as mpimg
import numpy as np
import scipy.misc
import imageio
import random
import os

from collections import namedtuple
from matplotlib import pyplot as plt
from Cityscapes_utils_label import get_labels

# Get data
root_dir = "CityScapes/"
# for imgage data
label_dir = os.path.join(root_dir, "gtFine")
train_dir = os.path.join(label_dir, "train")
val_dir = os.path.join(label_dir, "val")
test_dir = os.path.join(label_dir, "test")
# for label indexes
label_idx_dir = os.path.join(root_dir, "Labeled_idx")
train_idx_dir = os.path.join(label_idx_dir, "train")
val_idx_dir = os.path.join(label_idx_dir, "val")
test_idx_dir = os.path.join(label_idx_dir, "test")

for directory in [train_idx_dir, val_idx_dir, test_idx_dir]:
    if not os.path.exists(directory):
        os.makedirs(directory)

train_file = os.path.join(root_dir, "train.csv")
val_file = os.path.join(root_dir, "val.csv")
test_file = os.path.join(root_dir, "test.csv")

color2index = {}

# Labels list: ['name', 'id', 'trainId', 'category', 'categoryId', 'hasInstances', 'ignoreInEval', 'color']
labels = get_labels()


def parse_label():
    # change label to class index
    color2index[(0, 0, 0)] = 0  # add an void class
    for obj in labels:
        if obj.ignoreInEval:
            continue
        idx = obj.trainId
        label = obj.name
        color = obj.color
        color2index[color] = idx

    # parse train, val, test data
    for label_dir, index_dir, csv_file in zip([train_dir, val_dir, test_dir],
                                              [train_idx_dir, val_idx_dir, test_idx_dir],
                                              [train_file, val_file, test_file]):
        f = open(csv_file, "w")
        f.write("img,label\n")
        for city in os.listdir(label_dir):
            city_dir = os.path.join(label_dir, city)
            city_idx_dir = os.path.join(index_dir, city)
            data_dir = city_dir.replace("gtFine", "leftImg8bit")
            if not os.path.exists(city_idx_dir):
                os.makedirs(city_idx_dir)
            for filename in os.listdir(city_dir):
                if 'color' not in filename:
                    continue
                lab_name = os.path.join(city_idx_dir, filename)
                img_name = filename.split("gtFine")[0] + "leftImg8bit.png"
                img_name = os.path.join(data_dir, img_name)
                f.write("{},{}.npy\n".format(img_name, lab_name))

                if os.path.exists(lab_name + '.npy'):
                    print("Skip %s" % (filename))
                    continue
                print("Parse %s" % (filename))
                img = os.path.join(city_dir, filename)
                # img = scipy.misc.imread(img, mode='RGB')
                img = scipy.misc.imread(img, mode='RGB')
                height, weight, _ = img.shape

                idx_mat = np.zeros((height, weight))
                for h in range(height):
                    for w in range(weight):
                        color = tuple(img[h, w])
                        try:
                            index = color2index[color]
                            idx_mat[h, w] = index
                        except:
                            # no index, assign to void
                            idx_mat[h, w] = 19
                idx_mat = idx_mat.astype(np.uint8)
                np.save(lab_name, idx_mat)
                print("Finish %s" % (filename))


'''debug function'''


def imshow(img, title=None):
    try:
        img = mpimg.imread(img)
        imgplot = plt.imshow(img)
    except:
        plt.imshow(img, interpolation='nearest')

    if title is not None:
        plt.title(title)

    plt.show()


if __name__ == '__main__':
    parse_label()

