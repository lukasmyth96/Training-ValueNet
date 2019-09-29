"""
Training-ValueNet

Base class for dataset and data item.

Licensed under the MIT License (see LICENSE for details)
Written by Luka Smyth
"""

import copy
import ntpath
import os
import random

import skimage.io
import skimage.color
from keras.utils import to_categorical

class Dataset:
    def __init__(self):
        self.items = []
        self.num_classes = None
        self.class_names = []
        self.class_names_to_one_hot = {}

    def load_dataset(self, data_dir):
        """
        Load specified dataset subset

        assumes dataset is structured like

        > dataset
            > class_1
                example_1.jpg
                .
                .
            > class_1
            .
            etc.

        Parameters
        ----------
        data_dir: str
            directory containing dataset
        """

        class_names = [dir for dir in os.listdir(data_dir)]
        self._add_classes(class_names)

        for class_name in class_names:
            class_dir = os.path.join(data_dir, class_name)
            filename_list = [f for f in os.listdir(class_dir)]

            # Create a data item for each example
            for filename in filename_list:
                filepath = os.path.join(data_dir, filename)
                data_item = DataItem(filepath=filepath, class_name=class_name)
                self.items.append(data_item)

    def load_images(self):
        """
        Loads images from filepaths
        """
        for item in self.items:
            image = skimage.io.imread(item.filepath)
            # If grayscale. Convert to RGB for consistency.
            if image.ndim != 3:
                image = skimage.color.gray2rgb(image)
            # If has an alpha channel, remove it for consistency
            if image.shape[-1] == 4:
                image = image[..., :3]

            item.image = image

    def _add_classes(self, class_names):
        """
        Add classes to dataset
        Parameters
        ----------
        class_names: list[str]
            list of class names
        """

        self.class_names = class_names
        self.num_classes = len(class_names)

        # Create dict mapping class name to one hot vector
        self.class_names_to_one_hot = {name: to_categorical(idx, self.num_classes)
                                       for idx, name in enumerate(class_names)}

    def shuffle_examples(self):
        """
        Randomly shuffle list of data items
        """
        random.shuffle(self.items)


class DataItem:
    def __init__(self, filepath, class_name):
        self.filepath = filepath
        self.filename = ntpath.basename(filepath)  # compatible with any os
        self.image = None
        self.class_name = class_name
        self.feature_vector = None
        self.training_value = None


def get_random_subset(dataset_object, num_items):
    """
    Return a new dataset object that contains a random subset of the original data items
    Parameters
    ----------
    dataset_object: tv_net.dataset.Dataset
        instance of the dataset class to take subset from
    num_items: int
        number of examples to include in subset

    Returns
    -------
    subset: tv_net.dataset.Dataset
        new dataset object that contains only a random subset
    """
    subset = copy.copy(dataset_object)
    subset.shuffle_examples()  # randomly shuffle before choosing subset
    if len(subset.items) > num_items:
        subset.items = subset.items[:num_items]

    return subset