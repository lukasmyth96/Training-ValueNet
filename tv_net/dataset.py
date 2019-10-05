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
            # TODO add something to ignore files not of the correct type
            # Create a data item for each example
            for filename in filename_list:
                filepath = os.path.join(class_dir, filename)
                data = self.load_single_example(filepath)
                data_item = DataItem(filepath=filepath, data=data, class_name=class_name)
                self.items.append(data_item)

    @staticmethod
    def load_single_example(filepath):
        """
        Load data for a single example from filepath
        Parameters
        ----------
        filepath: str
            path to data for single example
        Returns
        -------
        image:
        """

        image = skimage.io.imread(filepath)
        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        # If has an alpha channel, remove it for consistency
        if image.shape[-1] == 4:
            image = image[..., :3]

        return image

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
    def __init__(self, filepath, data, class_name):
        self.filepath = filepath
        self.filename = ntpath.basename(filepath)  # npath is compatible with any os
        self.data = data
        self.feature_vector = None
        self.class_name = class_name

        # a list to store point estimates of training-value from the MC estimation phase
        self.tv_point_estimates = list()
        self.training_value = None


def get_random_subset(dataset_object, num_items):
    """
    Return a copy of the dataset object that only contains a random subset of the original data items
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
    # TODO add ability to fix seed for shuffle to ensure same subset is used on repeat runs
    subset = copy.copy(dataset_object)
    subset.shuffle_examples()  # randomly shuffle before choosing subset
    if len(subset.items) > num_items:
        subset.items = subset.items[:num_items]

    return subset

