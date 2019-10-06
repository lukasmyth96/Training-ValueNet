"""
Training-ValueNet

Base class for dataset and data item.

Licensed under the MIT License (see LICENSE for details)
Written by Luka Smyth
"""

import copy
import logging
import ntpath
import os
import random
import re
from tqdm import tqdm

import skimage.io
import skimage.color
from keras.utils import to_categorical


class Dataset:
    def __init__(self, dataset_dir):
        self.items = []
        self.num_classes = None
        self.num_examples = None
        self.class_names = []
        self.class_names_to_one_hot = {}

        self.dataset_dir = dataset_dir
        self.load_dataset()

    def load_dataset(self):
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
        """

        class_names = [directory for directory in os.listdir(self.dataset_dir)]
        self._add_classes(class_names)
        logging.info('Loading dataset with following class names: {}'.format(self.class_names))

        for class_name in class_names:
            class_dir = os.path.join(self.dataset_dir, class_name)
            filename_list = [f for f in os.listdir(class_dir)]
            # TODO add something to ignore files not of the correct type
            # Create a data item for each example
            for filename in tqdm(filename_list[:1000]):
                filepath = os.path.join(class_dir, filename)
                data = self.load_single_example(filepath)
                data_item = DataItem(filepath=filepath, data=data, class_name=class_name)
                self.items.append(data_item)
        self.num_examples = len(self.items)
        logging.info('Finished loading dataset with {} examples'.format(len(self.items)))

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

        class_names.sort()  # sort alphabetically
        self.class_names = [self._clean_class_name(name) for name in class_names]  # clean names
        self.num_classes = len(class_names)

        # Create dict mapping class name to one hot vector
        self.class_names_to_one_hot = {name: to_categorical(idx, self.num_classes)
                                       for idx, name in enumerate(class_names)}

    @staticmethod
    def _clean_class_name(class_name):
        """ Returns cleaner version of class name"""
        class_name = class_name.lower()
        class_name = class_name.strip()
        class_name = re.sub(' +', '_', class_name)
        return class_name

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

        self.tv_point_estimates = list()  # to store point estimates of training-value from the MC estimation phase
        self.estimated_tv = None  # estimated training-value from MC estimation
        self.predicted_tv = None  # predicted training-value from Training-ValueNet


def get_random_subset(dataset_object, num_examples_per_class):
    """
    Return a copy of the dataset object that only contains a random subset of the original data items
    Parameters
    ----------
    dataset_object: tv_net.dataset.Dataset
        instance of the dataset class to take subset from
    num_examples_per_class: int
        number of examples to include in subset

    Returns
    -------
    subset: tv_net.dataset.Dataset
        new dataset object that contains only a random subset
    """
    subset = copy.copy(dataset_object)
    subset.items = list()  # re-initialise to empty list
    for class_name in dataset_object.class_names:
        # Get a list of the examples belonging to this class
        class_examples = [example for example in dataset_object.items if example.class_name == class_name]

        # take subset
        subset.items += class_examples[:num_examples_per_class]

        if len(class_examples) < num_examples_per_class:
            logging.warning('Class: {} only contains {} training examples'.format(class_name, len(class_examples)))
            logging.warning('This is less than the {} examples per class specified for the MC estimation phase'.format(
                num_examples_per_class))

    return subset

