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
import time

from tqdm import tqdm
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from skimage.transform import resize
from keras.utils import to_categorical


class Dataset:
    def __init__(self, config, subset):

        self.config = config
        assert subset in ['train', 'val', 'test']

        self.items = []
        self.num_classes = None
        self.num_examples = None

        self.class_names = []
        self.class_names_to_one_hot = {}

        if subset == 'train':
            self.dataset_dir = config.TRAIN_DATASET_DIR
        elif subset == 'val':
            self.dataset_dir = config.VAL_DATASET_DIR

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
        class_names.sort()
        print('Found following class names: {}'.format(class_names))

        # override list of class names if a subset to use is specified
        if self.config.CLASSES_TO_USE is not None:
            class_names = self.config.CLASSES_TO_USE
            print('Using following subset of classes: {}'.format(class_names))

        self._add_classes(class_names)
        print('Loading dataset with following class names: {}'.format(self.class_names))
        time.sleep(0.2) # gives time for print before progress bar appears
        for class_name in class_names:
            class_dir = os.path.join(self.dataset_dir, class_name)
            filename_list = [f for f in os.listdir(class_dir)]
            filename_list.sort()
            # TODO add something to ignore files not of the correct type
            # Create a data item for each example
            for filename in tqdm(filename_list):
                filepath = os.path.join(class_dir, filename)
                data = self.load_single_example(filepath)
                data_item = DataItem(filepath=filepath, data=data, class_name=class_name)
                self.items.append(data_item)
        self.num_examples = len(self.items)
        logging.info('Finished loading dataset with {} examples'.format(len(self.items)))

    def load_single_example(self, filepath):
        """
        Load data for a single example from filepath
        Parameters
        ----------
        filepath: str
            path to data for single example
        Returns
        -------
        img_array: np.ndarray
        """
        img = load_img(filepath, target_size=self.config.IMG_DIMS)
        img_array = img_to_array(img)

        return img_array

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
        self._filepath = filepath
        self.filename = ntpath.basename(filepath)  # npath is compatible with any os
        self._data = data
        self._feature_vector = None
        self._class_name = class_name

        self.tv_point_estimates = list()  # to store point estimates of training-value from the MC estimation phase
        self.estimated_tv = None  # estimated training-value from MC estimation
        self.predicted_tv = None  # predicted training-value from Training-ValueNet

    @property
    def filepath(self):
        return self._filepath

    @property
    def data(self):
        return self._data

    @property
    def class_name(self):
        return self._class_name


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
        random.shuffle(class_examples)
        # take subset
        subset.items += class_examples[:num_examples_per_class]

        if len(class_examples) < num_examples_per_class:
            logging.warning('Class: {} only contains {} training examples'.format(class_name, len(class_examples)))
            logging.warning('This is less than the {} examples per class specified for the MC estimation phase'.format(
                num_examples_per_class))
            
    subset.num_examples = len(subset.items)
    return subset

