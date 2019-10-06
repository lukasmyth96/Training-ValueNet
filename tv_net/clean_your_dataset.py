"""
Training-ValueNet

Script to clean dataset

Licensed under the MIT License (see LICENSE for details)
Written by Luka Smyth
"""
import logging
import os
import pickle
from tv_net.config import Config
from tv_net.dataset import Dataset
from tv_net.utils.common import pickle_save
from tv_net.training_value_network import TrainingValueNet

if __name__ == '__main__':

    # Instantiate config object
    config = Config()

    # Create output dir
    if not os.path.isdir(config.OUTPUT_DIR):
        os.makedirs(config.OUTPUT_DIR)

    # Set up logger
    logging.basicConfig(filename=os.path.join(config.OUTPUT_DIR, 'logs.log'), level=logging.INFO)

    # Load train and val dataset objects
    logging.info('Loading training set')
    train_dataset = Dataset(config.TRAIN_DATASET_DIR)
    logging.info('Loading validation set')
    val_dataset = Dataset(config.VAL_DATASET_DIR)

    if not train_dataset.class_names == val_dataset.class_names:
        raise ValueError('Mismatch between train and val class names')  # TODO should probably move all easy checks to the start

    # Create Training-ValueNets
    training_value_net = TrainingValueNet(config)
    training_value_net.build_tv_nets(train_dataset)

    # Train baseline classifier
    training_value_net.train_baseline_classifier(train_dataset, val_dataset)

    # Extract feature vectors from baseline model and store them as attribute of each example
    training_value_net.extract_feature_vectors(train_dataset)
    training_value_net.extract_feature_vectors(val_dataset)

    # Monte-Carlo Estimation Phase
    train_subset = training_value_net.mc_estimation_phase(train_dataset, val_dataset)

    # Training each Training-ValueNet on the estimates from the MC estimation phase
    training_value_net.train_tv_nets(train_subset)

    # Predict training-value for ALL training examples using trained Training-ValueNets
    training_value_net.predict_training_values(train_dataset)

    # For now just save the objects
    pickle_save(os.path.join(config.OUTPUT_DIR, 'train_dataset.pkl'), train_dataset)
    pickle_save(os.path.join(config.OUTPUT_DIR, 'training_value_net.pkl'), training_value_net)

    # TODO write something to copy or move the images into clean/dirty subsets for inspection

