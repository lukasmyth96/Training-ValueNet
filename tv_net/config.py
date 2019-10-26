"""
Training-ValueNet

Base class for all configurable parameters.

Licensed under the MIT License (see LICENSE for details)
Written by Luka Smyth
"""

import os

class Config:

    NAME = 'example'

    TRAIN_DATASET_DIR = '/home/ubuntu/data_store/training_value_net/aircraft_7_dataset/clean_train'  # directory containing weakly-labeled training data
    VAL_DATASET_DIR = '/home/ubuntu/data_store/training_value_net/aircraft_7_dataset/val'  # directory containing validation data - must be cleanly labelled!

    OUTPUT_DIR = '/home/ubuntu/results_store/training_value_net/experiment_01'  # directory to store all output from algorithm
    LOG_PATH = os.path.join(OUTPUT_DIR, 'logs.log')

    NUM_CLASSES = 2
    CLASSES_TO_USE = ['airliner', 'helicopter']
    IMG_DIMS = (128, 128)  # must be square image of one of the following for MobileNet [96, 128, 160, 192, 224]

    # General
    EVAL_BATCH_SIZE = 32

    # Visualizations
    PRODUCE_TSNE = True  # If True a TSNE visualization will be produced of the feature vectors after baseline training
    TSNE_NUM_EXAMPLES = 1000  # Number of examples to include in T-SNE

    PRODUCE_TV_HISTOGRAM = True  # If True a histogram will be made showing distribution of predicted tvs for each class

    # Training baseline model
    BASELINE_CLF_EPOCHS = 1  # set to a large number as early stopping should prevent overfitting
    BASELINE_CLF_BATCH_SIZE = 32
    BASELINE_EARLY_STOP_PATIENCE = 1
    BASLINE_EARLY_STOP_MIN_DELTA = 0.01
    BASELINE_CLF_OPTIMIZER = 'adam'
    BASELINE_CLF_LR = 0.001

    # Monte-Carlo estimation phase
    TRAIN_SUBSET_NUM_PER_CLASS = 50
    VAL_SUBSET_NUM_PER_CLASS = 10
    MC_EPISODES = 2
    MC_EPOCHS = 2

    # Training-ValueNet architecture
    TVNET_HL1_UNITS = 100
    TVNET_HL1_DROPOUT = 0.5
    TVNET_OPTIMIZER = 'adam'  # must be a valid keras optimizer
    TVNET_LOSS = 'mean_absolute_error'  # must be a valid keras loss function for regression

    # Training hyper-params for Training-ValueNets
    TVNET_VAL_SPLIT = 0.1  # proportion of examples to be used as validation data
    TVNET_BATCH_SIZE = 32
    TVNET_EPOCHS = 10
    TVNET_EARLY_STOP_PATIENCE = 10
    TVNET_EARLY_STOP_MIN_DELTA = 0.01
