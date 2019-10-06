"""
Training-ValueNet

Base class for all configurable parameters.

Licensed under the MIT License (see LICENSE for details)
Written by Luka Smyth
"""


class Config:

    NAME = 'example'
    TRAIN_DATASET_DIR = '/home/luka/Documents/training_value_net/aircraft_7_dataset_resized/train'  # directory containing weakly-labeled training data
    VAL_DATASET_DIR = '/home/luka/Documents/training_value_net/aircraft_7_dataset_resized/train/val'  # directory containing validation data - must be cleanly labelled!
    OUTPUT_DIR = '/home/luka/Documents/training_value_net/experiments/aircraft_experiment_01'  # directory to store all output from algorithm
    NUM_CLASSES = 7
    IMAGE_SIZE = (128, 128)  # must be square image of one of the following for MobileNet [96, 128, 160, 192, 224]

    # General
    EVAL_BATCH_SIZE = 32

    # Underlying Classifier
    FEATURE_VECTOR_SHAPE = 100  # Dimension of final conv feature layer

    # Training baseline model
    BASELINE_CLF_EPOCHS = 1000  # set to a large number as early stopping should prevent overfitting
    BASELINE_CLF_BATCH_SIZE = 32
    BASELINE_EARLY_STOP_PATIENCE = 10
    BASLINE_EARLY_STOP_MIN_DELTA = 0.01
    BASELINE_CLF_OPTIMIZER = 'adam'

    # Monte-Carlo estimation phase
    TRAIN_SUBSET_NUM_PER_CLASS = 100
    VAL_SUBSET_NUM_PER_CLASS = 10
    MC_EPISODES = 100
    MC_EPOCHS = 1

    # Training-ValueNet architecture
    TVNET_HL1_UNITS = 100
    TVNET_HL1_DROPOUT = 0.5
    TVNET_OPTIMIZER = 'adam'  # must be a valid keras optimizer
    TVNET_LOSS = 'mean_absolute_error'  # must be a valid keras loss function for regression

    # Training hyper-params for Training-ValueNets
    TVNET_VAL_SPLIT = 0.1  # proportion of examples to be used as validation data
    TVNET_BATCH_SIZE = 32
    TVNET_EPOCHS = 10000
    TVNET_EARLY_STOP_PATIENCE = 100
    TVNET_EARLY_STOP_MIN_DELTA = 0.001

