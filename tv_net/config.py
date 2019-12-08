"""
Training-ValueNet

Base class for all configurable parameters.

Licensed under the MIT License (see LICENSE for details)
Written by Luka Smyth
"""

import os
from datetime import datetime


class Config:

    NAME = 'example'

    TRAIN_DATASET_DIR = '/home/ubuntu/data_store/training_value_net/aircraft_7_dataset_224/train'  # directory containing weakly-labeled training data
    VAL_DATASET_DIR = '/home/ubuntu/data_store/training_value_net/aircraft_7_dataset_224/val'  # directory containing validation data - must be cleanly labelled!

    timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    OUTPUT_DIR = os.path.join('/home/ubuntu/results_store/training_value_net/experiment_01', timestamp)  # directory to store all output from algorithm
    LOG_PATH = os.path.join(OUTPUT_DIR, 'logs.log')

    NUM_CLASSES = 7
    CLASSES_TO_USE = None  # Specify subset to use - set to None to use all
    IMG_DIMS = (224, 224)  # must be square image of one of the following for MobileNet [96, 128, 160, 192, 224]

    # General
    EVAL_BATCH_SIZE = 32

    # Visualizations
    PRODUCE_TSNE = True  # If True a TSNE visualization will be produced of the feature vectors after baseline training
    TSNE_NUM_EXAMPLES = NUM_CLASSES * 1000  # Number of examples to include in T-SNE (to keep run-time reasonable)

    PRODUCE_TV_HISTOGRAM = True  # If True a histogram will be made showing distribution of predicted tvs for each class

    # Baseline Classifier - either trained or load a pre-trained model
    TRAIN_BASELINE_CLF = True  # If True then LOAD_BASELINE_CLF must be False (and vice-versa)
    BASELINE_CLF_EPOCHS = 10  # set to a large number as early stopping should prevent overfitting
    BASELINE_CLF_BATCH_SIZE = 32
    BASELINE_CLF_LR = 0.001
    BASELINE_CLF_LR_DECAY = 1e-4
    BASELINE_CLF_MOMENTUM = 0.9
    BASELINE_CLF_NESTEROV = True
    BASELINE_EARLY_STOP_PATIENCE = 1
    BASELINE_EARLY_STOP_METRIC = 'val_acc'  # which metric to use for early stopping - 'val_loss' or 'val_acc'
    BASELINE_EARLY_STOP_MIN_DELTA = 0.005

    LOAD_BASELINE_CLF = False   # If True then TRAIN_BASELINE_CLF must be False (and vice-versa)
    BASELINE_CLF_WEIGHTS = '/home/ubuntu/data_store/training_value_net/aircraft_7_baseline_resnet.h5'

    # Monte-Carlo estimation phase
    TRAIN_SUBSET_NUM_PER_CLASS = 1000
    VAL_SUBSET_NUM_PER_CLASS = 100
    MC_EPISODES = 100
    MC_EPOCHS = 1
    MC_LR = 0.001
    MC_CHECKPOINT_EVERY = 10  # number of episodes between checkpoints

    # Training-ValueNet architecture
    TVNET_HL1_UNITS = 100
    TVNET_HL1_DROPOUT = 0.5
    TVNET_OPTIMIZER = 'adam'  # must be a valid keras optimizer
    TVNET_LOSS = 'mean_absolute_error'  # must be a valid keras loss function for regression

    # Training hyper-params for Training-ValueNets
    TVNET_VAL_SPLIT = 0.1  # proportion of examples to be used as validation data
    TVNET_BATCH_SIZE = 32
    TVNET_EPOCHS = 100
    TVNET_EARLY_STOP_PATIENCE = 10
    TVNET_EARLY_STOP_MIN_DELTA = 0.01
