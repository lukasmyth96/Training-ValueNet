"""
Training-ValueNet

Base class for all configurable parameters.

Licensed under the MIT License (see LICENSE for details)
Written by Luka Smyth
"""


class Config:

    NAME = 'example'
    NUM_CLASSES = 2
    IMAGE_SIZE = 128  # must be square image of one of the following for MobileNet [96, 128, 160, 192, 224]

    # General
    EVAL_BATCH_SIZE = 32

    # Underlying Classifier
    CNN_LAYERS_TRAINABLE = True
    FINAL_CONV_FEATURE_DIM = 100  # Dimension of final conv feature layer

    # Training baseline classifier
    FINETUNE_BASELINE_CLF = True  # whether to fine-tune baseline classifier from pre-trained weights
    BASELINE_CLF_INIT_WEIGHTS = ''  # path to initialisation weights for classifier
    
    BASELINE_CLF_BATCH_SIZE = 32
    BASELINE_EARLY_STOP_PATIENCE = 5
    BASLINE_EARLY_STOP_MIN_DELTA = 0.01

    # Monte-Carlo estimation phase
    TRAIN_SUBSET_NUM_PER_CLASS = 1000
    VAL_SUBSET_NUM_PER_CLASS = 50
    MC_EPISODES = 100
    MC_EPOCHS = 1

