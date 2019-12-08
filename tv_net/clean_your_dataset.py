"""
Training-ValueNet

Script to clean dataset

Licensed under the MIT License (see LICENSE for details)
Written by Luka Smyth
"""

import os
import shutil
from tqdm import tqdm


from tv_net.utils.common import initialise_logger
from tv_net.config import Config
from tv_net.dataset import Dataset
from tv_net.utils.common import pickle_save
from tv_net.utils.configuration_checks import check_configuration
from tv_net.training_value_network import TrainingValueNet
from tv_net.utils.visualize import tsne_visualization, produce_tv_histograms
from tv_net.utils.evaluate_cleaning_performance import evaluate_cleaning_performance


if __name__ == '__main__':

    # Instantiate config object
    config = Config()
    check_configuration(config)  # check that config is valid

    # Create output dir
    os.makedirs(config.OUTPUT_DIR)
    os.mkdir(os.path.join(config.OUTPUT_DIR, 'visualizations'))
    pickle_save(os.path.join(config.OUTPUT_DIR, 'config.pkl'), config)  # save config

    # Set up module_logger
    module_logger = initialise_logger(config.LOG_PATH)

    module_logger.info(' \n Starting Label Cleaning Process - Output will be saved to {}'.format(config.OUTPUT_DIR))

    # Load train and val dataset objects
    train_dataset = Dataset(config, subset='train')
    val_dataset = Dataset(config, subset='val')

    # Create Training-ValueNets
    training_value_net = TrainingValueNet(config)
    training_value_net.build_tv_nets(train_dataset)

    # Train baseline classifier or Load pre-trained baseline
    if config.TRAIN_BASELINE_CLF:
        training_value_net.train_baseline_classifier(train_dataset, val_dataset)
    elif config.LOAD_BASELINE_CLF:
        training_value_net.load_baseline_classifier(config.BASELINE_CLF_WEIGHTS)

    # Extract feature vectors from baseline model and store them as attribute of each example
    training_value_net.extract_feature_vectors(train_dataset)
    pickle_save(os.path.join(config.OUTPUT_DIR, 'train_dataset.pkl'), train_dataset)
    training_value_net.extract_feature_vectors(val_dataset)
    pickle_save(os.path.join(config.OUTPUT_DIR, 'val_dataset.pkl'), val_dataset)
    if config.PRODUCE_TSNE:
        tsne_visualization(train_dataset, num_examples=config.TSNE_NUM_EXAMPLES)

    # Monte-Carlo Estimation Phase
    train_subset = training_value_net.mc_estimation_phase(train_dataset, val_dataset)
    pickle_save(os.path.join(config.OUTPUT_DIR, 'train_subset.pkl'), train_subset)

    # Training each Training-ValueNet on the estimates from the MC estimation phase
    training_value_net.train_tv_nets(train_subset)

    # Predict training-value for ALL training examples using trained Training-ValueNets
    module_logger.info('Predicting training-values for all examples...')
    training_value_net.predict_training_values(train_dataset)
    if config.PRODUCE_TV_HISTOGRAM:
        produce_tv_histograms(train_dataset)

    # Copy images into clean and dirty folders - TODO This should move to it's own function
    clean_dir = os.path.join(config.OUTPUT_DIR, 'clean_training_examples')
    os.mkdir(clean_dir)
    dirty_dir = os.path.join(config.OUTPUT_DIR, 'dirty_training_examples')
    os.mkdir(dirty_dir)
    
    for class_name in train_dataset.class_names:
        os.mkdir(os.path.join(clean_dir, class_name))
        os.mkdir(os.path.join(dirty_dir, class_name))
    
    for item in tqdm(train_dataset.items):
        filename = os.path.basename(item.filepath)
        new_filename = 'tv={:.2E}_{}'.format(item.predicted_tv, filename)
        
        if item.predicted_tv > 0:
            dest = os.path.join(clean_dir, item.class_name, new_filename) 
        else:
            dest = os.path.join(dirty_dir, item.class_name, new_filename) 

        shutil.copy(item.filepath, dest)

    #  For now just save the objects
    pickle_save(os.path.join(config.OUTPUT_DIR, 'train_dataset.pkl'), train_dataset)


