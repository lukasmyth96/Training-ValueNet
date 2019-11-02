"""
Training-ValueNet

Script to clean dataset

Licensed under the MIT License (see LICENSE for details)
Written by Luka Smyth
"""
import os
import shutil
from tqdm import tqdm

from tv_net.utils.common import create_logger
from tv_net.config import Config
from tv_net.dataset import Dataset
from tv_net.utils.common import pickle_save
from tv_net.utils.configuration_checks import check_configuration
from tv_net.training_value_network import TrainingValueNet
from tv_net.utils.visualize import tsne_visualization, produce_tv_histograms

if __name__ == '__main__':

    # Instantiate config object
    config = Config()

    # Create output dir
    if not os.path.isdir(config.OUTPUT_DIR):
        os.makedirs(config.OUTPUT_DIR)
    if not os.path.isdir(os.path.join(config.OUTPUT_DIR, 'visualizations')):
        os.mkdir(os.path.join(config.OUTPUT_DIR, 'visualizations'))

    # Set up logger
    logger = create_logger(config.LOG_PATH)

    # Load train and val dataset objects
    train_dataset = Dataset(config, subset='train')
    val_dataset = Dataset(config, subset='val')

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
    if config.PRODUCE_TSNE:
        tsne_visualization(train_dataset, num_examples=config.TSNE_NUM_EXAMPLES)

    # Monte-Carlo Estimation Phase
    train_subset = training_value_net.mc_estimation_phase(train_dataset, val_dataset)
    # TODO figure out why there is a 'Mean of empty slice.' warning at end of above line running
    # Training each Training-ValueNet on the estimates from the MC estimation phase
    training_value_net.train_tv_nets(train_subset)

    # Predict training-value for ALL training examples using trained Training-ValueNets
    training_value_net.predict_training_values(train_dataset)
    if config.PRODUCE_TV_HISTOGRAM:
        produce_tv_histograms(train_dataset)

    # For now just save the objects
    pickle_save(os.path.join(config.OUTPUT_DIR, 'train_dataset.pkl'), train_dataset)
    
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