"""
Training-ValueNet

TrainingValueNet class encapsulates all functionality for algorithm.

Licensed under the MIT License (see LICENSE for details)
Written by Luka Smyth
"""
import copy
import math
import numpy as np
import os

from tqdm import tqdm

from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping

from tv_net.utils.common import create_logger
from tv_net.classifier import Classifier
from tv_net.dataset import get_random_subset


class TrainingValueNet:
    """
    Encapsulates all functionality of Training-ValueNet
    """
    def __init__(self, config):
        """
        Parameters
        ----------
        config: tv_net.config.Config
            configuration object
        """
        # TODO for missing config params at this stage
        self.config = config
        self.logger = create_logger(config.LOG_PATH)
        self.classifier = Classifier(config)
        self.tv_nets = dict()
        self._set_log_dir()

    def build_tv_nets(self, dataset_object):
        """
        Build one Training-ValueNet per class
        Each T-VNet is a MLP regression network
        Each network is stored as a value in a dictionary where the its class name is the key
        Parameters
        -------
        dataset_object: tv_net.dataset.Dataset
            dataset object to build T-VNets for - contains information about class names and feature vector size
        Returns
        -------
        tv_nets: dict[str, keras.engine.sequential.Sequential]
            dictionary where class name is key and T-VNet for that class is the value
        """
        # Extract class names and feature vector shape from dataset
        class_names = dataset_object.class_names
        input_dim = self.classifier.feature_extractor.output.shape[1].value

        # Extract architecture params from config
        hl1_units = self.config.TVNET_HL1_UNITS
        hl1_drop = self.config.TVNET_HL1_DROPOUT

        tv_nets = {class_name: None for class_name in class_names}  # dict to store class names
        # Create a separate Training-ValueNet for each class
        for class_name in class_names:
            tv_net = Sequential()
            tv_net.add(Dense(hl1_units, input_dim=input_dim, activation='relu'))
            tv_net.add(Dropout(hl1_drop))
            tv_net.add(Dense(1, activation='linear'))

            tv_net.compile(loss=self.config.TVNET_LOSS, optimizer=self.config.TVNET_OPTIMIZER)

            # Add Training-ValueNet to dictionary
            tv_nets[class_name] = tv_net

        self.tv_nets = tv_nets
        self.logger.info('Finished building Training-ValueNetworks for classes: {}'.format(class_names))

    def train_baseline_classifier(self, train_dataset, val_dataset):
        """
        Train baseline in order to extract feature vectors for each example
        Parameters
        ----------
        train_dataset: tv_net.dataset.Dataset
            training dataset object
        val_dataset: tv_net.dataset.Dataset
            validation dataset object
        """
        self.logger.info('Training baseline classification model')
        self.logger.info('Training on {} examples'.format(train_dataset.num_examples))
        self.logger.info('Evaluating on {} examples'.format(val_dataset.num_examples))
        self.classifier.train_baseline(train_dataset, val_dataset)

    def extract_feature_vectors(self, dataset_object):
        """
        Extract feature vector from train baseline classifier
        Feature vectors will be store as an attribute of each data item
        Parameters
        ----------
        dataset_object: tv_net.dataset.Dataset
        """
        self.logger.info('Extracting feature vectors')
        batch_size = self.config.EVAL_BATCH_SIZE
        steps = math.ceil(dataset_object.num_examples / batch_size)
        # Shuffle must be False!
        batch_generator = self.classifier.batch_generator(dataset_object, shuffle=False, batch_size=batch_size)
        feature_vector_array = self.classifier.feature_extractor.predict_generator(batch_generator, steps=steps, verbose=1)
        feature_vector_array = feature_vector_array[:dataset_object.num_examples]
        assert feature_vector_array.shape[0] == dataset_object.num_examples

        # loop through examples and add feature vector as an attribute
        for idx, example in enumerate(dataset_object.items):
            example.feature_vector = feature_vector_array[idx]

    def mc_estimation_phase(self, train_dataset, val_dataset):
        """
        Monte-Carlo estimation phase.
        Here we estimate the training-values of a subset of training examples from each class.
        These estimates are then used at regression targets to train the Training-ValueNet.
        Parameters
        ----------
        train_dataset: tv_net.dataset.Dataset
            training dataset object
        val_dataset: tv_net.dataset.Dataset
            validation dataset object
        """

        # Get random subset of train and val to estimate on
        train_subset = get_random_subset(train_dataset, self.config.TRAIN_SUBSET_NUM_PER_CLASS)
        val_subset = get_random_subset(val_dataset, self.config.VAL_SUBSET_NUM_PER_CLASS)
        
        # Compile classification head
        classification_head = self.classifier._build_classification_head()
        self.classifier.compile_classifier(classification_head)
        
        for episode in range(self.config.MC_EPISODES):
            self.logger.info('Starting episode: {} of MC estimation phase'.format(episode + 1))
            
            # Re-initialize classification head weights at start of each episode 
            classification_head = self.classifier.reinitialize_classification_head(classification_head)  
            
            # Shuffle training examples
            train_subset.shuffle_examples()  
            
            # Compute and store initial validation loss
            val_losses = list()  # Create list to store val losses
            val_losses.append(self.classifier.compute_loss_on_dataset_object(classification_head, val_subset))  # append initial val loss

            iteration = 0
            for iteration in tqdm(range(self.config.MC_EPOCHS * train_subset.num_examples)):

                # Get next item to train on
                item = train_subset.items[iteration % train_subset.num_examples]
                feature_vector = copy.copy(item.feature_vector)
                feature_vector = np.expand_dims(feature_vector, axis=0)
                one_hot_label = train_subset.class_names_to_one_hot[item.class_name]
                one_hot_label = np.expand_dims(one_hot_label, axis=0)

                # Train classification head on example
                classification_head.train_on_batch(feature_vector, one_hot_label)

                # Compute immediate improvement in val loss
                new_val_loss = self.classifier.compute_loss_on_dataset_object(classification_head, val_subset)
                loss_improvement = val_losses[-1] - new_val_loss

                # store loss improvement in the data item
                item.tv_point_estimates.append(loss_improvement)

                # Updates
                val_losses.append(new_val_loss)

        # Now compute estimate of training-value for each example in the training subset
        # This is simply the mean immediate improvement in loss that was observed when that example was trained on
        for item in train_dataset.items:
            item.estimated_tv = np.mean(item.tv_point_estimates)

        return train_subset

    def train_tv_nets(self, training_subset):
        """
        Train a Training-ValueNet for each class
        Parameters
        ----------
        training_subset: tv_net.dataset.Dataset
            subset of training set which were used in the MC estimation phase
        """
        self.logger.info('Training Training-ValueNets - weights will be saved in: {}'.format(self.log_dir))
        for class_name, tv_net in self.tv_nets.items():
            self.logger.info('Starting training Training-ValueNet for class: {}'.format(class_name))

            # Get a list of features for all examples in this class
            features = [example.feature_vector for example in training_subset.items if example.class_name == class_name]
            features_array = np.array(features)

            # Get a list of the estimated tv for each example
            targets = [example.estimated_tv for example in training_subset.items if example.class_name == class_name]
            targets_array = np.array(targets)

            # Get configs
            early_stop_patience = self.config.TVNET_EARLY_STOP_PATIENCE
            early_stop_delta = self.config.TVNET_EARLY_STOP_MIN_DELTA
            val_split = self.config.TVNET_VAL_SPLIT
            batch_size = self.config.TVNET_BATCH_SIZE
            epochs = self.config.TVNET_EPOCHS
            checkpoint_path = os.path.join(self.log_dir, 'tvnet_{}'.format(class_name))

            callbacks = [
                ModelCheckpoint(checkpoint_path, verbose=1, monitor='val_loss', mode='auto',
                                save_weights_only=True, save_best_only=True),
                EarlyStopping(monitor='val_loss', min_delta=early_stop_delta, patience=early_stop_patience, verbose=1,
                              mode='auto', restore_best_weights=True)
            ]

            tv_net.fit(features_array, targets_array, epochs=epochs, batch_size=batch_size, callbacks=callbacks, validation_split=val_split)
            self.logger.info('Finished training Training-ValueNet for class {}'.format(class_name))

    def predict_training_values(self, dataset_object):
        """
        Use the trained Training-ValueNet to predict the training-value of all examples in dataset.
        Predictions will be stored as an attribute of each example.
        Parameters
        ----------
        dataset_object: tv_net.dataset.Dataset
        """
        for class_name, tv_net in self.tv_nets.items():
            # Get a list of features for all examples in this class
            class_items = [item for item in dataset_object.items if item.class_name == class_name]
            features_list = [item.feature_vector for item in class_items]
            features_array = np.array(features_list)

            predictions_array = tv_net.predict(features_array, verbose=0)
            # loop through examples and add feature vector as an attribute
            for idx, item in enumerate(class_items):
                item.predicted_tv = predictions_array[idx][0]

    def _set_log_dir(self):
        # Directory for training logs
        self.log_dir = os.path.join(self.config.OUTPUT_DIR, 'training_value_networks')
        if not os.path.isdir(self.log_dir):
            os.mkdir(self.log_dir)

        # Path to save after each epoch. Epoch placeholder gets filled by Keras in ModelCheckpoint Callback
        self.__checkpoint_path = os.path.join(self.log_dir, 'best_checkpoint.h5')





