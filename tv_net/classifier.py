"""
Training-ValueNet

Classifier class encapsulates all functionality for underlying classifier.

- classifier is initially trained/ fine-tuned
- a feature-vector is extracted for each example
- classifier is used during the MC estimation phase of the algorithm

Licensed under the MIT License (see LICENSE for details)
Written by Luka Smyth
"""
import copy
import datetime
import math
import os
import random

import numpy as np

from keras.models import Model
from keras.layers import Input, Dense, Conv2D, Reshape, GlobalAveragePooling2D
from keras.optimizers import SGD
from keras.applications.mobilenet import MobileNet
from keras.applications.mobilenet import preprocess_input as mobilenet_preprocess_input
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard


class Classifier:
    """
    Encapsulates functionality of underlying classification model
    """
    def __init__(self, config):
        """
        Parameters
        ----------
        config: tv_net.config.Config
            configuration object
        """
        # TODO do some checks on the config here
        self.config = config

        # build feature extractor and classifier and compile
        self.feature_extractor = self._build_feature_extractor()
        self.classification_head = self._build_classication_head()
        # Store initial weights to randomly re-initialize during MC estimation phase
        self.classification_head_init_weights = self.classification_head.get_weights()
        self._set_log_dir()

    def train_baseline(self, train_dataset, val_dataset):
        """
        Train or fine-tune the baseline classifier
        This is done in order to extract a feature-vector for each example
        Parameters
        ----------
        train_dataset: tv_net.dataset.Dataset
            training dataset object
        val_dataset: tv_net.dataset.Dataset
            validation dataset object
        """

        assert self.config.NUM_CLASSES == train_dataset.num_classes
        assert self.config.NUM_CLASSES == val_dataset.num_classes
        
        # Combine feature extractor and MLP classifier
        input_layer = self.feature_extractor.input
        output_layer = self.classification_head(self.feature_extractor(input_layer))
        baseline_model = Model(inputs=input_layer, outputs=output_layer)
        
        # Compile
        self.compile_classifier(baseline_model)

        # Train and Val batch generators
        batch_size = self.config.BASELINE_CLF_BATCH_SIZE
        train_generator = self.batch_generator(train_dataset, batch_size=batch_size)
        val_generator = self.batch_generator(val_dataset, batch_size=batch_size)

        # Callbacks
        early_stop_patience = self.config.BASELINE_EARLY_STOP_PATIENCE
        early_stop_delta = self.config.BASLINE_EARLY_STOP_MIN_DELTA
        checkpoint_path = self.__checkpoint_path

        callbacks = [
            TensorBoard(log_dir=self.log_dir, histogram_freq=0, write_graph=True, write_images=False),
            ModelCheckpoint(checkpoint_path, verbose=1, monitor='val_acc', mode='auto',
                            save_weights_only=True, save_best_only=True),
            EarlyStopping(monitor='val_acc', min_delta=early_stop_delta, patience=early_stop_patience, verbose=1,
                          mode='auto', restore_best_weights=True)
        ]

        # Training

        train_steps = math.ceil(len(train_dataset.items) / batch_size)
        val_steps = math.ceil(len(val_dataset.items) / batch_size)

        baseline_model.fit_generator(
            train_generator,
            initial_epoch=0,
            epochs=self.config.BASELINE_CLF_EPOCHS,
            steps_per_epoch=train_steps,
            callbacks=callbacks,
            validation_data=val_generator,
            validation_steps=val_steps,
            max_queue_size=100,
            workers=1,
            use_multiprocessing=False,)

    def compute_loss_on_dataset_object(self, classification_head, dataset_object):
        """
        Compute loss of classification head on dataset object
        This assumes that feature vectors have already been extracted for all items in dataset object
        Parameters
        ----------
        classification_head: keras.engine.training.Model
        dataset_object: tv_net.dataset.Dataset
        Returns
        -------
        loss: float
        """
        # TODO find way to break down evaluation into multiple steps
        batch_size = self.config.EVAL_BATCH_SIZE
        
        features_list = [item.feature_vector for item in dataset_object.items]
        features_array = np.array(features_list)
        
        label_list = [item.class_name for item in dataset_object.items]
        class_one_hot_list = [dataset_object.class_names_to_one_hot[label] for label in label_list]
        labels_array = np.stack(class_one_hot_list, axis=0)
        
        loss = classification_head.evaluate(features_array, labels_array, batch_size=batch_size, verbose=0)[0] 
        return loss

    def _build_feature_extractor(self):
        """
        Build keras model for feature extractor - using pre-trained mobilenet_v2 here because it is fast to train
        """
        image_shape = self.config.IMG_DIMS + (3,)
        input_layer = Input(shape=image_shape)
        feature_vector = MobileNet(include_top=False, weights='imagenet', input_shape=image_shape, pooling='avg')(input_layer)
        feature_extractor = Model(inputs=input_layer, outputs=feature_vector)
        return feature_extractor

    def _build_classication_head(self):
        """
        Build keras model for underlying classifier
        The classifier should be a small MLP that sits on top of the feature extractor
        Returns
        -------
        classification_head: keras.engine.training.Model
            underlying classification model to be used
        """
        input_layer = Input(shape=(self.feature_extractor.output_shape[1],))
        softmax_layer = Dense(self.config.NUM_CLASSES, activation='softmax')(input_layer)
        classification_head = Model(inputs=input_layer, outputs=softmax_layer)
        return classification_head
    
    def compile_classifier(self, model):
        """ Compile classification model"""
        optimizer = SGD(lr=self.config.BASELINE_CLF_LR)
        model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])
    
    def reinitialize_classification_head(self, classification_head):
        """ Approximation of randomly reinitializing weights of classification head
        This is used at the start of each episode during the MC estimation phase.
        Parameters
        ----------
        classification_head: keras.engine.training.Model
            model to reinitialize
        """
        init_weights = self.classification_head_init_weights
        new_weights = [np.random.permutation(w.flat).reshape(w.shape) for w in init_weights]
        classification_head.set_weights(new_weights)
        
        return classification_head

    def batch_generator(self, dataset_object, shuffle=True, batch_size=1):
        """
        A generator that yields batches of examples and ther class labels in one hot form
        Method is public because it is called directly by the MC estimation method of the Training-ValueNet class
        Parameters
        ----------
        dataset_object: tv_net.dataset.Dataset
            dataset object to generate batches from
        shuffle: bool
            whether to shuffle examples at the start of each epoch
        batch_size: int

        Yields
        -------
        batch_examples: np.ndarray
            batch of examples
        batch_labels: np.ndarray
            batch of one-hot class labels

        """
        b = 0  # batch item index
        item_index = 0
        data_items = dataset_object.items  # create a copy of data item to avoid editing it

        batch_examples = list()  # initialising as list - convert both to numpy array before yielding
        batch_labels = list()

        # Keras requires a generator to run indefinitely.
        while True:
            try:
                # Increment index to pick next image. Shuffle if at the start of an epoch.
                item_index = item_index % len(data_items)
                if shuffle and item_index == 0:
                    random.shuffle(data_items)

                # Get items natrix image and class id.
                item = data_items[item_index]
                data = copy.copy(item.data) # TODO investigate why this is needed
                preprocessed_data = self._preprocess_example(data)
                class_label_one_hot = dataset_object.class_names_to_one_hot[item.class_name]

                # Reset batch arrays at start of batch
                if b == 0:
                    batch_examples = list()
                    batch_labels = list()

                # Add to batch  #
                batch_examples.append(preprocessed_data)
                batch_labels.append(class_label_one_hot)

                b += 1
                item_index += 1

                # Batch full?
                if b >= batch_size:
                    yield np.concatenate(batch_examples, axis=0), np.stack(batch_labels, axis=0)  # convert lists to np array
                    # start a new batch
                    b = 0

            except (GeneratorExit, KeyboardInterrupt):
                raise             

    @staticmethod
    def _preprocess_example(example):
        """
        Preprocess data for a single example
        Parameters
        ----------
        example:
            data for single example
        Returns
        -------
        preprocessed_example:

        """
        example = np.expand_dims(example, axis=0)  # add extra dimension for batch
        preprocessed_example = mobilenet_preprocess_input(example)
        return preprocessed_example

    def _set_log_dir(self):
        # Directory for training logs
        self.log_dir = os.path.join(self.config.OUTPUT_DIR, 'baseline_classifier_weights')
        if not os.path.isdir(self.log_dir):
            os.mkdir(self.log_dir)

        # Path to save after each epoch. Epoch placeholder gets filled by Keras in ModelCheckpoint Callback
        self.__checkpoint_path = os.path.join(self.log_dir, 'best_checkpoint.h5')



