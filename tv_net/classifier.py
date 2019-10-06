"""
Training-ValueNet

Classifier class encapsulates all functionality for underlying classifier.

- classifier is initially trained/ fine-tuned
- a feature-vector is extracted for each example
- classifier is used during the MC estimation phase of the algorithm

Licensed under the MIT License (see LICENSE for details)
Written by Luka Smyth
"""
import datetime
import math
import multiprocessing
import os
import random

import numpy as np

from keras.models import Model
from keras.layers import Input, Dense, Conv2D, Reshape
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess_input
from keras.callbacks import ModelCheckpoint, EarlyStopping


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
        self.classification_model = self._build_classifier()
        self.compile_classifier()

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

        # Train and Val batch generators
        batch_size = self.config.BASELINE_CLF_BATCH_SIZE
        train_generator = self.batch_generator(train_dataset, batch_size=batch_size)
        val_generator = self.batch_generator(val_dataset, batch_size=batch_size)

        # Callbacks
        early_stop_patience = self.config.BASELINE_EARLY_STOP_PATIENCE
        early_stop_delta = self.config.BASLINE_EARLY_STOP_MIN_DELTA
        callbacks = [
            ModelCheckpoint(self.__checkpoint_path, verbose=1, monitor='val_acc', mode='auto',
                            save_weights_only=True, save_best_only=True),
            EarlyStopping(monitor='val_acc', min_delta=early_stop_delta, patience=early_stop_patience, verbose=1,
                          mode='auto')
        ]

        # Training

        # Work-around for Windows: Keras fails on Windows when using multiprocessing workers
        if os.name is 'nt':
            workers = 0
        else:
            workers = multiprocessing.cpu_count()

        train_steps = math.ceil(len(train_dataset.items) / batch_size)
        val_steps = math.ceil(len(val_dataset.items) / batch_size)

        self.classification_model.fit_generator(
            train_generator,
            initial_epoch=0,
            epochs=self.config.BASELINE_CLF_EPOCHS,
            steps_per_epoch=train_steps,
            callbacks=callbacks,
            validation_data=val_generator,
            validation_steps=val_steps,
            max_queue_size=100,
            workers=workers,
            use_multiprocessing=True,
        )

    def compute_loss_on_dataset_object(self, dataset_object):
        """
        Compute loss of classifier on dataset object
        Parameters
        ----------
        dataset_object: tv_net.dataset.Dataset
        Returns
        -------
        loss: float
        """
        # TODO find way to break down evaluation into multiple steps
        evaluation_generator = self.batch_generator(dataset_object, batch_size=dataset_object.num_examples)
        loss = self.classification_model.evaluate_generator(evaluation_generator, steps=1)
        return loss

    def _build_feature_extractor(self):
        """
        Build keras model for feature extractor - using pre-trained mobilenet_v2 here because it is fast to train
        """
        image_shape = self.config.IMAGE_DIMS + (3,)
        input_layer = Input(shape=image_shape)
        mobilenet = MobileNetV2(input_shape=image_shape, include_top=False)

        # set whether mobilenet layers are trainable
        for layer in mobilenet.layers:
            layer.trainable = True

        mobilenet_feature_map = mobilenet(input_layer)
        mobilenet_feature_map_size = int(mobilenet_feature_map.shape[1].value)  # feature map side length used tp determine kernel size
        final_conv_features = Conv2D(self.config.FEATURE_VECTOR_SHAPE, kernel_size=mobilenet_feature_map_size,
                                     activation='relu')(mobilenet_feature_map)
        feature_vector = Reshape((self.config.FEATURE_VECTOR_SHAPE,))(final_conv_features)

        feature_extractor = Model(inputs=input_layer, outputs=feature_vector)
        return feature_extractor

    def _build_classifier(self):
        """
        Build keras model for underlying classifier
        The classifier should be a small MLP that sits on top of the feature extractor
        Returns
        -------
        classifier: keras.engine.training.Model
            underlying classification model to be used
        """
        softmax_layer = Dense(self.config.NUM_CLASSES, activation='softmax')(self.feature_extractor.output)

        classifier = Model(inputs=self.feature_extractor.input, outputs=softmax_layer)
        return classifier

    def compile_classifier(self):
        """Method is public so that the classifier can be re-initialized during the MC estimation phase"""
        # TODO optimizer should be configurable
        self.classification_model.compile(loss="categorical_crossentropy", optimizer=self.config.BASELINE_CLF_OPTIMIZER)

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
        data_items = dataset_object.items

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
                data = item.data
                data = self._preprocess_example(data)
                class_label_one_hot = dataset_object.class_names_to_one_hot[item.class_name]

                # Reset batch arrays at start of batch
                if b == 0:
                    batch_examples = list()
                    batch_labels = list()

                # Add to batch  #
                batch_examples.append(data)
                batch_labels.append(class_label_one_hot)

                b += 1
                item_index += 1

                # Batch full?
                if b >= batch_size:
                    yield np.array(batch_examples), np.array(batch_labels)  # convert lists to np array
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

        # Path to save after each epoch. Epoch placeholder gets filled by Keras in ModelCheckpoint Callback
        self.__checkpoint_path = os.path.join(self.log_dir, 'best_checkpoint.h5')



