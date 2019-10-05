"""
Training-ValueNet

Classifier class encapsulates all functionality for underlying classifier.

- classifier is initially trained/ fine-tuned
- a feature-vector is extracted for each example
- classifier is used during the MC estimation phase of the algorithm

Licensed under the MIT License (see LICENSE for details)
Written by Luka Smyth
"""
import random

import numpy as np

from keras.models import Model
from keras.layers import Input, Dense, Conv2D, Reshape
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess_input
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping

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

        # build and compile underlying keras model for classifier
        self.keras_model = self._build_classifier()
        self.compile_keras_model()

    def train_baseline(self, dataset_train, dataset_val):
        """
        Train or fine-tune the baseline classifier
        This is done in order to extract a feature-vector for each example
        Parameters
        ----------
        dataset_train: tv_net.dataset.Dataset
            training dataset object
        dataset_val: tv_net.dataset.Dataset
            validation dataset object
        """
        # Load initialisation weights if fine-tuning
        if self.config.FINETUNE_BASELINE_CLF:
            init_weights_path = self.config.BASELINE_CLF_INIT_WEIGHTS
            self.keras_model.load_weights(filepath=init_weights_path, by_name=True)

        # Train and Val batch generators
        train_generator = self.batch_generator(dataset_train, batch_size=self.config.BASELINE_CLF_BATCH_SIZE)
        val_generator = self.batch_generator(dataset_val, batch_size=self.config.BASELINE_CLF_BATCH_SIZE)

        # Callbacks
        early_stop_patience = self.config.BASELINE_EARLY_STOP_PATIENCE
        early_stop_delta = self.config.BASLINE_EARLY_STOP_MIN_DELTA
        callbacks = [
            TensorBoard(log_dir=self.log_dir, histogram_freq=0, write_graph=True, write_images=False),
            ModelCheckpoint(self.__checkpoint_path, verbose=1, monitor='val_acc', mode='auto',
                            save_weights_only=True, save_best_only=True),
            EarlyStopping(monitor='val_acc', min_delta=early_stop_delta, patience=early_stop_patience, verbose=1,
                          mode='auto')
        ]

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
        num_items = len(dataset_object.items)  # compute in one large batch for now
        evaluation_generator = self.batch_generator(dataset_object, batch_size=num_items)
        loss = self.keras_model.evaluate_generator(evaluation_generator, steps=1)

        return loss

    def _build_classifier(self):
        """
        Build keras model for underlying classifier
        Returns
        -------
        classifier: keras.engine.training.Model
            underlying classification model to be used
        """
        image_shape = (self.config.IMAGE_SIZE, self.config.IMAGE_SIZE, 3)
        # Download MobileNet
        input_layer = Input(shape=image_shape)
        mobilenet = MobileNetV2(input_shape=image_shape, include_top=False)

        # set whether mobilenet layers are trainable
        for layer in mobilenet.layers:
            layer.trainable = self.config.CNN_LAYERS_TRAINABLE

        mobilenet_feature_map = mobilenet(input)

        # Get side length of feature map as this determines kernel size for final conv layer
        mobilenet_feature_map_size = int(mobilenet_feature_map.output.shape[1].value)

        # Pass cnn features through final CNN layer - output shape will be (batch, 1, 1, 100)
        final_conv_features = Conv2D(self.config.FINAL_CONV_FEATURE_DIM, kernel_size=mobilenet_feature_map_size,
                                     activation='relu')(mobilenet_feature_map)
        flattened_conv_features = Reshape((self.config.FINAL_CONV_FEATURE_DIM,))(
            final_conv_features)  # reshape to 1D vector
        output_layer = Dense(self.config.NUM_CLASSES, activation='softmax')(flattened_conv_features)
        classifier = Model(inputs=input_layer, outputs=output_layer)

        return classifier

    def compile_keras_model(self):
        """Method is public so that the classifier can be re-initialized during the MC estimation phase"""
        # TODO optimizer should be configurable
        self.keras_model.compile(loss="categorical_crossentropy", optimizer='sgd')

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
                class_label_one_hot = dataset_object.class_names_to_one_hot(item.class_name)

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

