"""
Training-ValueNet

TrainingValueNet class encapsulates all functionality for algorithm.

Licensed under the MIT License (see LICENSE for details)
Written by Luka Smyth
"""
import numpy as np

from keras.models import Model
from keras.layers import Input, Dense, Conv2D, Reshape
from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input


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
        self.config = config

    def mc_estimation_phase(self, dataset_train, dataset_val):
        """
        Monte-Carlo estimation phase.
        Here we estimate the training-values of a subset of training examples from each class.
        These estimates are then used at regression targets to train the Training-ValueNet.
        Parameters
        ----------
        dataset_train: tv_net.dataset.Dataset
            training dataset object
        dataset_val: tv_net.dataset.Dataset
            validation dataset object
        """

        assert dataset_train.num_classes == self.config.NUM_CLASSES, "Mismatch in number of classes"
        assert dataset_val.num_classes == self.config.NUM_CLASSES, "Mismatch in number of classes"

        train_subset = get_random_subset(dataset_train, self.config.TRAIN_SUBSET_NUM_PER_CLASS)
        train_subset.load_images()

        val_subset = get_random_subset(dataset_val, self.config.VAL_SUBSET_NUM_PER_CLASS)
        val_subset.load_images()

        for episode in self.config.MC_EPISODES:
            classifier = self._build_classifier()

            num_train_examples = len(train_subset.items)
            item_idx = 0
            iteration = 0

            val_losses = list()  # Create list to store val losses
            val_losses.append(self.compute_loss_on_dataset(classifier, val_subset))  # compute starting val loss
            while iteration < (self.config.MC_EPOCHS * num_train_examples):

                # Shuffle at start of epoch
                if iteration % num_train_examples == 0:
                    train_subset.shuffle_examples()

                # Get example to train on
                item = train_subset.items[item_idx]
                image = np.expand_dims(item.image, axis=0)  # add batch dimension
                image = preprocess_input(image)  # Preprocessing is specific to MobileNet_V2
                class_one_hot = dataset_train.class_names_to_one_hot(item.class_name)

                # Train classifier on image
                classifier.train_on_batch(image, class_one_hot)

                # Compute improvement in val loss
                new_val_loss = self.compute_loss_on_dataset(classifier, val_subset)
                loss_improvement = val_losses[-1] - new_val_loss
                # TODO store the improvements somewhere

                # Updates
                val_losses.append(new_val_loss)
                item_idx += 1
                iteration += 1

    def _build_classifier(self):

        image_shape = (self.config.IMAGE_SIZE, self.config.IMAGE_SIZE, 3)

        # Create MobileNet
        input_layer = Input(shape=image_shape)
        mobilenet = MobileNetV2(input_shape=image_shape, include_top=False)

        # to freeze layers
        for layer in mobilenet.layers:
            layer.trainable = self.config.CNN_LAYERS_TRAINABLE

        mobilenet_feature_map = mobilenet(input)
        # Get side length of feature map as this determines kernel size for final conv layer
        mobilenet_feature_map_size = int(mobilenet.layers[-1].output.shape[1].value)

        # Pass cnn features through final CNN layer - output shape will be (batch, 1, 1, 100)
        final_conv_features = Conv2D(self.config.FINAL_CONV_FEATURE_DIM, kernel_size=mobilenet_feature_map_size, activation='relu')(mobilenet_feature_map)
        flattened_conv_features = Reshape((self.config.FINAL_CONV_FEATURE_DIM,))(final_conv_features)  # reshape to 1D vector
        output_layer = Dense(self.config.NUM_CLASSES, activation='softmax')(flattened_conv_features)
        classifier = Model(inputs=input_layer, outputs=output_layer)

        # Compile
        classifier.compile(loss="categorical_crossentropy", optimizer='sgd')

        return classifier

    def compute_loss_on_dataset(self, classifier, dataset):
        """
        Compute mean loss on all examples in dataset
        Parameters
        ----------
        classifier: keras.engine.training.Model
            classifier model
        dataset: tv_net.dataset.Dataset
            dataset object on which to compute loss

        Returns
        -------
        loss: float
            mean loss on items in dataset
        """

        image_size = (self.config.IMAGE_SIZE, self.config.IMAGE_SIZE, 3)
        num_examples = len(dataset.items)

        # Create containers for batch images and labels
        batch_image_size = (num_examples,) + image_size  # tuple
        batch_images = np.zeros(batch_image_size)
        batch_one_hot_labels = np.zeros([num_examples, dataset.num_classes])

        # Add images and one hot labels to batch
        for idx, item in enumerate(dataset.items):
            batch_images[idx] = item.image
            batch_one_hot_labels = dataset.class_names_to_one_hot(item.class_name)

        # Evaluate on batch
        # TODO check what .evaluate actually returns a scalar loss
        loss = classifier.evaluate(batch_images, batch_one_hot_labels, batch_size=self.config.EVAL_BATCH_SIZE)

        return loss
