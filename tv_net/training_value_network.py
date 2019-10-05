"""
Training-ValueNet

TrainingValueNet class encapsulates all functionality for algorithm.

Licensed under the MIT License (see LICENSE for details)
Written by Luka Smyth
"""
import numpy as np

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
        self.classifier = Classifier(config)

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

        # Get random subset of train and val to estimate on
        train_subset = get_random_subset(dataset_train, self.config.TRAIN_SUBSET_NUM_PER_CLASS)
        val_subset = get_random_subset(dataset_val, self.config.VAL_SUBSET_NUM_PER_CLASS)

        # Get batch generators for training set - batch size must be 1
        batch_generator = self.classifier.batch_generator(train_subset, shuffle=True, batch_size=1)

        for episode in range(self.config.MC_EPISODES):
            self.classifier.compile_keras_model()  # re-initialize at start of each episode

            num_train_examples = len(train_subset.items)
            iteration = 0

            val_losses = list()  # Create list to store val losses
            val_losses.append(self.classifier.compute_loss_on_dataset_object(val_subset))  # append initial val loss
            while iteration < (self.config.MC_EPOCHS * num_train_examples):

                # Get next example to train on
                example, one_hot_label = batch_generator.__next__()

                # Train classifier on image
                self.classifier.keras_model.train_on_batch(example, one_hot_label)

                # Compute immediate improvement in val loss
                new_val_loss = self.classifier.compute_loss_on_dataset_object(val_subset)
                loss_improvement = val_losses[-1] - new_val_loss

                # store loss improvement in the data item
                example.tv_point_estimates.append(loss_improvement)

                # Updates
                val_losses.append(new_val_loss)
                iteration += 1

