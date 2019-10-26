"""
Training-ValueNet

Visualization tools

Licensed under the MIT License (see LICENSE for details)
Written by Luka Smyth
"""
import os

from matplotlib import pyplot as plt
import numpy as np
from yellowbrick.text import TSNEVisualizer


# TODO should probably do some checks on the input to these two functions
def tsne_visualization(dataset_object, num_examples):
    """
    Produce and save T-SNE visualization of feature vectors for given dataset
    Parameters
    ----------
    dataset_object: tv_net.dataset.Dataset
        dataset object containing feature vectors and class names
    num_examples: int
        number of examples to plot
    """

    dataset_object.shuffle_examples()  # shuffle so that we don't get all one class
    feature_vectors = np.array([item.feature_vector for item in dataset_object.items[:num_examples]])
    label_list = [item.class_name for item in dataset_object.items[:num_examples]]

    title = 'T-SNE of feature vectors extracted from baseline classifier - using random sample of {} images'.format(num_examples)
    tsne = TSNEVisualizer(colormap='rainbow', title=title)
    tsne.fit(feature_vectors, label_list)
    output_path = os.path.join(dataset_object.config.OUTPUT_DIR, 'visualizations', 'feature_vector_tsne.png')
    tsne.show(outpath=output_path)
    tsne.show()  # have to repeat to show and save


def produce_tv_histograms(dataset_object):
    """
    Produce a histogram of the predicted training-values for all examples in the training set
    Parameters
    ----------
    dataset_object: tv_net.dataset.Dataset
    """

    for class_name in dataset_object.class_names:
        class_predicted_tvs = [item.predicted_tv for item in dataset_object.items if item.class_name == class_name]
        output_path = os.path.join(dataset_object.config.OUTPUT_DIR, 'visualizations',
                                   'predicted_tv_hist_{}.png'.format(class_name))

        # Plot and save
        fig, ax = plt.subplots()
        ax.hist(class_predicted_tvs, bins=100)
        fig.text(0.1, 0.8, 'Dirty Examples: {}'.format(len([val for val in class_predicted_tvs if val < 0])), ha='left')
        fig.text(0.9, 0.8, 'Clean Examples: {}'.format(len([val for val in class_predicted_tvs if val > 0])), ha='right')
        ax.axvline(0, color='red', linestyle='dashed', linewidth=2)  # to mark threshold between clean and dirty ex
        ax.set_title('Histogram of predicted training-values for class: {}'.format(class_name))
        fig.savefig(output_path)
        plt.show()


def plot_training_history(history):
    """
    Plot loss and acc for baseline training
    Parameters
    ----------
    history: keras.callbacks.History
        keras history object
    """
    # Plot training & validation accuracy values
    fig, ax = plt.subplots()
    ax.plot(history.history['acc'])
    ax.plot(history.history['val_acc'])
    ax.title('Model accuracy')
    ax.ylabel('Accuracy')
    ax.xlabel('Epoch')
    ax.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    fig, ax = plt.subplots()
    ax.plot(history.history['loss'])
    ax.plot(history.history['val_loss'])
    ax.title('Model loss')
    ax.ylabel('Loss')
    ax.xlabel('Epoch')
    ax.legend(['Train', 'Test'], loc='upper left')
    plt.show()