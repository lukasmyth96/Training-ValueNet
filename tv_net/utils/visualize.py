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

    title = 'T-SNE visualization of feature vectors extracted from baseline classifier'
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
        ax.hist(class_predicted_tvs, bins=50)
        ax.set_title('Histogram of predicted training-values for class: {}'.format(class_name))
        fig.savefig(output_path)
        plt.show()