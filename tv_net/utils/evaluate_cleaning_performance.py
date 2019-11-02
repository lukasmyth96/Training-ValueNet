"""
Training-ValueNet

Evaluate the label cleaning detection performance.

For this you will need to create a directory containing a subset of the weakly-supervised training data.

The evaluation directory should be in the following format:

> evaluation_dir
    > class_1
        > clean
        > dirty
    .
    .

The evaluate_cleaning_performance function will then compute the precision and recall of the detections.

Licensed under the MIT License (see LICENSE for details)
Written by Luka Smyth
"""
import os

import pandas as pd


def evaluate_cleaning_performance(evaluation_dir, dataset_object):
    """
    Evaluates the cleaning performance on an evaluation set containing known clean and dirty examples
    Parameters
    ----------
    evaluation_dir: str
        dir containing subset of training examples from each class that should be manually sub-divided into clean
        and dirty subsets.
    dataset_object: tv_net.dataset.Dataset
        dataset object which contains the training examples and their predicted training-values

    Returns
    -------
    precision: float
        The precision in detecting mislabelled examples
    recall: float
        The recall in detecting mislabelled examples
    """

    # TODO we should start with some checks on the input

    evaluation_df = pd.DataFrame(columns=['filename',
                                          'weak_label',
                                          'true_positive',
                                          'true_negative',
                                          'false_positive',
                                          'false_negative',
                                          'predicted_tv'])

    for class_name in dataset_object.class_names:

        # Get list of filenames of clean and dirty evaluation examples for this class
        clean_eval_filenames = sorted(os.listdir(os.path.join(evaluation_dir, class_name, 'clean')))
        dirty_eval_filenames = sorted(os.listdir(os.path.join(evaluation_dir, class_name, 'dirty')))
        all_eval_filenames = clean_eval_filenames + dirty_eval_filenames

        for item in dataset_object.items:
            if item.class_name == class_name and item.filename in all_eval_filenames:
                is_mislabeled = item.filename in dirty_eval_filenames
                predicted_mislabeled = item.predicted_tv < 0
                evaluation_df = evaluation_df.append({'filename': item.filename,
                                                      'weak_label': class_name,
                                                      'true_positive': is_mislabeled and predicted_mislabeled,
                                                      'true_negative': not is_mislabeled and not predicted_mislabeled,
                                                      'false_positive': not is_mislabeled and predicted_mislabeled,
                                                      'false_negative': is_mislabeled and not predicted_mislabeled,
                                                      'predicted_tv': item.predicted_tv})

    true_positives = sum(evaluation_df[evaluation_df.true_positive])
    false_positives = sum(evaluation_df[evaluation_df.false_positive])
    false_negatives = sum(evaluation_df[evaluation_df.false_negative])

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)

    return precision, recall


