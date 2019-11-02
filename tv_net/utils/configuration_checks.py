"""
To check that the passed configuration is valid
"""
import os


def check_configuration(config):
    """
    A function to check the configuration is valid.

    Parameters
    ----------
    config: tv_net.config.Config

    Raises
    ------
    ValueError
        If the configuration is invalid
    """

    if config.CLASSES_TO_USE is not None:

        # Check that specified CLASSES_TO_USE actually exist
        for subset_dir in [config.TRAIN_DATASET_DIR, config.VAL_DATASET_DIR]:
            classes_found = [directory for directory in os.listdir(subset_dir)]
            if not set(config.CLASSES_TO_USE).issubset(classes_found):
                rogue_classes = set(config.CLASSES_TO_USE) - set(classes_found)
                raise ValueError('Config Error: \n '
                                 'Specified classes to use not found: {}'.format(rogue_classes))

        # Check NUM_CLASSES is same number as CLASSES_TO_USE
        if len(config.CLASSES_TO_USE) != config.NUM_CLASSES:
            raise ValueError('Configuration Error: \n'
                             'NUM_CLASSES: {} \n'
                             'CLASSES_TO_USE: {}'.format(config.NUM_CLASSES, len(config.CLASSES_TO_USE)))

    else:

        # Check that the classes found are the same for Train and Val
        train_classes_found = sorted(os.listdir(config.TRAIN_DATASET_DIR))
        val_classes_found = sorted(os.listdir(config.VAL_DATASET_DIR))
        if train_classes_found != val_classes_found:
            raise ValueError('Config Error: classes found do not match \n'
                             'train classes found: {} \n'
                             'val classes found: {}'.format(train_classes_found, val_classes_found))