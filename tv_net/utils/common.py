"""
Various functions commonly used
"""
import logging
import pickle


def pickle_save(filename, an_object):

    with open(filename, 'wb') as output_file:
        pickle.dump(an_object, output_file, protocol=4)


def pickle_load(filename):

    with open(filename, 'rb') as input_file:
        an_object = pickle.load(input_file, encoding='bytes')

    return an_object


def create_logger(log_path):
    """ Setup logger to filepath and stderr"""

    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ])

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    return logger

