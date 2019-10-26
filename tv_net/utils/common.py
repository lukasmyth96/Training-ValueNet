"""
Various functions commonly used
"""
from distutils.util import strtobool
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


# TODO use below function to is setup checks
def ask_user_confirmation(question):
    """
    Ask yes / question to user and return bool
    Parameters
    ----------
    question: str
        yes / no question to ask
    Returns
    -------
    response: bool
        whether user confirms or not
    """
    question += ' [y / n]'
    try:
        response_str = input(question)
        response_bool = strtobool(response_str)
        return response_bool
    except ValueError:
        print('Invalid response - Try again')
        ask_user_confirmation(question)
