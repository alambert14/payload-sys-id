import numpy as np


def stats(filename):
    """
    Do stats on some data
    :param filename: File containing data that has been collected
    :return:
    """
    data = np.loadtxt(filename)

    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)

    return mean, std
