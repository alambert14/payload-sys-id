import numpy as np


class Voxel:

    def __init__(self):
        self.mean = None
        self.std = None
        self.coordinate = (0, 0, 0)

    def evaluate(self, seed = None):
        """
        Sample a value from the normal distribution of the voxel
        :param seed: Optional seed for the rng
        :return:
        """
        np.random.seed(seed)
        return np.random.normal(self.mean, self.std)

    def __hash__(self):
        """
        Overriden hash function for the voxel object
        :return:
        """
        return hash(self.coordinate)

    def __eq__(self, other):
        return self.coordinate == other.coordinate


