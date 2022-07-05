import random

from datasets.generator.CollectionGenerator import CollectionGenerator


class DatasetGenerator:
    """
    This class represents a mechanism through which training and testing collections for the experiments E1 and E2 are formed.
    """

    def __init__(self, labelling_function=lambda S: S[0], var_continuous=False, var_range=(0, 99)):
        """
        Constructor for the dataset generator

        :param labelling_function: A function L which assigns to each collection S, a label L(S)
        :param var_continuous: A boolean value indicating that sampling occurs uniformly (True) or discretely, in the desired range
        :param var_range: A tuple (a,b) representing the lower and upper bounds (both inclusive) of the sampling range
        """
        # Create a generator for the constituent collections
        self._generator = CollectionGenerator(var_continuous=var_continuous, var_range=var_range)

        # Assign the labelling function
        self._labelling_function = labelling_function

    def get_training_set(self, n=10 ** 5, upper=10):
        """
        Produces a training set according to the desired specifications. Each collection can vary in size, but is bounded above

        :param n: The number of collections in the training set
        :param upper: The maximum number of elements in each collection
        :return: A tuple (C,L) of collections C and their labels L
        """
        # The size of a collection in the training set varies, but is bounded above by 'upper'
        # A near uniform amount of sets of each size is produced.
        collections = [self._generator.create_collection(random.randint(1, upper)) for _ in range(n)]
        labels = [self._labelling_function(feat) for feat in collections]

        return collections, labels

    def get_testing_set(self, n=10 ** 5, size=10):
        """
        Produces a testing set according to the desired specifications. Each collection is of equivalent cardinality

        :param n: The number of collections in the testing set
        :param size: The maximum number of elements in each collection
        :return: A tuple (C,L) of collections C and their labels L
        """
        # The size of a collection in the is fixed to be 'size'
        collections = [self._generator.create_collection(size) for _ in range(n)]
        labels = [self._labelling_function(feat) for feat in collections]

        return collections, labels
