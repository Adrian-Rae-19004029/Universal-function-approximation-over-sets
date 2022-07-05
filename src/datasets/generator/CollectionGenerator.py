import random

import numpy as np


class CollectionGenerator:
    """
    This class represents a generic interface through which a collection of testing and training set related to E1 and E2 are developed.
    """

    def __init__(self,
                 var_continuous=False,
                 var_range=(0, 99),
                 ):
        """
        The constructor for the generator

        :param var_continuous: A boolean value indicating that sampling occurs uniformly (True) or discretely, in the desired range
        :param var_range: A tuple (a,b) representing the lower and upper bounds (both inclusive) of the sampling range
        """

        # Set the upper/lower bound on generation
        self._lower_bound, self._upper_bound = var_range

        # Determine whether sampling is performed discretely or uniformly
        if var_continuous:
            self._sampling_function = lambda: random.uniform(self._lower_bound, self._upper_bound)
        else:
            self._sampling_function = lambda: random.randint(self._lower_bound, self._upper_bound)

    def create_collection(self, n=5):
        """
        Creates a collection of specified size according to the established sampling strategy

        :param n: The desired number of set elements
        :return: A list of n elements.
        Note here that a list is preferred over a 'set' proper as we permit multi-sets containing non-distinct elements as 'sets' for our purposes
        """

        # n is a positive integer
        assert n > 0

        return np.array([self._sampling_function() for _ in range(n)])