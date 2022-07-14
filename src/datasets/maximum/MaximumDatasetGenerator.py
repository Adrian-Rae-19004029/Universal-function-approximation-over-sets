import numpy as np

from datasets.generator.DatasetGenerator import DatasetGenerator


class MaximumDatasetGenerator(DatasetGenerator):
    """
    This class represents a mechanism through which training and testing collections for the experiment E1 is formed.
    """

    def __init__(self, var_range=(0, 99)):
        # This is merely a specialisation of the DatasetGenerator class with summation as its labelling
        super().__init__(labelling_function=lambda S: np.sum(S), var_continuous=False, var_range=var_range)


