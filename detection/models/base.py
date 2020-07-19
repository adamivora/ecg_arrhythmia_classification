from abc import ABC, abstractmethod
from collections import namedtuple

import joblib

Score = namedtuple('Score', 'mean dev data')


class BaseModel(ABC):
    """
    The base model that defines the interface all other model wrappers should use.
    """

    def __repr__(self):
        return self.name()

    def save(self, output_filename):
        """
        Saves the binary model using joblib into `output_filename`.
        """
        joblib.dump(self, output_filename)

    def load(self, input_filename):
        """

        :param input_filename:
        """
        return joblib.load(input_filename)

    @abstractmethod
    def score(self, dataset, cv):
        pass

    @abstractmethod
    def name(self):
        """
        Returns the pretty name of the underlying classifier.

        :return: str
        """
        pass

    @abstractmethod
    def train(self, dataset, **kwargs):
        """
        Trains the underlying classifier on the dataset.

        :param dataset: a dataset deriving from the BaseDataset class
        :param kwargs: keyword args, implementation and usage left for the derived classes
        """
        pass

    @abstractmethod
    def predict(self, dataset):
        """
        Returns the predicted labels for all the `dataset` data.

        :param dataset: a dataset deriving from the BaseDataset class
        :return: the predicted label codes
        """
        pass

    def export(self, output_dir):
        """
        Exports the underlying classifier in ONNX format to `output_dir`.

        :param output_dir: the output directory
        """
        print(f'Export for model {self} not implemented.')
