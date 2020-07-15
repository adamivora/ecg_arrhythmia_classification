from abc import ABC, abstractmethod

from collections import namedtuple

Score = namedtuple('Score', 'mean dev data')


class BaseModel(ABC):
    """
    The base model that defines the interface all other model wrappers should use.
    """
    def __init__(self, checkpoint=None, **kwargs):
        if checkpoint is not None:
            self.load(checkpoint)

    def __repr__(self):
        return self.name()

    # def score(self, dataset, scoring_func=f1_score_classes):
    #    y_true = dataset.get_labels()
    #    y_pred = self.predict(dataset)
    #    return scoring_func(y_true, y_pred)

    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def train(self, dataset):
        pass

    @abstractmethod
    def predict(self, dataset):
        pass

    def export(self, output_dir):
        print(f'Export for model {self} not implemented.')