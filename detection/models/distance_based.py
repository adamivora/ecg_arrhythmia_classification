import torchvision.transforms as transforms
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
from tslearn.utils import to_time_series_dataset

from detection.models.base import BaseModel
from detection.preprocessing.transforms import Resample, Standardize
from detection.utils.constants import CV
from .base import Score


def get_distance_based_transforms():
    return transforms.Compose([
        Resample(fs=100),
        Standardize()
    ])


class DistanceBasedClassifier(BaseModel):
    """
    Wrapper class for all the distance-based models.
    """

    def __init__(self, classifier, transform=None, checkpoint=None):
        self.clf = classifier
        if transform is None:
            transform = get_distance_based_transforms()
        self.transform = transform

        super().__init__(checkpoint)

    def name(self):
        return type(self.clf).__name__

    def score(self, dataset, cv):
        X, y_true, _ = dataset.get_signals(self.transform)
        X = to_time_series_dataset(X)

        if cv:
            score = cross_val_score(self.clf, X, y_true, scoring='f1_macro', cv=CV)
            return Score(score.mean(), 2 * score.std(), score)
        else:
            y_pred = self.clf.predict(X)
            return f1_score(y_true, y_pred, average='macro')

    def train(self, dataset, **kwargs):
        X, y, _ = dataset.get_signals(self.transform)
        X = to_time_series_dataset(X)
        self.clf.fit(X, y, **kwargs)

    def predict(self, dataset):
        X, *_ = dataset.get_signals(self.transform)
        X = to_time_series_dataset(X)
        return self.clf.predict(X)
