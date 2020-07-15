from os import path

from onnxconverter_common import FloatTensorType
from onnxmltools import convert_sklearn
from onnxmltools.utils import save_model
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler

from detection.utils.constants import CV
from .base import BaseModel, Score


def get_default_pipeline():
    return make_pipeline(
        SimpleImputer(),
        MinMaxScaler(),
    )


class FeatureBasedClassifier(BaseModel):
    """
    Wrapper class for all the scikit-learn feature-based models.
    """
    cv = True

    def __init__(self, classifier, pipeline=None, checkpoint=None):
        self.clf = classifier
        if pipeline is None:
            pipeline = get_default_pipeline()

        self.pipe = pipeline
        self.pipe = make_pipeline(self.pipe, classifier)

        super().__init__(checkpoint)

    def name(self):
        return type(self.clf).__name__

    def score(self, dataset, cv=False):
        X, y_true = dataset.get_features()

        if cv:
            score = cross_val_score(self.pipe, X, y_true.values, scoring='f1_macro', cv=CV)
            return Score(score.mean(), 2 * score.std(), score)
        else:
            y_pred = self.pipe.predict(X)
            return f1_score(y_true.values, y_pred, average='macro')

    def train(self, dataset, **fit_kwargs):
        X, y = dataset.get_features()
        self.pipe.fit(X, y, **fit_kwargs)

    def predict(self, dataset):
        X, _ = dataset.get_features()
        return self.pipe.predict(X)

    def export(self, output_dir):
        onnx_model = convert_sklearn(self.pipe, initial_types=[("input", FloatTensorType([1, 1]))])
        save_model(onnx_model, path.join(output_dir, f'{self.name()}.onnx'))
