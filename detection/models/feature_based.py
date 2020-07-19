from os import path

from onnxconverter_common import FloatTensorType
from onnxmltools import convert_sklearn
from onnxmltools.utils import save_model
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .base import BaseModel, Score


def get_default_pipeline():
    """
    Get the default pipeline for models with hand-engineered features.

    :return: Pipeline
    """

    return make_pipeline(
        SimpleImputer(),
        StandardScaler(),
    )


def get_default_rf_pipeline():
    """
    Get the default pipeline for RandomForestClassifier.

    :return: Pipeline
    """

    return make_pipeline(
        SimpleImputer()
    )


class FeatureBasedClassifier(BaseModel):
    """
    Wrapper class for all the models based on the hand-engineered features.
    """
    cv = True

    def __init__(self, classifier, pipeline='default', cv_folds=5):
        super().__init__()

        if pipeline == 'default':
            pipeline = get_default_pipeline()

        self.clf = classifier
        self.pipe = pipeline
        self.pipe = make_pipeline(self.pipe, classifier)
        self.cv_folds = cv_folds

    def name(self):
        return type(self.clf).__name__

    def score(self, dataset, cv=False):
        X, y_true = dataset.get_features()

        if cv:
            score = cross_val_score(self.pipe, X, y_true.values, scoring='f1_macro', cv=self.cv_folds, n_jobs=-1)
            return Score(score.mean(), score.std(), score)
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
