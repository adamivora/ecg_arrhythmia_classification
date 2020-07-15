from collections import OrderedDict
from glob import glob
from os import path

import joblib


class ModelPersistence:
    def __init__(self, models_dir):
        self.models_dir = models_dir

    def save_models(self, models):
        for model in models:
            joblib.dump(model, path.join(self.models_dir, f'{model.name()}.gz'))

    def load_models(self, models):
        model_dict = OrderedDict((model.name(), model) for model in models)

        for filename in glob(path.join(self.models_dir, '*.gz')):
            name = path.splitext(filename)[0]
            if name in model_dict:
                model_dict[name] = joblib.load(filename)

        return model_dict.values()
