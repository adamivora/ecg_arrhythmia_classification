import warnings

import numpy as np
import torch
from onnxmltools.convert.xgboost.operator_converters.XGBoost import convert_xgboost
from onnxmltools.convert.xgboost.shape_calculators.Classifier import calculate_xgboost_classifier_output_shapes
from skl2onnx import update_registered_converter
from tqdm import tqdm
from xgboost import XGBClassifier


def enable_progress_apply():
    with warnings.catch_warnings():  # tqdm.pandas() call raises a FutureWarning in Pandas, we ignore the warning
        warnings.simplefilter("ignore")
        tqdm.pandas()  # tqdm support for Pandas df.progress_apply function


def enable_xgboost_onnx_support():
    update_registered_converter(XGBClassifier, 'XGBClassifier', calculate_xgboost_classifier_output_shapes,
                                convert_xgboost)


def seed_random_generators(random_seed):
    """
    Make the NumPy and PyTorch operations deterministic using `random_seed` as an RNG seed.
    """
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
