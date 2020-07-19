import random
import warnings

import numpy as np
import torch
from onnxmltools.convert.xgboost.operator_converters.XGBoost import convert_xgboost
from onnxmltools.convert.xgboost.shape_calculators.Classifier import calculate_xgboost_classifier_output_shapes
from skl2onnx import update_registered_converter
from tqdm import tqdm
from xgboost import XGBClassifier


def enable_progress_apply():
    """
    Enable the progress_apply operation which shows the tqdm progressbar for pd.DataFrame.
    """

    with warnings.catch_warnings():  # tqdm.pandas() call raises a FutureWarning in Pandas, we ignore the warning
        warnings.simplefilter("ignore")
        tqdm.pandas()  # tqdm support for Pandas df.progress_apply function


def enable_xgboost_onnx_support():
    """
    Enable XGBoost classifier support for the onnxmltools exporter.

    Thanks to Jordan McDonald, who published the gist:
        https://gist.github.com/jordantoaster/316394b594c578875a5780b489e24d63#file-sklearn-xgboost-onnx-py
    """
    update_registered_converter(XGBClassifier, 'XGBClassifier', calculate_xgboost_classifier_output_shapes,
                                convert_xgboost)


def seed_random_generators(random_seed):
    """
    Make the NumPy and PyTorch operations deterministic using `random_seed` as an RNG seed.

    :param random_seed: the random seed
    """

    random_seed = random_seed % (2 ** 32)  # coerce to valid range of values

    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)


def get_available_devices():
    """
    Get all the available devices for training neural networks.

    :return: list of available devices
    """
    devices = ['cpu']
    if torch.cuda.is_available():
        devices.append('cuda')
    devices.extend(f'cuda:{index}' for index in range(torch.cuda.device_count()))

    return devices
