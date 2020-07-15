from random import random

import numpy as np
import torchvision.transforms as transforms
from scipy.signal import resample_poly
from scipy.stats import zscore

from .signal import bandpass


class BandpassFilter:
    """
    Processes the signal using a band-pass Butterworth second-order section digital filter.

    Args:
        lowcut (float): The lower cut-off frequency.
        highcut (float): The upper cut-off frequency.
        order (int): The order of the filter.
    """

    def __init__(self, lowcut=0.5, highcut=40, order=5):
        self.lowcut = lowcut
        self.highcut = highcut
        self.order = order

    def __call__(self, data):
        signal, label, fs = data
        return bandpass(signal, self.lowcut, self.highcut, fs, self.order), label, fs


class Resample:
    """
    Resamples the given signal to the given frequency using polyphase filtering.

    Args:
        fs (int): Desired output frequency.
    """

    def __init__(self, fs):
        self.fs = fs

    def __call__(self, data):
        signal, label, orig_fs = data
        return resample_poly(signal, self.fs, orig_fs), label, self.fs


class Standardize:
    """
    Standardizes the signal to zero mean and unit variance.
    """

    def __call__(self, data):
        signal, label, fs = data
        if np.std(signal) == 0:
            return data
        return zscore(signal, axis=-1), label, fs


class RandomFlip:
    """
    Randomly flips the sign of the signal.

    Args:
        p (float): Probability of flipping the signal.
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        signal, label, fs = data
        if random() < self.p:
            signal = -signal
        return signal, label, fs
