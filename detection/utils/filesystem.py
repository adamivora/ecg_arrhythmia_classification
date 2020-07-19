import os
from os import path
from pathlib import Path


def ensure_directory_exists(dir):
    Path(dir).mkdir(exist_ok=True, parents=True)

def data_dir(root_dir):
    return path.join(root_dir, 'data')


def datasets_dir(root_dir):
    return path.join(root_dir, 'datasets')


def images_dir(root_dir):
    return path.join(root_dir, 'images')


def saved_models_dir(root_dir):
    return path.join(data_dir(root_dir), 'saved_models')


def results_dir(root_dir):
    return path.join(root_dir, 'results')
