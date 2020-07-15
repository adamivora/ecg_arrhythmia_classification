import os
from os import path


def ensure_directory_exists(dir):
    if not path.exists(dir):
        os.mkdir(dir)


def data_dir(root_dir):
    return path.join(root_dir, 'data')


def datasets_dir(root_dir):
    return path.join(root_dir, 'datasets')


def images_dir(root_dir):
    return path.join(root_dir, 'images')


def saved_models_dir(root_dir):
    return path.join(data_dir(root_dir), 'saved_models')