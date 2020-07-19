import argparse
from pprint import pprint

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from tslearn.neighbors import KNeighborsTimeSeriesClassifier

from os import getcwd
from .evaluation import evaluate
from .export.onnx import export_onnx_models
from .models import *
from .preprocessing.dataset import *
from .preprocessing.features import extract_features_datasets
from .preprocessing.split import split_and_save_datasets
from .preprocessing.transforms import BandpassFilter
from .training import train
from .utils.constants import RANDOM_SEED
from .utils.filesystem import data_dir, saved_models_dir, results_dir
from .utils.setup import *
from .visualization.images import ImageGenerator
from .models.feature_based import get_default_rf_pipeline

def get_args():
    """
    Parse the program's arguments and return them.
    """

    parser = argparse.ArgumentParser(
        prog='python -m detection',
        description='ECG Arrhythmia Detection Comparison',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--root-dir', type=str, default=getcwd(),
                        help='The root directory for datasets and data. '
                             'All downloaded and extracted data are saved under this directory. (default: %(default)s)')
    parser.add_argument('--generate-plots', action='store_true',
                        help='Generate all plots and figures for the thesis.')
    parser.add_argument('--include-private', action='store_true',
                        help='Include private dataset in all experiments (private data must be in directory '
                             '\'datasets\').')
    parser.add_argument('--random-seed', type=int, default=RANDOM_SEED,
                        help='The random seed for seeding the PyTorch and NumPy random number generators.')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        choices=get_available_devices(),
                        help='The device for training and inference of PyTorch models.')
    parser.add_argument('--include-knn', action='store_true',
                        help='Include the K-NN DTW time series classifier. Warning: the evaluation takes a very long '
                             'time with this classifier enabled.')
    parser.add_argument('--export-onnx', action='store_true',
                        help='Export the ONNX models for use in inference applications.')
    parser.add_argument('--save-trained-models', action='store_false',
                        help='Save the trained models for future application launches.')
    parser.add_argument('--retrain', action='store_true',
                        help='Force retraining of the models even if a saved trained model already exists.')

    return parser.parse_args()


def init(random_seed):
    """
    Do the one-time setup for the third-party libraries.
    """

    enable_progress_apply()
    enable_xgboost_onnx_support()
    seed_random_generators(random_seed)


def get_models(device, include_knn=False):
    """
    Get the models used in the comparison.

    :param device: a PyTorch device used for training / inference
    :param include_knn: True if k-NN DTW model should be used
    :return: list of models
    """

    models = [
        FeatureBasedClassifier(LogisticRegression(
            n_jobs=-1,
            class_weight='balanced',
            C=100.0,
            max_iter=1000
        )),
        FeatureBasedClassifier(RandomForestClassifier(
            class_weight='balanced',
            max_depth=20,
            min_samples_leaf=5,
            n_estimators=1000,
            n_jobs=-1,
        ), pipeline=get_default_rf_pipeline()),
        FeatureBasedClassifier(XGBClassifier(
            colsample_bytree=0.8,
            gamma=0.1,
            learning_rate=0.1,
            max_depth=7,
            min_child_weight=4,
            n_estimators=1000,
            nthread=8,
            subsample=0.8,
        ), pipeline=None),
        FeatureBasedClassifier(MLPClassifier(
            early_stopping=True,
            hidden_layer_sizes=(512,),
            batch_size=128
        )),
        SignalBasedClassifier(
            ResNet18, device,
            batch_size=32,
            epochs=20,
            optimizer_args=dict(
                lr=3e-4,
                weight_decay=1e-4
            )
        ),
        SignalBasedClassifier(
            CnnGru, device,
            batch_size=32,
            epochs=10,
            gru_dropout=0,
            gru_layers=1,
            optimizer_args=dict(
                lr=3e-4,
                weight_decay=1e-4
            )
        )
    ]

    if include_knn:
        models.append(DistanceBasedClassifier(
            KNeighborsTimeSeriesClassifier(
                metric='dtw',
                n_jobs=-1,
                n_neighbors=1,
            )
        ))

    return models


def get_datasets(root_dir, include_private=False):
    """
    Get the datasets used in the comparison.

    :param root_dir: the root directory of the program
    :param include_private: True if the private dataset should be included in the copmarison
    :return:
    """

    datasets = [
        Cinc2017Dataset(root_dir=root_dir, transform=None),
        Cpsc2018Dataset(root_dir=root_dir, transform=BandpassFilter()),
    ]

    if include_private:
        datasets.append(PrivateDataset(root_dir=root_dir, transform=BandpassFilter()))
    return datasets


def main():
    args = get_args()
    init(args.random_seed)

    print('Arguments:', args)

    device = torch.device(args.device)
    datasets = get_datasets(args.root_dir, args.include_private)
    models = get_models(device, args.include_knn)
    print('Datasets:', end=' ')
    pprint(datasets)
    print('Models:', end=' ')
    pprint(models)

    split_and_save_datasets(datasets, output_dir=data_dir(args.root_dir))
    extract_features_datasets(datasets, output_dir=data_dir(args.root_dir))

    if args.generate_plots:
        ImageGenerator(args.root_dir).generate_images()

    trained_models = train(datasets, models, saved_models_dir(args.root_dir), load_models=not args.retrain,
                           save_models=args.save_trained_models)
    evaluate(trained_models, datasets, results_dir(args.root_dir), saved_models_dir(args.root_dir))

    if args.export_onnx:
        export_onnx_models(trained_models, saved_models_dir(args.root_dir))
