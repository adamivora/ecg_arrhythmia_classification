import argparse
from pprint import pprint

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.neural_network import MLPClassifier
from tslearn.neighbors import KNeighborsTimeSeriesClassifier

from .models import FeatureBasedClassifier, DistanceBasedClassifier, NNBasedClassifier, resnet18, RNN
from .preprocessing.dataset import *
from .preprocessing.features import extract_features_datasets
from .preprocessing.split import split_and_save_datasets
from .preprocessing.transforms import BandpassFilter
from .training.train import train
from .utils.constants import RANDOM_SEED, ROOT_DIR
from .utils.filesystem import data_dir, saved_models_dir
from .utils.setup import *
from .visualization.images import ImageGenerator
from .export.onnx import export_onnx_models

def get_args():
    parser = argparse.ArgumentParser(
        prog='python -m detection',
        description='ECG Arrhythmia Detection Comparison',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--root-dir', type=str, default=ROOT_DIR,
                        help='The root directory for datasets and data.'
                             'All downloaded and extracted data are saved under this directory. (default: %(default)s)')
    parser.add_argument('--generate-plots', action='store_true',
                        help='Generate all plots and figures for the thesis.')
    parser.add_argument('--include-private', action='store_true',
                        help='Include private dataset in all experiments (private data must be in directory '
                             '\'datasets\').')
    parser.add_argument('--random-seed', type=int, default=RANDOM_SEED,
                        help='The random seed for seeding the PyTorch and NumPy random number generators.')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='The device for training and inference of PyTorch models.')
    parser.add_argument('--include-knn', action='store_true',
                        help='Include the K-NN DTW time series classifier. Warning: the evaluation takes a very long '
                             'time with this classifier enabled.')
    parser.add_argument('--export', action='store_true',
                        help='Export the ONNX models for use in other frameworks.')
    # parser.add_argument('--cv', type=int, default=RANDOM_SEED,
    #                    help='The random seed for seeding the PyTorch and NumPy random number generators.')

    return parser.parse_args()


def init(random_seed):
    """
    Do the one-time setup for all things needed.
    """
    enable_progress_apply()
    enable_xgboost_onnx_support()
    seed_random_generators(random_seed)


def main():
    args = get_args()
    init(args.random_seed)

    print('Arguments:', args)

    datasets = [
        Cinc2017Dataset(root_dir=args.root_dir, transform=None),
        Cpsc2018Dataset(root_dir=args.root_dir, transform=BandpassFilter()),
    ]

    if args.include_private:
        datasets.append(PrivateDataset(root_dir=args.root_dir, transform=BandpassFilter()))

    if args.generate_plots:
        ImageGenerator(args.root_dir).generate_images()

    split_and_save_datasets(datasets, output_dir=data_dir(args.root_dir))
    extract_features_datasets(datasets, output_dir=data_dir(args.root_dir))

    device = torch.device(args.device)
    models = [
        FeatureBasedClassifier(LogisticRegressionCV(n_jobs=-1, class_weight='balanced')),
        FeatureBasedClassifier(RandomForestClassifier(n_jobs=-1, class_weight='balanced')),
        FeatureBasedClassifier(XGBClassifier()),
        FeatureBasedClassifier(MLPClassifier(hidden_layer_sizes=(512,), early_stopping=True)),
        NNBasedClassifier(resnet18, device, batch_size=32, epochs=1),
        NNBasedClassifier(RNN, device, batch_size=32, epochs=1)
    ]

    print('Datasets: ')
    pprint(datasets)
    print('Models: ')
    pprint(models)

    if args.include_knn:
        models.append(DistanceBasedClassifier(KNeighborsTimeSeriesClassifier(n_jobs=-1, n_neighbors=1, metric='dtw')))

    train(datasets, models)

    if args.export:
        export_onnx_models(models, saved_models_dir(args.root_dir))
