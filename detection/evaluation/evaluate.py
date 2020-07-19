from os import path

import pandas as pd

from detection.utils.filesystem import ensure_directory_exists
from detection.utils.time import timestamp
from detection.export.persistence import load_model


def evaluate(models, datasets, output_dir, model_dir, split='test'):
    """
    Runs the evaluation of every model in `models` on every dataset from `datasets`.

    :param models: list of models deriving from `BaseModel`
    :param datasets: list of datasets deriving from `BaseDataset`
    :param output_dir: the output directory for the evaluation .csv file
    :param split: the split (train, val, test) of the dataset to use
    """

    results = {
        'model': [],
        'dataset': [],
        'score': []
    }

    for dataset in datasets:
        eval_set = dataset.get_split(split)

        for model in models:
            model = load_model(model, dataset, model_dir)

            print(f'Evaluating model {model.name()} on dataset {dataset.name()}, \'{split}\' partition...')

            score = model.score(eval_set, cv=False)
            print(score)

            results['model'].append(model.name())
            results['dataset'].append(dataset.name())
            results['score'].append(score)

    results = pd.DataFrame(results)
    print(results)

    ensure_directory_exists(output_dir)

    output_filename = path.join(output_dir, f'results_{timestamp()}.csv')
    results.to_csv(output_filename)
    print(f'Evaluation results saved to {output_filename}.')
