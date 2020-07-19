from detection.export.persistence import save_model, load_model, trained_model_exists


def train(datasets, models, saved_models_dir=None, load_models=True, save_models=True, calculate_score=False):
    assert not save_models or saved_models_dir is not None, f'saved_models_dir has to be specified when saving models.'

    for dataset in datasets:
        trainval_set, test_set = dataset.get_splits(cv=True)
        val_set = dataset.get_split('val')

        sample_weights = trainval_set.get_sample_weights()

        for idx in range(len(models)):
            model = models[idx]

            if load_models and trained_model_exists(model, dataset, saved_models_dir):
                model = load_model(model, dataset, saved_models_dir)
                models[idx] = model
                print(f'Skipping training model {model.name()} on dataset {dataset.name()}.')
                continue

            print(f'Training model {model.name()} on dataset {dataset.name()}...')
            train_args = {}
            if model.name() == 'XGBClassifier':
                train_args['xgbclassifier__sample_weight'] = sample_weights

            model.train(trainval_set, **train_args)

            if calculate_score:
                print(f'Computing validation score...')
                if model.cv:
                    print(model.score(trainval_set, cv=True))
                else:
                    print(model.score(val_set, cv=False))

            if save_models:
                save_model(model, dataset, saved_models_dir)

    return models
