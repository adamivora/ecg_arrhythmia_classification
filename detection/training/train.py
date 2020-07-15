def train(datasets, models):
    for dataset in datasets:
        trainval_set, test_set = dataset.get_splits(cv=True)
        val_set = dataset.get_split('val')

        sample_weights = trainval_set.get_sample_weights()

        for model in models:
            print(f'Training model {model.name()} on dataset {dataset.name()}...')
            train_args = {}
            if model.name() == 'XGBClassifier':
                train_args['xgbclassifier__sample_weight'] = sample_weights

            model.train(trainval_set, **train_args)
            print(f'Computing validation score...')
            if model.cv:
                score = model.score(trainval_set, cv=True)
                print(score)
                #print(f'{score.mean:.4f} +- {score.dev:.4f} - {score.data}')
            else:
                print(model.score(val_set, cv=False))
