from os import path

from sklearn.model_selection import train_test_split, GroupShuffleSplit

from detection.utils.filesystem import ensure_directory_exists


def stratify_split(df, val_size, test_size, stratify_column, random_state):
    df_trainval, df_test = train_test_split(df, stratify=df[stratify_column], test_size=test_size,
                                            random_state=random_state)
    df_train, df_val = train_test_split(df_trainval, stratify=df_trainval[stratify_column], test_size=val_size,
                                        random_state=random_state)

    return df_train, df_val, df_test


def group_split(df, val_size, test_size, group_column, random_state):
    split = GroupShuffleSplit(test_size=test_size / len(df), random_state=random_state)
    trainval_idx, test_idx = next(split.split(df, groups=df[group_column].values))
    df_trainval = df.iloc[trainval_idx]

    val_split = GroupShuffleSplit(test_size=val_size / len(df_trainval), random_state=random_state)
    train_idx, val_idx = next(val_split.split(trainval_idx, groups=df_trainval[group_column].values))
    train_idx = trainval_idx[train_idx]
    val_idx = trainval_idx[val_idx]

    return df.iloc[train_idx], df.iloc[val_idx], df.iloc[test_idx]


def train_val_test_split(df, val_size, test_size, stratify_column=None, group_column=None, random_state=42):
    val_len = int(len(df) * val_size)
    test_len = int(len(df) * test_size)

    if group_column is not None:
        df_train, df_val, df_test = group_split(df, val_len, test_len, group_column, random_state)
    else:
        df_train, df_val, df_test = stratify_split(df, val_len, test_len, stratify_column, random_state)

    df.loc[df.Record.isin(df_train.Record), 'Split'] = 'train'
    df.loc[df.Record.isin(df_val.Record), 'Split'] = 'val'
    df.loc[df.Record.isin(df_test.Record), 'Split'] = 'test'
    df.Split = df.Split.astype('category')


def split_and_save_datasets(datasets, output_dir, val_size=0.1, test_size=0.1):
    """
    Create a new column called `Split`, split the data into train-val-test split and save them to `output_directory`.

    :param datasets: a collection of `BaseDataset`s
    :param output_dir: the split dataset output directory
    """
    print('Splitting datasets...')
    ensure_directory_exists(output_dir)

    for dataset in datasets:
        name = dataset.name()
        if dataset.dataset_exists():
            print(f'Dataset {name} is already split. Skipping...')
            continue

        print(f'Splitting dataset {name}...')

        df = dataset.get_dataframe()
        train_val_test_split(df, val_size, test_size, dataset.get_label_column(), dataset.get_group_column())

        output_filename = path.join(output_dir, f'{name}.pkl')

        df.to_pickle(output_filename,
                     protocol=4)  # Pickle protocol 4 ensures compatibility with Python < 3.8
        print(f'Saved split dataset to {output_filename}.')
