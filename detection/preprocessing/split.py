from os import path

from sklearn.model_selection import train_test_split

from detection.utils.filesystem import ensure_directory_exists


def train_val_test_split(df, stratify_column, val_size, test_size, random_state=42, **kwargs):
    val_len = int(len(df) * val_size)
    test_len = int(len(df) * test_size)

    df_trainval, df_test = train_test_split(df, stratify=df[stratify_column], test_size=test_len,
                                            random_state=random_state, **kwargs)
    df_train, df_val = train_test_split(df_trainval, stratify=df_trainval[stratify_column], test_size=val_len,
                                        random_state=random_state, **kwargs)
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
        train_val_test_split(df, dataset.get_label_column(), val_size, test_size)

        output_filename = path.join(output_dir, f'{name}.pkl')

        df.to_pickle(output_filename,
                     protocol=4)  # Pickle protocol 4 ensures compatibility with Python < 3.8
        print(f'Saved split dataset to {output_filename}.')
