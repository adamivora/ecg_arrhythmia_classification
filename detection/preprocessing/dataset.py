import os
import shutil
from abc import ABC, abstractmethod
from os import path

import numpy as np
import pandas as pd
import torchvision.transforms as transforms
from scipy.io import loadmat
from sklearn.utils import compute_class_weight
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_and_extract_archive

from detection.utils.filesystem import ensure_directory_exists, data_dir, datasets_dir

__all__ = ['Cinc2017Dataset', 'Cpsc2018Dataset', 'PrivateDataset']


class BaseDataset(ABC, Dataset):
    """
    The base dataset that all the other datasets derive from. Contains useful helper functions.
    """
    resources = {}
    label_column = 'Label'
    record_column = 'Record'

    def __init__(self, root_dir, transform=None, data=None, split=None):
        self.transform = transform
        self.root_dir = root_dir
        if data is not None:
            self.data = data
        else:
            self.init()
        if split is not None:
            cond = self.data.Split == split if split != 'trainval' else self.data.Split.isin(['train', 'val'])
            self.data = self.data[cond].reset_index(drop=True)
        self.data[self.label_column] = self.data[self.label_column].astype('category')
        self.data['LabelCode'] = self.data[self.label_column].cat.codes

    def __len__(self):
        return len(self.get_dataframe())

    def __getitem__(self, item):
        if isinstance(item, slice):
            items = (self[i] for i in range(*item.indices(len(self))))
            return tuple(list(x) for x in zip(*items))

        row = self.data.iloc[item]
        data = self.read_record(row[self.record_column]), row['LabelCode'], row['Fs']
        if self.transform:
            return self.transform(data)
        return data

    def __iter__(self):
        return iter(self[i] for i in range(len(self)))

    def __repr__(self):
        return self.name()

    @abstractmethod
    def init(self):
        pass

    @abstractmethod
    def read_record(self, record):
        pass

    def get_dataframe(self):
        return self.data

    def get_label_column(self):
        return self.label_column

    def data_exist(self):
        return path.exists(self.root_dir)

    def dataset_exists(self):
        return path.isfile(path.join(data_dir(self.root_dir), f'{self.name()}.pkl'))

    def features_exist(self):
        return path.isfile(path.join(data_dir(self.root_dir), f'{self.name()}Features.pkl'))

    def name(self):
        return type(self).__name__

    def download(self):
        if self.data_exist():
            return

        download_dir = path.join(datasets_dir(self.root_dir), 'download')
        ensure_directory_exists(download_dir)

        for filename, (url, md5) in self.resources.items():
            download_and_extract_archive(url, download_root=download_dir, filename=filename, md5=md5,
                                         remove_finished=False)

        print('Done!')

    def get_split(self, split):
        return type(self)(root_dir=self.root_dir, transform=self.transform, data=self.data, split=split)

    def get_splits(self, cv=False):
        splits = ['trainval', 'test'] if cv else ['train', 'val', 'test']
        return [self.get_split(split) for split in splits]

    def get_features(self):
        features = self.data.drop(columns=['Record', 'Split', 'Fs', self.label_column, 'LabelCode'])
        labels = self.data[self.label_column]
        features.replace([np.inf, -np.inf], np.nan, inplace=True)
        features.dropna(axis=1, how='all', inplace=True)
        return features, labels

    def get_signals(self, transform):
        orig_transform = self.transform
        self.transform = transforms.Compose([orig_transform, transform]) if orig_transform is not None else transform

        X, y, fs = self[:]

        self.transform = orig_transform
        return X, y, fs

    def get_labels(self):
        return self.data['LabelCode']

    def get_num_classes(self):
        labels = self.get_dataframe()[self.get_label_column()]
        return len(labels.cat.categories)

    def get_sample_weights(self):
        labels = self.get_dataframe()[self.get_label_column()]
        categories = labels.cat.categories
        class_weights = compute_class_weight('balanced', classes=categories, y=labels)
        sample_weights = class_weights[labels.cat.codes]
        return sample_weights


class Cinc2017Dataset(BaseDataset):
    """
    `The PhysioNet Computing in Cardiology Challenge 2017` dataset.
    Available from: https://physionet.org/content/challenge-2017/1.0.0/
    """

    resources = {
        'training2017.zip': (
            'https://www.physionet.org/files/challenge-2017/1.0.0/training2017.zip?download',
            '7a220c80e10881bad86045e8ce97b49d'
        )
    }

    def __init__(self, root_dir='.', transform=None, data=None, split=None):
        super().__init__(root_dir, transform, data, split)

    def init(self):
        if not self.data_exist():
            self.download()
            src_dir = path.join(datasets_dir(self.root_dir), 'download', 'training2017')
            shutil.move(src_dir, self.root_dir)
            print(f'Successfully downloaded dataset {self.name()} to {self.root_dir}.')

        if self.features_exist():
            self.data = pd.read_pickle(path.join(data_dir(self.root_dir), f'{self.name()}Features.pkl'))
            return

        if self.dataset_exists():
            self.data = pd.read_pickle(path.join(data_dir(self.root_dir), f'{self.name()}.pkl'))
        else:
            self.data = pd.read_csv(path.join(datasets_dir(self.root_dir), self.name(), 'REFERENCE.csv'),
                                    names=['Record', 'Label'],
                                    dtype={'Label': 'category'})
            self.data['Fs'] = 300

    def read_record(self, record):
        mat = loadmat(path.join(datasets_dir(self.root_dir), self.name(), record))
        signal = mat['val'][0]

        return signal.astype(np.float32)


class Cpsc2018Dataset(BaseDataset):
    """
    `The China Physiological Signal Challenge 2018` dataset.
    Available from: http://2018.icbeb.org/Challenge.html
    """
    resources = {
        'TrainingSet1.zip': (
            'http://hhbucket.oss-cn-hongkong.aliyuncs.com/TrainingSet1.zip',
            'edf7962902e8bd75e6f6a26658fbae35'
        ),
        'TrainingSet2.zip': (
            'http://hhbucket.oss-cn-hongkong.aliyuncs.com/TrainingSet2.zip',
            '79e36236cb781ebecd5eb3e2ca132608'
        ),
        'TrainingSet3.zip': (
            'http://hhbucket.oss-cn-hongkong.aliyuncs.com/TrainingSet3.zip',
            'd49dc5e0cf3f16c5e291590ded922cc7'
        )
    }

    def __init__(self, root_dir='.', transform=None, data=None, split=None, lead=0):
        super().__init__(root_dir, transform, data, split)
        self.lead = lead

    def init(self):
        if not self.data_exist():
            self.download()
            for dir in ['TrainingSet1', 'TrainingSet2', 'TrainingSet3']:
                src_dir = path.join(datasets_dir(self.root_dir), 'download', dir)
                shutil.copytree(src_dir, self.root_dir, dirs_exist_ok=True, copy_function=shutil.move)
                os.rmdir(src_dir)
            print(f'Successfully downloaded dataset {self.name()} to {self.root_dir}.')

        if self.features_exist():
            self.data = pd.read_pickle(path.join(data_dir(self.root_dir), f'{self.name()}Features.pkl'))
            return

        if self.dataset_exists():
            self.data = pd.read_pickle(path.join(data_dir(self.root_dir), f'{self.name()}.pkl'))
        else:
            self.data = pd.read_csv(path.join(datasets_dir(self.root_dir), self.name(), 'REFERENCE.csv'),
                                    dtype={'First_label': 'Int64',
                                           'Second_label': 'Int64',
                                           'Third_label': 'Int64'})
            self.data.columns = ['Record', 'Label', 'Label2', 'Label3']
            self.data = self.data[self.data['Label2'].isnull()].reset_index(drop=True)
            self.data['Fs'] = 500

    def read_record(self, record):
        mat = loadmat(path.join(datasets_dir(self.root_dir), self.name(), record))
        signal = mat['ECG'][0, 0][2][self.lead]
        return signal.astype(np.float32)


class PrivateDataset(BaseDataset):
    """
    The private ECG dataset.
    """
    def __init__(self, root_dir='.', transform=None, data=None, split=None):
        super().__init__(root_dir, transform, data, split)
        self.signals = pd.read_pickle(path.join(datasets_dir(self.root_dir), self.name(), 'private_dataset.pkl'))

    def init(self):
        if self.features_exist():
            self.data = pd.read_pickle(path.join(data_dir(self.root_dir), f'{self.name()}Features.pkl'))
            return

        if self.dataset_exists():
            self.data = pd.read_pickle(path.join(data_dir(self.root_dir), f'{self.name()}.pkl'))
        else:
            self.data = pd.read_pickle(
                path.join(datasets_dir(self.root_dir), self.name(), 'private_dataset.pkl')
            ).filter(['Diagnose', 'Signal'])
            self.data.columns = ['Label', 'Signal']

            self.data = self.data.filter(['Label'])
            self.data['Fs'] = 200
            self.data['Record'] = self.data.index

    def read_record(self, record):
        row = self.signals.loc[record]
        return np.array(row.Signal, dtype=np.float32)
