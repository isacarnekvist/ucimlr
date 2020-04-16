import os
import sys
import inspect

import pandas as pd
from unlzw import unlzw

from ucimlr.helpers import (download_file, download_unzip, one_hot_encode_df_, xy_split, label_encode_df_,
                            clean_na_, normalize_df_, split_normalize_sequence, split_df, get_split,
                            split_classification_df)
from ucimlr.dataset import Dataset
from ucimlr.constants import TRAIN
from ucimlr.constants import CLASSIFICATION


def all_datasets():
    """
    Returns a list of all ClassificationDataset classes.
    """
    return [cls for _, cls in inspect.getmembers(sys.modules[__name__])
            if inspect.isclass(cls)
            and issubclass(cls, ClassificationDataset)
            and cls != ClassificationDataset]


class ClassificationDataset(Dataset):
    type_ = CLASSIFICATION  # Is this necessary?

    @property
    def num_classes(self):
        raise NotImplementedError('A classification dataset needs to have attribute `num_classes`.')


class Adult(ClassificationDataset):
    """
    # Parameters
    root (str): Local path for storing/reading dataset files.
    split (str): One of {'train', 'validation', 'test'}
    validation_size (float): How large fraction in (0, 1) of the training partition to use for validation.
    """
    num_classes = 2

    def __init__(self, root, split=TRAIN, validation_size=0.2):
        dataset_path = os.path.join(root, self.name)
        filename_train = 'data_train.csv'
        filename_test = 'data_test.csv'
        file_path_train = os.path.join(dataset_path, filename_train)
        file_path_test = os.path.join(dataset_path, filename_test)
        url_train = 'http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
        url_test = 'http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test'
        download_file(url_train, dataset_path, filename_train)
        download_file(url_test, dataset_path, filename_test)

        df_train_valid = pd.read_csv(file_path_train, header=None, skiprows=0)
        df_test = pd.read_csv(file_path_test, header=None, skiprows=1)

        # Trailing period in test file
        df_test[14] = df_test[14].str.rstrip('.')

        df_test.index += len(df_train_valid)
        df = pd.concat([df_train_valid, df_test])
        y_columns = df.columns[-1:]
        one_hot_encode_df_(df, skip_columns=y_columns)
        label_encode_df_(df, y_columns[0])
        df_train_valid, df_test = (df.loc[df_train_valid.index], df.loc[df_test.index])
        df_train, df_valid = split_df(df_train_valid, [1 - validation_size, validation_size])
        normalize_df_(df_train, other_dfs=[df_valid, df_test], skip_column=y_columns[0])
        df_res = get_split(df_train, df_valid, df_test, split)
        self.x, self.y = xy_split(df_res, y_columns)
        self.y = self.y[:, 0]  # Flatten for classification


class APSFailure(ClassificationDataset):
    """
    # Parameters
    root (str): Local path for storing/reading dataset files.
    split (str): One of {'train', 'validation', 'test'}
    validation_size (float): How large fraction in (0, 1) of the training partition to use for validation.
    """
    num_classes = 2

    def __init__(self, root, split=TRAIN, validation_size=0.2):
        file_name_train = 'train.csv'
        file_name_test = 'test.csv'
        dataset_path = os.path.join(root, self.name)
        url_train = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00421/aps_failure_training_set.csv'
        url_test = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00421/aps_failure_test_set.csv'
        download_file(url_train, dataset_path, file_name_train)
        download_file(url_test, dataset_path, file_name_test)
        file_path_train = os.path.join(dataset_path, file_name_train)
        file_path_test = os.path.join(dataset_path, file_name_train)
        df_train_valid = pd.read_csv(file_path_train, skiprows=20, na_values='na')
        df_test = pd.read_csv(file_path_test, skiprows=20, na_values='na')

        # TODO This is risky business since test and train might be cleaned to have different columns
        clean_na_(df_train_valid)
        clean_na_(df_test)
        if not (df_train_valid.columns == df_test.columns).all():
            raise Exception('Cleaning lead to different set of columns for train/test')

        y_columns = ['class']
        label_encode_df_([df_train_valid, df_test], y_columns[0])
        df_train, df_valid = split_df(df_train_valid, [1 - validation_size, validation_size])
        normalize_df_(df_train, other_dfs=[df_valid, df_test], skip_column=y_columns[0])
        df_res = get_split(df_train, df_valid, df_test, split)
        self.x, self.y = xy_split(df_res, y_columns)
        self.y = self.y[:, 0]


class Avila(ClassificationDataset):
    """
    # Parameters
    root (str): Local path for storing/reading dataset files.
    split (str): One of {'train', 'validation', 'test'}
    validation_size (float): How large fraction in (0, 1) of the training partition to use for validation.
    """
    num_classes = 12

    def __init__(self, root, split=TRAIN, validation_size=0.2):
        dataset_path = os.path.join(root, self.name)
        url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00459/avila.zip'
        download_unzip(url, dataset_path)
        file_path_train = os.path.join(dataset_path, 'avila', 'avila-tr.txt')
        file_path_test = os.path.join(dataset_path, 'avila', 'avila-ts.txt')
        df_train_valid = pd.read_csv(file_path_train, header=None)
        df_test = pd.read_csv(file_path_test, header=None)
        y_columns = [10]
        label_encode_df_([df_train_valid, df_test], y_columns[0])  # Assumes encoding will be identical for train/test
        df_train, df_valid = split_classification_df(df_train_valid,
                                                     [1 - validation_size, validation_size],
                                                     y_columns[0])
        normalize_df_(df_train, other_dfs=[df_valid, df_test], skip_column=y_columns[0])
        df_res = get_split(df_train, df_valid, df_test, split)
        self.x, self.y = xy_split(df_res, y_columns)
        self.y = self.y[:, 0]


class BankMarketing(ClassificationDataset):
    """
    # Parameters
    root (str): Local path for storing/reading dataset files.
    split (str): One of {'train', 'validation', 'test'}
    validation_size (float): How large fraction in (0, 1) of the training partition to use for validation.
    """
    num_classes = 2

    def __init__(self, root, split=TRAIN, validation_size=0.2):
        dataset_path = os.path.join(root, self.name)
        url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip'
        download_unzip(url, dataset_path)
        file_path = os.path.join(dataset_path, 'bank-additional', 'bank-additional-full.csv')
        df = pd.read_csv(file_path, sep=';')
        y_columns = ['y']
        one_hot_encode_df_(df, skip_columns=y_columns)
        label_encode_df_(df, y_columns[0])
        self.x, self.y = split_normalize_sequence(df, y_columns, validation_size, split, self.type_)


class CardDefault(ClassificationDataset):
    """
    # Parameters
    root (str): Local path for storing/reading dataset files.
    split (str): One of {'train', 'validation', 'test'}
    validation_size (float): How large fraction in (0, 1) of the training partition to use for validation.
    """
    num_classes = 2

    def __init__(self, root, split=TRAIN, validation_size=0.2):
        dataset_path = os.path.join(root, self.name)
        filename = 'data.xls'
        file_path = os.path.join(dataset_path, filename)
        url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00350/' \
              'default%20of%20credit%20card%20clients.xls'
        download_file(url, dataset_path, filename)
        df = pd.read_excel(file_path, skiprows=1, index_col='ID')
        y_columns = ['default payment next month']
        self.x, self.y = split_normalize_sequence(df, y_columns, validation_size, split, self.type_)


class Landsat(ClassificationDataset):
    """
    # Parameters
    root (str): Local path for storing/reading dataset files.
    split (str): One of {'train', 'validation', 'test'}
    validation_size (float): How large fraction in (0, 1) of the training partition to use for validation.
    """
    num_classes = 6

    def __init__(self, root, split=TRAIN, validation_size=0.2):
        dataset_path = os.path.join(root, self.name)
        file_name_train = 'train.csv'
        file_name_test = 'test.csv'
        url_train = 'http://archive.ics.uci.edu/ml/machine-learning-databases/statlog/satimage/sat.trn'
        url_test = 'http://archive.ics.uci.edu/ml/machine-learning-databases/statlog/satimage/sat.tst'
        download_file(url_train, dataset_path, file_name_train)
        download_file(url_test, dataset_path, file_name_test)
        file_path_train = os.path.join(dataset_path, file_name_train)
        file_path_test = os.path.join(dataset_path, file_name_test)
        df_train_valid = pd.read_csv(file_path_train, sep=' ', header=None)
        df_test = pd.read_csv(file_path_test, sep=' ', header=None)
        df_test.index += len(df_train_valid)
        df = pd.concat([df_train_valid, df_test])
        y_columns = [36]
        label_encode_df_(df, y_columns[0])
        df_train_valid = df.loc[df_train_valid.index, :]
        df_test = df.loc[df_test.index, :]
        df_train, df_valid = split_classification_df(df_train_valid, [1 - validation_size, validation_size], 36)
        normalize_df_(df_train, other_dfs=[df_valid, df_test], skip_column=y_columns[0])
        df_res = get_split(df_train, df_valid, df_test, split)
        self.x, self.y = xy_split(df_res, y_columns)
        self.y = self.y[:, 0]


class LetterRecognition(ClassificationDataset):
    """
    # Parameters
    root (str): Local path for storing/reading dataset files.
    split (str): One of {'train', 'validation', 'test'}
    validation_size (float): How large fraction in (0, 1) of the training partition to use for validation.
    """
    num_classes = 26

    def __init__(self, root, split=TRAIN, validation_size=0.2):
        file_name = 'data.csv'
        dataset_path = os.path.join(root, self.name)
        url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/letter-recognition.data'
        download_file(url, dataset_path, file_name)
        file_path = os.path.join(dataset_path, file_name)
        df = pd.read_csv(file_path, header=None)
        y_columns = [0]
        label_encode_df_(df, y_columns[0])
        self.x, self.y = split_normalize_sequence(df, y_columns, validation_size, split, self.type_)


class MagicGamma(ClassificationDataset):
    """
    # Parameters
    root (str): Local path for storing/reading dataset files.
    split (str): One of {'train', 'validation', 'test'}
    validation_size (float): How large fraction in (0, 1) of the training partition to use for validation.
    """
    num_classes = 2

    def __init__(self, root, split=TRAIN, validation_size=0.2):
        file_name = 'data.csv'
        dataset_path = os.path.join(root, self.name)
        url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/magic/magic04.data'
        download_file(url, dataset_path, file_name)
        file_path = os.path.join(dataset_path, file_name)
        df = pd.read_csv(file_path, header=None)
        y_columns = [10]
        label_encode_df_(df, y_columns[0])
        self.x, self.y = split_normalize_sequence(df, y_columns, validation_size, split, self.type_)


class SensorLessDrive(ClassificationDataset):
    """
    # Parameters
    root (str): Local path for storing/reading dataset files.
    split (str): One of {'train', 'validation', 'test'}
    validation_size (float): How large fraction in (0, 1) of the training partition to use for validation.
    """
    num_classes = 11

    def __init__(self, root, split=TRAIN, validation_size=0.2):
        file_name = 'data.csv'
        dataset_path = os.path.join(root, self.name)
        url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00325/Sensorless_drive_diagnosis.txt'
        download_file(url, dataset_path, file_name)
        file_path = os.path.join(dataset_path, file_name)
        df = pd.read_csv(file_path, header=None, sep=' ')
        y_columns = [48]
        label_encode_df_(df, y_columns[0])
        self.x, self.y = split_normalize_sequence(df, y_columns, validation_size, split, self.type_)


class Shuttle(ClassificationDataset):
    """
    Description of dataset [here](http://archive.ics.uci.edu/ml/datasets/Statlog+(Shuttle)).

    # Parameters
    root (str): Local path for storing/reading dataset files.
    split (str): One of {'train', 'validation', 'test'}
    validation_size (float): How large fraction in (0, 1) of the training partition to use for validation.
    """
    num_classes = 7

    def __init__(self, root, split=TRAIN, validation_size=0.2):
        dataset_path = os.path.join(root, self.name)
        url_train = 'http://archive.ics.uci.edu/ml/machine-learning-databases/statlog/shuttle/shuttle.trn.Z'
        url_test = 'http://archive.ics.uci.edu/ml/machine-learning-databases/statlog/shuttle/shuttle.tst'
        file_name_train = 'train.csv'
        file_name_test = 'test.csv'
        file_path_train = os.path.join(dataset_path, file_name_train)
        file_path_test = os.path.join(dataset_path, file_name_test)
        file_name_z = 'train.z'
        fresh_download = download_file(url_train, dataset_path, file_name_z)
        if fresh_download:
            path_z = os.path.join(dataset_path, file_name_z)
            with open(path_z, 'rb') as f_in:
                with open(file_path_train, 'wb') as f_out:
                    f_out.write(unlzw(f_in.read()))
            download_file(url_test, dataset_path, file_name_test)
        df_train_valid = pd.read_csv(file_path_train, header=None, sep=' ')
        y_columns = [9]
        df_train, df_valid = split_classification_df(df_train_valid,
                                                     [1 - validation_size, validation_size],
                                                     y_columns[0])
        df_test = pd.read_csv(file_path_test, header=None, sep=' ')
        label_encode_df_([df_train, df_valid, df_test], y_columns[0])
        normalize_df_(df_train, other_dfs=[df_valid, df_test], skip_column=y_columns[0])
        df_res = get_split(df_train, df_valid, df_test, split)
        self.x, self.y = xy_split(df_res, y_columns)
        self.y = self.y[:, 0]
