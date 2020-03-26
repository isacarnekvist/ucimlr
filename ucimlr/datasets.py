import os
import abc

import numpy as np
import pandas as pd
from unlzw import unlzw

from ucimlr.helpers import (download_file, download_unzip, one_hot_encode_df_, xy_split, label_encode_df_,
                            clean_na_, normalize_df_, train_test_split, split_normalize_sequence)

# Dataset types
REGRESSION = 'regression'
CLASSIFICATION = 'classification'


class Dataset(abc.ABC):
    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, i):
        return self.x[i], self.y[i]

    @property
    def name(self):
        return self.__class__.__name__

    @property
    def num_features(self):
        return self.x.shape[1]

    @property
    def num_classes(self):
        if self.type_ == REGRESSION:
            raise ValueError('num_classes is only valid for classification, see num_targets')
        else:
            return len(set(self.y))  # TODO This is not a good way apparently

    @property
    def num_targets(self):
        if self.type_ == CLASSIFICATION:
            raise ValueError('num_targets is only valid for regression, see num_classes')
        else:
            return self.y.shape[1]


class Abalone(Dataset):
    """
    Link to the dataset [description](https://archive.ics.uci.edu/ml/datasets/Abalone).

    # Parameters
    root (str): Local path for storing/reading dataset files.
    train (bool): Whether to contain training or test set.
    """
    type_ = REGRESSION

    def __init__(self, root, train=True):
        dataset_path = os.path.join(root, self.name)
        filename = 'data.csv'
        file_path = os.path.join(dataset_path, filename)
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data'
        download_file(url, dataset_path, filename)
        df = pd.read_csv(file_path, header=None)
        y_columns = df.columns[-1:]
        one_hot_encode_df_(df)
        self.x, self.y = split_normalize_sequence(df, y_columns, train, self.type_)


class Adult(Dataset):
    """
    # Parameters
    root (str): Local path for storing/reading dataset files.
    train (bool): Whether to contain training or test set.
    """
    type_ = CLASSIFICATION

    def __init__(self, root, train=True):
        dataset_path = os.path.join(root, self.name)
        filename_train = 'data_train.csv'
        filename_test = 'data_test.csv'
        file_path_train = os.path.join(dataset_path, filename_train)
        file_path_test = os.path.join(dataset_path, filename_test)
        url_train = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
        url_test = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test'
        download_file(url_train, dataset_path, filename_train)
        download_file(url_test, dataset_path, filename_test)

        df_train = pd.read_csv(file_path_train, header=None, skiprows=0)
        df_test = pd.read_csv(file_path_test, header=None, skiprows=1)

        # Trailing period in test file
        df_test[14] = df_test[14].str.rstrip('.')

        df_test.index += len(df_train)
        df = pd.concat([df_train, df_test])
        y_columns = df.columns[-1:]
        one_hot_encode_df_(df, skip_columns=y_columns)
        label_encode_df_(df, y_columns[0])
        df_train, df_test = (df.loc[df_train.index], df.loc[df_test.index])
        normalize_df_(df_train, df_test=df_test, skip_column=y_columns[0])
        self.x, self.y = xy_split(df_train if train else df_test, y_columns)
        self.y = self.y[:, 0]  # Flatten for classification


class AirQuality(Dataset):
    """
    # Parameters
    root (str): Local path for storing/reading dataset files.
    train (bool): Whether to contain training or test set.
    """
    type_ = REGRESSION

    def __init__(self, root, train=True):
        dataset_path = os.path.join(root, self.name)
        filename = 'AirQualityUCI.csv'
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00360/AirQualityUCI.zip'
        download_unzip(url, dataset_path)
        file_path = os.path.join(dataset_path, filename)

        df = pd.read_csv(file_path, sep=';', parse_dates=[0, 1])
        df.dropna(axis=0, how='all', inplace=True)
        df.dropna(axis=1, how='all', inplace=True)

        df.Date = (df.Date - df.Date.min()).astype('timedelta64[D]')  # Days as int
        df.Time = df.Time.apply(lambda x: int(x.split('.')[0]))  # Hours as int
        df['C6H6(GT)'] = df['C6H6(GT)'].apply(lambda x: float(x.replace(',', '.')))  # Target as float

        # Some floats are given with ',' instead of '.'
        df = df.applymap(lambda x: float(x.replace(',', '.')) if type(x) is str else x)  # Target as float

        df = df[df['C6H6(GT)'] != -200]  # Drop all rows with missing target values
        df.loc[df['CO(GT)'] == -200, 'CO(GT)'] = -10  # -200 means missing value, shifting this to be closer to
        # the other values for this column

        y_columns = ['C6H6(GT)']
        self.x, self.y = split_normalize_sequence(df, y_columns, train, self.type_)


class APSFailure(Dataset):
    """
    # Parameters
    root (str): Local path for storing/reading dataset files.
    train (bool): Whether to contain training or test set.
    """
    type_ = CLASSIFICATION

    def __init__(self, root, train=True):
        file_name_train = 'train.csv'
        file_name_test = 'test.csv'
        dataset_path = os.path.join(root, self.name)
        url_train = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00421/aps_failure_training_set.csv'
        url_test = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00421/aps_failure_test_set.csv'
        download_file(url_train, dataset_path, file_name_train)
        download_file(url_test, dataset_path, file_name_test)
        file_path_train = os.path.join(dataset_path, file_name_train)
        file_path_test = os.path.join(dataset_path, file_name_train)
        df_train = pd.read_csv(file_path_train, skiprows=20, na_values='na')
        df_test = pd.read_csv(file_path_train, skiprows=20, na_values='na')

        # TODO This is risky business since test and train might be cleaned to have different columns
        clean_na_(df_train)
        clean_na_(df_test)
        if not (df_train.columns == df_test.columns).all():
            raise Exception('Cleaning lead to different set of columns for train/test')

        y_columns = ['class']
        label_encode_df_([df_train, df_test], y_columns[0])
        normalize_df_(df_train, df_test=df_test, skip_column=y_columns[0])
        self.x, self.y = xy_split(df_train if train else df_test, y_columns)
        self.y = self.y[:, 0]


class Avila(Dataset):
    """
    # Parameters
    root (str): Local path for storing/reading dataset files.
    train (bool): Whether to contain training or test set.
    """
    type_ = CLASSIFICATION

    def __init__(self, root, train=True):
        dataset_path = os.path.join(root, self.name)
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00459/avila.zip'
        download_unzip(url, dataset_path)
        file_path_train = os.path.join(dataset_path, 'avila', 'avila-tr.txt')
        file_path_test = os.path.join(dataset_path, 'avila', 'avila-ts.txt')
        df_train = pd.read_csv(file_path_train, header=None)
        df_test = pd.read_csv(file_path_test, header=None)
        y_columns = [10]
        label_encode_df_([df_train, df_test], y_columns[0])  # Assumes encoding will be identical for train/test
        normalize_df_(df_train, df_test=df_test, skip_column=y_columns[0])
        self.x, self.y = xy_split(df_train if train else df_test, y_columns)
        self.y = self.y[:, 0]


class BankMarketing(Dataset):
    """
    # Parameters
    root (str): Local path for storing/reading dataset files.
    train (bool): Whether to contain training or test set.
    """
    type_ = CLASSIFICATION

    def __init__(self, root, train=True):
        dataset_path = os.path.join(root, self.name)
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip'
        download_unzip(url, dataset_path)
        file_path = os.path.join(dataset_path, 'bank-additional', 'bank-additional-full.csv')
        df = pd.read_csv(file_path, sep=';')
        y_columns = ['y']
        one_hot_encode_df_(df, skip_columns=y_columns)
        label_encode_df_(df, y_columns[0])
        self.x, self.y = split_normalize_sequence(df, y_columns, train, self.type_)


class BlogFeedback(Dataset):
    """
    # Parameters
    root (str): Local path for storing/reading dataset files.
    train (bool): Whether to contain training or test set.
    """
    type_ = REGRESSION

    def __init__(self, root, train=True):
        file_name = 'blogData_train.csv'
        dataset_path = os.path.join(root, self.name)
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00304/BlogFeedback.zip'
        download_unzip(url, dataset_path)

        # Iterate all test csv and concatenate to one DataFrame
        test_dfs = []
        for fn in os.listdir(dataset_path):
            if 'blogData_test' not in fn:
                continue
            file_path = os.path.join(dataset_path, fn)
            test_dfs.append(pd.read_csv(file_path, header=None))
        df_test = pd.concat(test_dfs)

        file_path = os.path.join(dataset_path, file_name)
        df_train = pd.read_csv(file_path, header=None)
        y_columns = [280]
        df_train[y_columns[0]] = np.log(df_train[y_columns[0]] + 0.01)
        df_test[y_columns[0]] = np.log(df_test[y_columns[0]] + 0.01)
        normalize_df_(df_train, df_test=df_test)
        self.x, self.y = xy_split(df_train if train else df_test, y_columns)


class CardDefault(Dataset):
    """
    # Parameters
    root (str): Local path for storing/reading dataset files.
    train (bool): Whether to contain training or test set.
    """
    type_ = CLASSIFICATION

    def __init__(self, root, train=True):
        dataset_path = os.path.join(root, self.name)
        filename = 'data.xls'
        file_path = os.path.join(dataset_path, filename)
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00350/' \
              'default%20of%20credit%20card%20clients.xls'
        download_file(url, dataset_path, filename)
        df = pd.read_excel(file_path, skiprows=1, index_col='ID')
        y_columns = ['default payment next month']
        self.x, self.y = split_normalize_sequence(df, y_columns, train, self.type_)


class CTSlices(Dataset):
    """
    # Parameters
    root (str): Local path for storing/reading dataset files.
    train (bool): Whether to contain training or test set.
    """
    type_ = REGRESSION

    def __init__(self, root, train=True):
        dataset_path = os.path.join(root, self.name)
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00206/slice_localization_data.zip'
        download_unzip(url, dataset_path)
        file_name = 'slice_localization_data.csv'
        file_path = os.path.join(dataset_path, file_name)
        df = pd.read_csv(file_path)
        # No patient should be in both train and test set
        df_train = df[df.patientId < 80]
        df_test = df[df.patientId >= 80]
        df_train = df_train.drop(columns='patientId')
        df_test = df_test.drop(columns='patientId')
        y_columns = ['reference']
        normalize_df_(df_train, df_test=df_test)
        self.x, self.y = xy_split(df_train if train else df_test, y_columns)


class FacebookComments(Dataset):
    """
    # Parameters
    root (str): Local path for storing/reading dataset files.
    train (bool): Whether to contain training or test set.
    """
    type_ = REGRESSION

    def __init__(self, root, train=True):
        dataset_path = os.path.join(root, self.name)
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00363/Dataset.zip'
        download_unzip(url, dataset_path)
        dataset_path = os.path.join(dataset_path, 'Dataset')

        # The 5th variant has the most data
        train_path = os.path.join(dataset_path, 'Training', 'Features_Variant_5.csv')
        test_path = os.path.join(dataset_path, 'Testing', 'Features_TestSet.csv')
        df_train = pd.read_csv(train_path, header=None)
        df_test = pd.read_csv(test_path, header=None)
        y_columns = df_train.columns[-1:]
        normalize_df_(df_train, df_test=df_test)
        self.x, self.y = xy_split(df_train if train else df_test, y_columns)


class Landsat(Dataset):
    """
    # Parameters
    root (str): Local path for storing/reading dataset files.
    train (bool): Whether to contain training or test set.
    """
    type_ = CLASSIFICATION

    def __init__(self, root, train=True):
        dataset_path = os.path.join(root, self.name)
        file_name_train = 'train.csv'
        file_name_test = 'test.csv'
        url_train = 'https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/satimage/sat.trn'
        url_test = 'https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/satimage/sat.tst'
        download_file(url_train, dataset_path, file_name_train)
        download_file(url_test, dataset_path, file_name_test)
        file_path = os.path.join(dataset_path, file_name_train if train else file_name_test)
        df = pd.read_csv(file_path, sep=' ', header=None)
        y_columns = [36]
        label_encode_df_(df, y_columns[0])
        self.x, self.y = split_normalize_sequence(df, y_columns, train, self.type_)


class LetterRecognition(Dataset):
    """
    # Parameters
    root (str): Local path for storing/reading dataset files.
    train (bool): Whether to contain training or test set.
    """
    type_ = CLASSIFICATION

    def __init__(self, root, train=True):
        file_name = 'data.csv'
        dataset_path = os.path.join(root, self.name)
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/letter-recognition.data'
        download_file(url, dataset_path, file_name)
        file_path = os.path.join(dataset_path, file_name)
        df = pd.read_csv(file_path, header=None)
        y_columns = [0]
        label_encode_df_(df, y_columns[0])
        self.x, self.y = split_normalize_sequence(df, y_columns, train, self.type_)


class MagicGamma(Dataset):
    """
    # Parameters
    root (str): Local path for storing/reading dataset files.
    train (bool): Whether to contain training or test set.
    """
    type_ = CLASSIFICATION

    def __init__(self, root, train=True):
        file_name = 'data.csv'
        dataset_path = os.path.join(root, self.name)
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/magic/magic04.data'
        download_file(url, dataset_path, file_name)
        file_path = os.path.join(dataset_path, file_name)
        df = pd.read_csv(file_path, header=None)
        y_columns = [10]
        label_encode_df_(df, y_columns[0])
        self.x, self.y = split_normalize_sequence(df, y_columns, train, self.type_)


class OnlineNews(Dataset):
    """
    # Parameters
    root (str): Local path for storing/reading dataset files.
    train (bool): Whether to contain training or test set.
    """
    type_ = REGRESSION

    def __init__(self, root, train=True):
        dataset_path = os.path.join(root, self.name)
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00332/OnlineNewsPopularity.zip'
        download_unzip(url, dataset_path)
        file_path = os.path.join(dataset_path, 'OnlineNewsPopularity', 'OnlineNewsPopularity.csv')
        df = pd.read_csv(file_path, )
        df.drop(columns=['url', ' timedelta'], inplace=True)
        y_columns = [' shares']
        df[y_columns[0]] = np.log(df[y_columns[0]])
        self.x, self. y = split_normalize_sequence(df, y_columns, train, self.type_)


class Parkinson(Dataset):
    """
    # Parameters
    root (str): Local path for storing/reading dataset files.
    train (bool): Whether to contain training or test set.
    """
    type_ = REGRESSION

    def __init__(self, root, train=True):
        dataset_path = os.path.join(root, self.name)
        filename = 'data.csv'
        file_path: str = os.path.join(dataset_path, filename)
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/' \
              'parkinsons/telemonitoring/parkinsons_updrs.data'
        download_file(url, dataset_path, filename)
        df = pd.read_csv(file_path)
        y_columns = ['motor_UPDRS', 'total_UPDRS']

        df_train = df[df['subject#'] <= 30]
        df_test = df[df['subject#'] > 30]
        df_train = df_train.drop(columns='subject#')
        df_test = df_test.drop(columns='subject#')
        normalize_df_(df_train, df_test=df_test)
        df = df_train if train else df_test
        self.x, self.y = xy_split(df, y_columns)


class PowerPlant(Dataset):
    """
    # Parameters
    root (str): Local path for storing/reading dataset files.
    train (bool): Whether to contain training or test set.
    """
    type_ = REGRESSION

    def __init__(self, root, train=True):
        dataset_path = os.path.join(root, self.name)
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00294/CCPP.zip'
        download_unzip(url, dataset_path)
        file_path = os.path.join(dataset_path, 'CCPP', 'Folds5x2_pp.xlsx')
        df = pd.read_excel(file_path)
        y_columns = ['PE']  # Not clear if this is the aim of the dataset
        self.x, self.y = split_normalize_sequence(df, y_columns, train, self.type_)


class SensorLessDrive(Dataset):
    """
    # Parameters
    root (str): Local path for storing/reading dataset files.
    train (bool): Whether to contain training or test set.
    """
    type_ = CLASSIFICATION

    def __init__(self, root, train=True):
        file_name = 'data.csv'
        dataset_path = os.path.join(root, self.name)
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00325/Sensorless_drive_diagnosis.txt'
        download_file(url, dataset_path, file_name)
        file_path = os.path.join(dataset_path, file_name)
        df = pd.read_csv(file_path, header=None, sep=' ')
        y_columns = [48]
        label_encode_df_(df, y_columns[0])
        self.x, self.y = split_normalize_sequence(df, y_columns, train, self.type_)


class Shuttle(Dataset):
    """
    # Parameters
    root (str): Local path for storing/reading dataset files.
    train (bool): Whether to contain training or test set.
    """
    type_ = CLASSIFICATION

    def __init__(self, root, train=True):
        dataset_path = os.path.join(root, self.name)
        url_train = 'https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/shuttle/shuttle.trn.Z'
        url_test = 'https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/shuttle/shuttle.tst'
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
        df_train = pd.read_csv(file_path_train, header=None, sep=' ')
        df_test = pd.read_csv(file_path_train if train else file_path_test, header=None, sep=' ')
        y_columns = [9]
        label_encode_df_([df_train, df_test], y_columns[0])
        normalize_df_(df_train, df_test=df_test, skip_column=y_columns[0])
        self.x, self.y = xy_split(df_train if train else df_test, y_columns)
        self.y = self.y[:, 0]


class Superconductivity(Dataset):
    """
    # Parameters
    root (str): Local path for storing/reading dataset files.
    train (bool): Whether to contain training or test set.
    """
    type_ = REGRESSION

    def __init__(self, root, train=True):
        dataset_path = os.path.join(root, self.name)
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00464/superconduct.zip'
        download_unzip(url, dataset_path)
        file_path = os.path.join(dataset_path, 'train.csv')
        df = pd.read_csv(file_path)
        y_columns = ['critical_temp']
        self.x, self.y = split_normalize_sequence(df, y_columns, train, self.type_)
