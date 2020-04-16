import os
import sys
import inspect
from copy import deepcopy

import numpy as np
import pandas as pd

from ucimlr.helpers import (download_file, download_unzip, one_hot_encode_df_, xy_split,
                            normalize_df_, split_normalize_sequence, split_df, get_split, split_df_on_column)
from ucimlr.dataset import Dataset
from ucimlr.constants import TRAIN
from ucimlr.constants import REGRESSION


def all_datasets():
    """
    Returns a list of all RegressionDataset classes.
    """
    return [cls for _, cls in inspect.getmembers(sys.modules[__name__])
            if inspect.isclass(cls)
            and issubclass(cls, RegressionDataset)
            and cls != RegressionDataset]


class RegressionDataset(Dataset):
    type_ = REGRESSION  # Is this necessary?

    @property
    def num_targets(self):
        return self.y.shape[1]


class Abalone(RegressionDataset):
    """
    Link to the dataset [description](http://archive.ics.uci.edu/ml/datasets/Abalone).

    # Parameters
    root (str): Local path for storing/reading dataset files.
    split (str): One of {'train', 'validation', 'test'}
    validation_size (float): How large fraction in (0, 1) of the training partition to use for validation.
    """
    def __init__(self, root, split=TRAIN, validation_size=0.2):
        dataset_path = os.path.join(root, self.name)
        filename = 'data.csv'
        file_path = os.path.join(dataset_path, filename)
        url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data'
        download_file(url, dataset_path, filename)
        df = pd.read_csv(file_path, header=None)
        y_columns = df.columns[-1:]
        one_hot_encode_df_(df)
        df_test, df_train, df_valid = split_df(df, [0.2, 0.8 - 0.8 * validation_size, 0.8 * validation_size])
        normalize_df_(df_train, other_dfs=[df_valid, df_test])
        df_res = get_split(df_train, df_valid, df_test, split)
        self.x, self.y = xy_split(df_res, y_columns)


class AirFoil(RegressionDataset):
    """
    Link to the dataset [description](http://archive.ics.uci.edu/ml/datasets/Airfoil+Self-Noise).

    # Parameters
    root (str): Local path for storing/reading dataset files.
    split (str): One of {'train', 'validation', 'test'}
    validation_size (float): How large fraction in (0, 1) of the training partition to use for validation.
    """
    def __init__(self, root, split=TRAIN, validation_size=0.2):
        dataset_path = os.path.join(root, self.name)
        filename = 'airfoil_self_noise.dat'
        url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00291/airfoil_self_noise.dat'
        download_file(url, dataset_path, filename)
        file_path = os.path.join(dataset_path, filename)

        df = pd.read_csv(file_path, sep='\t', header=None)
        y_columns = [5]
        self.x, self.y = split_normalize_sequence(df, y_columns, validation_size, split, self.type_)


class AirQuality(RegressionDataset):
    """
    # Parameters
    root (str): Local path for storing/reading dataset files.
    split (str): One of {'train', 'validation', 'test'}
    validation_size (float): How large fraction in (0, 1) of the training partition to use for validation.
    """
    def __init__(self, root, split=TRAIN, validation_size=0.2):
        dataset_path = os.path.join(root, self.name)
        filename = 'AirQualityUCI.csv'
        url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00360/AirQualityUCI.zip'
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
        self.x, self.y = split_normalize_sequence(df, y_columns, validation_size, split, self.type_)


class BlogFeedback(RegressionDataset):
    """
    Link to the dataset [description](http://archive.ics.uci.edu/ml/datasets/BlogFeedback).

    # Parameters
    root (str): Local path for storing/reading dataset files.
    split (str): One of {'train', 'validation', 'test'}
    validation_size (float): How large fraction in (0, 1) of the training partition to use for validation.
    """
    def __init__(self, root, split=TRAIN, validation_size=0.2):
        file_name = 'blogData_train.csv'
        dataset_path = os.path.join(root, self.name)
        url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00304/BlogFeedback.zip'
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
        df_train_valid = pd.read_csv(file_path, header=None)
        y_columns = [280]
        df_train_valid[y_columns[0]] = np.log(df_train_valid[y_columns[0]] + 0.01)
        df_test[y_columns[0]] = np.log(df_test[y_columns[0]] + 0.01)

        page_columns = list(range(50))
        for i, (_, df_group) in enumerate(df_train_valid.groupby(page_columns)):
            df_train_valid.loc[df_group.index, 'page_id'] = i
        df_train, df_valid = split_df_on_column(df_train_valid, [1 - validation_size, validation_size], 'page_id')
        df_train.drop(columns='page_id', inplace=True)
        df_valid.drop(columns='page_id', inplace=True)

        normalize_df_(df_train, other_dfs=[df_valid, df_test])
        df_res = get_split(df_train, df_valid, df_test, split)
        self.x, self.y = xy_split(df_res, y_columns)


class CTSlices(RegressionDataset):
    """
    # Parameters
    root (str): Local path for storing/reading dataset files.
    split (str): One of {'train', 'validation', 'test'}
    validation_size (float): How large fraction in (0, 1) of the training partition to use for validation.
    """
    def __init__(self, root, split=TRAIN, validation_size=0.2):
        dataset_path = os.path.join(root, self.name)
        url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00206/slice_localization_data.zip'
        download_unzip(url, dataset_path)
        file_name = 'slice_localization_data.csv'
        file_path = os.path.join(dataset_path, file_name)
        df = pd.read_csv(file_path)
        # No patient should be in both train and test set
        df_train_valid = deepcopy(df.loc[df.patientId < 80, :])  # Pandas complains if it is a view
        df_test = deepcopy(df.loc[df.patientId >= 80, :])        # - " -
        df_train, df_valid = split_df_on_column(df_train_valid, [1 - validation_size, validation_size], 'patientId')
        y_columns = ['reference']
        normalize_df_(df_train, other_dfs=[df_valid, df_test])
        df_res = get_split(df_train, df_valid, df_test, split)
        df_res = df_res.drop(columns='patientId')
        self.x, self.y = xy_split(df_res, y_columns)


class FacebookComments(RegressionDataset):
    """
    Predict the number of likes on posts from a collection of Facebook pages.
    Every page has multiple posts, making the number of pages less than the samples
    in the dataset (each sample is one post).

    # Note
    The provided test split has a relatively large discrepancy in terms
    of distributions of the features and targets. Training and validation splits are
    also made to ensure that the same page is not in both splits. This makes the distributions
    of features in training and validation splits vary to a relatively large extent, possible
    because the number of pages are not that many, while the features are many.

    Link to the dataset [description](http://archive.ics.uci.edu/ml/datasets/Facebook+Comment+Volume+Dataset).

    # Parameters
    root (str): Local path for storing/reading dataset files.
    split (str): One of {'train', 'validation', 'test'}
    validation_size (float): How large fraction in (0, 1) of the training partition to use for validation.
    """
    def __init__(self, root, split=TRAIN, validation_size=0.2):
        dataset_path = os.path.join(root, self.name)
        url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00363/Dataset.zip'
        download_unzip(url, dataset_path)
        dataset_path = os.path.join(dataset_path, 'Dataset')

        # The 5th variant has the most data
        train_path = os.path.join(dataset_path, 'Training', 'Features_Variant_5.csv')
        test_path = os.path.join(dataset_path, 'Testing', 'Features_TestSet.csv')
        df_train_valid = pd.read_csv(train_path, header=None)
        df_test = pd.read_csv(test_path, header=None)
        y_columns = df_train_valid.columns[-1:]

        # Page ID is not included, but can be derived. Page IDs can not be
        # in both training and validation sets
        page_columns = list(range(29))
        for i, (_, df_group) in enumerate(df_train_valid.groupby(page_columns)):
            df_train_valid.loc[df_group.index, 'page_id'] = i
        df_train, df_valid = split_df_on_column(df_train_valid, [1 - validation_size, validation_size], 'page_id')
        df_train.drop(columns='page_id', inplace=True)
        df_valid.drop(columns='page_id', inplace=True)

        normalize_df_(df_train, other_dfs=[df_valid, df_test])
        df_res = get_split(df_train, df_valid, df_test, split)
        self.x, self.y = xy_split(df_res, y_columns)


class OnlineNews(RegressionDataset):
    """
    # Parameters
    root (str): Local path for storing/reading dataset files.
    split (str): One of {'train', 'validation', 'test'}
    validation_size (float): How large fraction in (0, 1) of the training partition to use for validation.
    """
    def __init__(self, root, split=TRAIN, validation_size=0.2):
        dataset_path = os.path.join(root, self.name)
        url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00332/OnlineNewsPopularity.zip'
        download_unzip(url, dataset_path)
        file_path = os.path.join(dataset_path, 'OnlineNewsPopularity', 'OnlineNewsPopularity.csv')
        df = pd.read_csv(file_path, )
        df.drop(columns=['url', ' timedelta'], inplace=True)
        y_columns = [' shares']
        df[y_columns[0]] = np.log(df[y_columns[0]])
        self.x, self. y = split_normalize_sequence(df, y_columns, validation_size, split, self.type_)


class Parkinson(RegressionDataset):
    """
    Link to the dataset [description](http://archive.ics.uci.edu/ml/datasets/parkinsons).

    # Parameters
    root (str): Local path for storing/reading dataset files.
    split (str): One of {'train', 'validation', 'test'}
    validation_size (float): How large fraction in (0, 1) of the training partition to use for validation.
    """
    def __init__(self, root, split=TRAIN, validation_size=0.2):
        dataset_path = os.path.join(root, self.name)
        filename = 'data.csv'
        file_path: str = os.path.join(dataset_path, filename)
        url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/' \
              'parkinsons/telemonitoring/parkinsons_updrs.data'
        download_file(url, dataset_path, filename)
        df = pd.read_csv(file_path)
        y_columns = ['motor_UPDRS', 'total_UPDRS']

        df_train_valid = df[df['subject#'] <= 30]
        df_test = deepcopy(df[df['subject#'] > 30])

        df_train, df_valid = split_df_on_column(df_train_valid, [1 - validation_size, validation_size], 'subject#')
        normalize_df_(df_train, other_dfs=[df_valid, df_test])
        df_res = get_split(df_train, df_valid, df_test, split)
        df_res.drop(columns='subject#', inplace=True)
        self.x, self.y = xy_split(df_res, y_columns)


class PowerPlant(RegressionDataset):
    """
    # Parameters
    root (str): Local path for storing/reading dataset files.
    split (str): One of {'train', 'validation', 'test'}
    validation_size (float): How large fraction in (0, 1) of the training partition to use for validation.
    """
    def __init__(self, root, split=TRAIN, validation_size=0.2):
        dataset_path = os.path.join(root, self.name)
        url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00294/CCPP.zip'
        download_unzip(url, dataset_path)
        file_path = os.path.join(dataset_path, 'CCPP', 'Folds5x2_pp.xlsx')
        df = pd.read_excel(file_path)
        y_columns = ['PE']  # Not clear if this is the aim of the dataset
        self.x, self.y = split_normalize_sequence(df, y_columns, validation_size, split, self.type_)


class RealEstate(RegressionDataset):
    """
    Link to the dataset [description](http://archive.ics.uci.edu/ml/datasets/Real+estate+valuation+data+set).

    # Parameters
    root (str): Local path for storing/reading dataset files.
    split (str): One of {'train', 'validation', 'test'}
    validation_size (float): How large fraction in (0, 1) of the training partition to use for validation.
    """
    def __init__(self, root, split=TRAIN, validation_size=0.2):
        dataset_path = os.path.join(root, self.name)
        filename = 'Real estate valuation data set.xlsx'
        url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00477/Real%20estate%20valuation%20data%20set.xlsx'
        download_file(url, dataset_path, filename)
        file_path = os.path.join(dataset_path, filename)

        df = pd.read_excel(file_path, index_col='No')
        y_columns = ['Y house price of unit area']
        self.x, self.y = split_normalize_sequence(df, y_columns, validation_size, split, self.type_)


class Superconductivity(RegressionDataset):
    """
    # Parameters
    root (str): Local path for storing/reading dataset files.
    split (str): One of {'train', 'validation', 'test'}
    validation_size (float): How large fraction in (0, 1) of the training partition to use for validation.
    """
    def __init__(self, root, split=TRAIN, validation_size=0.2):
        dataset_path = os.path.join(root, self.name)
        url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00464/superconduct.zip'
        download_unzip(url, dataset_path)
        file_path = os.path.join(dataset_path, 'train.csv')
        df = pd.read_csv(file_path)
        y_columns = ['critical_temp']
        self.x, self.y = split_normalize_sequence(df, y_columns, validation_size, split, self.type_)


class WhiteWineQuality(RegressionDataset):
    """
    Description of dataset [here](http://archive.ics.uci.edu/ml/datasets/Wine+Quality).

    Citation:
    ```
    P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.
    Modeling wine preferences by data mining from physicochemical properties.
    In Decision Support Systems, Elsevier, 47(4):547-553, 2009.
    ```

    # Parameters
    root (str): Local path for storing/reading dataset files.
    split (str): One of {'train', 'validation', 'test'}
    validation_size (float): How large fraction in (0, 1) of the training partition to use for validation.
    """
    def __init__(self, root, split=TRAIN, validation_size=0.2):
        dataset_path = os.path.join(root, self.name)
        filename = 'data.csv'
        file_path = os.path.join(dataset_path, filename)
        url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv'
        download_file(url, dataset_path, filename)
        df = pd.read_csv(file_path, sep=';')
        y_columns = ['quality']
        self.x, self.y = split_normalize_sequence(df, y_columns, validation_size, split, self.type_)
