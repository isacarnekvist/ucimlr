import os
import zipfile
from copy import deepcopy
from urllib import request

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from ucimlr.constants import CLASSIFICATION
from ucimlr.constants import TRAIN, VALIDATION, TEST


class RandomStateContext:
    def __init__(self, seed=0):
        self.original_state = 0
        self.seed = seed

    def __enter__(self):
        self.original_state = np.random.get_state()
        np.random.seed(self.seed)

    def __exit__(self, type=None, value=None, traceback=None):
        np.random.set_state(self.original_state)


def download_file(url, dataset_path, filename):
    """
    Downloads the file at the url to the path 'dataset_path/filename'.

    Returns True if dataset was downloaded, False if already existing
    """
    os.makedirs(dataset_path, exist_ok=True)
    file_path = os.path.join(dataset_path, filename)
    if not os.path.exists(file_path):
        request.urlretrieve(url, file_path)
        return True
    else:
        return False


def download_unzip(url: str, dataset_path: str):
    """
    Downloads the file at the specified 'url' and unzips it
    at the specified 'dataset_path'.
    """
    filename_zip = 'data.zip'
    file_path_zip = os.path.join(dataset_path, filename_zip)
    fresh_download = download_file(url, dataset_path, filename_zip)
    if fresh_download:
        # TODO This naively assumes downloaded implies extracted
        zf = zipfile.ZipFile(file_path_zip)
        zf.extractall(dataset_path)


# Pandas DataFrame operations

def one_hot_encode_df_(df, skip_columns=None):
    """
    One-hot encodes all non-numeric columns and drops those columns.
    This is done in place.
    """
    if skip_columns is None:
        skip_columns = set()
    else:
        skip_columns = set(skip_columns)

    for col in set(df.columns) - skip_columns:
        if col in skip_columns:
            continue
        dtype = df[col].dtype
        if not np.issubdtype(dtype, np.number):
            x = df[col].to_numpy().reshape(-1, 1)
            one_hot = OneHotEncoder(sparse=False).fit_transform(x)
            for col_i in range(one_hot.shape[1]):
                df[f'{col}_{col_i}'] = one_hot[:, col_i]
            df.drop(columns=col, inplace=True)


def normalize_df_(df_train, *, other_dfs=None, skip_column=None):
    """
    Normalizes all columns of `df_train` to zero mean unit variance in place.
    Optionally performs same transformation to `other_dfs`

    # Parameters
    other_dfs [pd.DataFrame]: List of other DataFrames to apply transformation to
    skip_column (str, int): Column to omit, for example categorical targets.
    """
    if skip_column is None:
        skip_columns = set()
    else:
        skip_columns = {skip_column}

    # Skip where standard deviation is zero or close to zero
    low_std_columns = df_train.columns[df_train.std() < 1e-6]
    df_train.loc[:, low_std_columns] = 0
    if other_dfs is not None:
        for df in other_dfs:
            df.loc[:, low_std_columns] = 0
    skip_columns.update(set(low_std_columns))

    columns = list(set(df_train.columns) - skip_columns)
    mean = df_train[columns].mean(axis=0)
    std = df_train[columns].std(axis=0)
    df_train.loc[:, columns] -= mean
    df_train.loc[:, columns] /= std
    if other_dfs is not None:
        for df in other_dfs:
            df.loc[:, columns] -= mean
            df.loc[:, columns] /= std


def xy_split(df: pd.DataFrame, y_columns: list):
    """
    Takes a DataFrame and return X and Y numpy arrays where
    Y is given by the 'y_columns' argument.
    """
    x = df.drop(columns=y_columns).to_numpy()
    y = df[y_columns].to_numpy()
    return x, y


def label_encode_df_(dfs, y_column):
    """
    Label encodes in place.

    # Parameters
    dfs:  DataFrame or list of DataFrames
    y_column: The label column
    """
    if type(dfs) is not list:
        dfs = [dfs]
    ys = set()
    for df in dfs:
        ys.update(set(df[y_column]))
    le = LabelEncoder().fit(list(ys))
    for df in dfs:
        df[y_column] = le.transform(df[y_column])


def clean_na_(df):
    for column in df.columns:
        df[f'{column}_isnan'] = df[column].isna().astype(float)
    df.fillna(0.0, inplace=True)


def split_normalize_sequence(df: pd.DataFrame, y_columns, validation_size: float, split, dataset_type):
    """
    Performs the common sequence of operations:
    train_test_split -> normalize -> x, y split
    """
    if dataset_type == CLASSIFICATION:
        df_test, df_train, df_valid = split_classification_df(df,
                                                              [0.2, 0.8 - 0.8 * validation_size, 0.8 * validation_size],
                                                              y_columns[0])
    else:
        df_test, df_train, df_valid = split_df(df, [0.2, 0.8 - 0.8 * validation_size, 0.8 * validation_size])
    if dataset_type == CLASSIFICATION:
        normalize_df_(df_train, other_dfs=[df_valid, df_test], skip_column=y_columns[0])
    else:
        normalize_df_(df_train, other_dfs=[df_valid, df_test])
    df_res = get_split(df_train, df_valid, df_test, split)
    x, y = xy_split(df_res, y_columns)
    if dataset_type == CLASSIFICATION:
        y = y[:, 0]
    return x, y


def split_df(df: pd.DataFrame, fractions):
    """
    Randomly splits (always with same seed) the dataframe `df` into
    `fractions`.

    If you want one split to be always the same (given the same fraction of course),
    let that split be the first one. For example:
    ```
    df_test, df_train, df_valid = split_df(df, [0.2, 0.4, 0.4])
    ```
    or
    ```
    df_test, df_train, df_valid = split_df(df, [0.2, 0.7, 0.1])
    ```
    These always produce the same df_test.
    """
    fractions = np.array(fractions)
    if abs(1 - sum(fractions)) > 1e-6:
        raise ValueError('Fractions must sum to one.')
    for i in range(len(fractions)):
        fraction = fractions[0]
        sub_df = df.sample(frac=fraction, random_state=0)
        yield sub_df
        fractions = fractions[1:]
        if sum(fractions) > 0:
            fractions /= sum(fractions)
        df = df.drop(sub_df.index)


def split_classification_df(df, fractions, y_column):
    """
    Like split_df but groups into respective classes and then samples
    fractions of those groups. Makes sure that:
    1) No labels/classes are missing from any of the splits
    2) The proportions of classes are the same in all splits
    """
    fractions = np.array(fractions)
    splits = [pd.DataFrame(columns=df.columns)] * len(fractions)
    for i in range(len(fractions)):
        fraction = fractions[0]
        for value, group in df.groupby(y_column):
            sample = group.sample(frac=fraction, random_state=0)
            splits[i] = splits[i].append(sample)
            df = df.drop(sample.index)
        fractions = fractions[1:]
        if fractions.sum() > 0:
            fractions /= fractions.sum()
    # np.isnan crashes if this cast is not done
    for i, split in enumerate(splits):
        splits[i] = split.astype(np.float32)
        splits[i].loc[:, y_column] = split.loc[:, y_column].astype(np.int)
    return splits


def split_df_on_column(df, fractions, column):
    """
    Randomly splits (always with same seed) the dataframe `df` into
    `fractions`. Values in the `column` are disjoint over the splits.
    """
    fractions = np.array(fractions)
    if sum(fractions) != 1.0:
        raise ValueError('Fractions must sum to one.')
    with RandomStateContext():
        groups = np.random.permutation(df[column].unique())
    for i in range(len(fractions)):
        fraction = fractions[0]
        groups_sample = groups[:int(fraction * len(groups))]
        df_sample = deepcopy(df.loc[df[column].isin(groups_sample), :])
        yield df_sample
        fractions = fractions[1:]
        if sum(fractions) > 0:
            fractions /= sum(fractions)
        df = df.drop(df_sample.index)


def get_split(df_train, df_valid, df_test, split):
    if split == TRAIN:
        return df_train
    elif split == VALIDATION:
        return df_valid
    elif split == TEST:
        return df_test
    else:
        raise ValueError('split is not correct, see ucimlr.constants')
