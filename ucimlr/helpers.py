import os
import zipfile
from urllib import request

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# Cross imports, extract these to types, or preferably
# different classes for regr. and classif.
CLASSIFICATION = 'classification'


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


def normalize_df_(df_train, *, df_test=None, skip_column=None):
    """
    Normalizes all columns of `df_train` to zero mean unit variance in place.
    Optionally performs same transformation to `df_test`

    # Parameters
    skip_column (str, int): Column to omit, for example categorical targets.
    """
    if skip_column is None:
        skip_columns = set()
    else:
        skip_columns = {skip_column}

    # Skip where standard deviation is zero or close to zero
    low_std_columns = df_train.columns[df_train.std() < 1e-6]
    df_train.loc[:, low_std_columns] = 0
    if df_test is not None:
        df_test.loc[:, low_std_columns] = 0
    skip_columns.update(set(low_std_columns))

    columns = list(set(df_train.columns) - skip_columns)
    mean = df_train[columns].mean(axis=0)
    std = df_train[columns].std(axis=0)
    df_train.loc[:, columns] -= mean
    df_train.loc[:, columns] /= std
    if df_test is not None:
        df_test.loc[:, columns] -= mean
        df_test.loc[:, columns] /= std


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


def train_test_split(df, fraction=0.8, random_state=0):
    df_train = df.sample(frac=fraction, random_state=random_state)
    df_test = df.drop(df_train.index)
    return df_train, df_test


def split_normalize_sequence(df, y_columns, train, dataset_type):
    """
    Performs the common sequence of operations:
    train_test_split -> normalize -> x, y split
    """
    df_train, df_test = train_test_split(df)
    if dataset_type == CLASSIFICATION:
        normalize_df_(df_train, df_test=df_test, skip_column=y_columns[0])
    else:
        normalize_df_(df_train, df_test=df_test)
    if train:
        x, y = xy_split(df_train, y_columns)
    else:
        x, y = xy_split(df_test, y_columns)
    if dataset_type == CLASSIFICATION:
        y = y[:, 0]
    return x, y
