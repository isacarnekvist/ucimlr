import os
import zipfile
from urllib import request

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


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


def xy_split(df: pd.DataFrame, y_columns: list):
    """
    Takes a DataFrame and return X and Y numpy arrays where
    Y is given by the 'y_columns' argument.
    """
    x = df.drop(columns=y_columns).to_numpy()
    y = df[y_columns].to_numpy()
    return x, y


def label_encode_df_(df, y_column):
    y = df[y_column]
    labels = LabelEncoder().fit_transform(y)
    df[y_column] = labels


def clean_na_(df):
    for column in df.columns:
        df[f'{column}_isnan'] = df[column].isna().astype(float)
    df.fillna(0.0, inplace=True)