import inspect

from . import datasets
from .datasets import CLASSIFICATION, REGRESSION

all_datasets = [cls for _, cls in inspect.getmembers(datasets)
                if inspect.isclass(cls)
                and issubclass(cls, datasets.Dataset)
                and cls != datasets.Dataset
                and cls != datasets.ClassificationDataset
                and cls != datasets.RegressionDataset]


def regression_datasets():
    return [ds for ds in all_datasets if ds.type_ == REGRESSION]


def classification_datasets():
    return [ds for ds in all_datasets if ds.type_ == CLASSIFICATION]
