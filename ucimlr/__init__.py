import inspect

from . import datasets

__all = [cls for _, cls in inspect.getmembers(datasets)
         if inspect.isclass(cls)
         and issubclass(cls, datasets.Dataset)
         and cls != datasets.Dataset]


def regression_datasets():
    return [ds for ds in __all if ds.type_ == datasets.REGRESSION]


def classification_datasets():
    return [ds for ds in __all if ds.type_ == datasets.CLASSIFICATION]
