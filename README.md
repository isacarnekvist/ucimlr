[![Build Status](https://travis-ci.com/isacarnekvist/ucimlr.svg?branch=master)](https://travis-ci.com/isacarnekvist/ucimlr)
[![Documentation Status](https://readthedocs.org/projects/ucimlr/badge/?version=latest)](https://ucimlr.readthedocs.io/en/latest/?badge=latest)

# An API to the UCI Machine Learning Repository

This is a python package to enable easy access to datasets
in the UCI Machine Learning Repository. Note that this is not
an official API. Any usage of datasets should be cited
according to instructions in the UCI Machine Learning
Repository.

The project is at an early alpha stage, so suggestion for
changes or additions are very welcome.

Link to [documentation](https://ucimlr.readthedocs.io/).

## Basic usage

```
from ucimlr import regression_datasets
abalone = regression_datasets.Abalone('dataset_folder')
print(abalone.type_)
>>> regression
print(len(abalone))
>>> 3341
```

Independent and dependent variables are accessed as
numpy arrays:
```
print(abalone.x.shape)
>>> (3341, 10)
print(abalone.y.shape)
>>> (3341, 1)
```

Or by element access:
```
x, y = dataset[0]
```
