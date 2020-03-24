[![Build Status](https://travis-ci.com/isacarnekvist/ucimlr.svg?branch=master)](https://travis-ci.com/isacarnekvist/ucimlr)
# UCI Machine Learning Repository - API 
This package provides easy access to a number of datasets from
the UCI Machine Learning repository.

More documentation will come.

## Citation
Make sure you visit UCI and the specific datasets you use to
make sure that they are cited properly.

## Basic usage
```
import ucimlr

dataset = ucimlr.datasets.Abalone('datasets', download=True)
# Datasets are either 'classification' or 'regression':
print(dataset.type_)
>>> regression

# The attribute 'num_features' show the number of dimensions
# of the independent variable.
print(dataset.num_features)
>>> 10
print(dataset.x.shape)
>>> (3341, 10)

# For datasets of type 'regression', you can inspect the
# dimensionality of the output:
print(dataset.num_targets)
>>> 1
print(dataset.y.shape)
>>> (3341, 1)

# For a classification dataset, the only difference is
# the shape of the dependent variable
dataset = ucimlr.datasets.CardDefault('datasets', download=True)
print(dataset.type_)
>>> classification
# Query number of classes instead of number of features
print(dataset.num_classes)
>>> 2
print(dataset.y.shape)
>>> (24000,)
```
