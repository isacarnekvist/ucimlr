# UCI Machine Learning Repository - an API

This is a python package to enable easy access to datasets
in the UCI Machine Learning Repository.

The project is at an early alpha stage, so suggestion for
changes or additions are very welcome. Please post an issue
on the [git repository](https://github.com/isacarnekvist/ucimlr).

## Basic usage

```
from ucimlr import datasets
abalone = datasets.Abalone()
print(abalone.type)
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
