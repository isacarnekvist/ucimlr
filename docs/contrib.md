# Contributing

## Implementing
Make a new dataset like those found in for example
```ucimlr/regression_datasets.py```.
By subclassing either
```ucimlr.classification_datasets.ClassificationDataset``` or
```ucimlr.regression_datasets.RegressionDataset```, it will
automatically be added to the lists of dataset and be tested.
Read the superclass definition to understand what the subclass
needs to implement.

# Testing
Add a short version of the dataset (~100 rows) to ```dataset_test_stubs```
to allow automatic testing in the cloud without repeated
downloads from UCI's servers. For classification datasets, make
sure all classes are present with multiple examples each as
this will be tested.
Stage and commit this file.

# Documentation generation
First install the document generation tool:

```pip install git+https://github.com/NiklasRosenstein/pydoc-markdown.git@develop```.

Then run 

```sh mkdocs.sh```.

Inspect documentation by running:

```mkdocs serve```
