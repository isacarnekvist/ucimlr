# Contributing

## Implementing
Make a new dataset like those found in ```ucimlr/dataset.py```.
By subclassing either the ```ucimlr.datasets.ClassificationDataset``` or
```ucimlr.datasets.RegressionDataset```, it will
automatically be added to the list of dataset. Read the superclass
definition to understand what the subclass needs to implement.

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

```pydoc-markdown -m ucimlr.datasets > docs/datasets.md```.
