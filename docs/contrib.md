# Contributing

## Implementing
Make a new dataset like those found in ```ucimlr/dataset.py```.
By subclassing the ```ucimlr.dataset.Dataset```, it will
automatically be added to the list of dataset.

# Testing
Add a short version of the dataset (~100 rows) to ```dataset_test_stubs```
to allow automatic testing in the cloud without repeated
downloads from UCI's servers. Stage and commit this file.

# Documentation generation
First install the document generation tool:

```pip install git+https://github.com/NiklasRosenstein/pydoc-markdown.git@develop```.

Then run 

```pydoc-markdown -m ucimlr.datasets > docs/datasets.md```.
