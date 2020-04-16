def all_datasets():
    from . import regression_datasets, classification_datasets
    return regression_datasets.all_datasets() + classification_datasets.all_datasets()
