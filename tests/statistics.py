from argparse import ArgumentParser

import numpy as np

from ucimlr import all_datasets
from ucimlr.constants import TRAIN, VALIDATION, TEST


def wasserstein(dataset_path):
    """
    This needs to be the full datasets to make sense.
    Since it is the full datasets, this is not included
    when running all tests.
    """
    from scipy.stats import wasserstein_distance
    for dataset_cls in all_datasets:
        print('Loading dataset:', dataset_cls.__name__)
        train = dataset_cls(dataset_path, split=TRAIN)
        valid = dataset_cls(dataset_path, split=VALIDATION)
        test = dataset_cls(dataset_path, split=TEST)
        splits = [train, valid, test]

        ws_max = 0.0
        ws_ave = 0.0
        for i in range(train.num_features):
            for j, split in enumerate(splits):
                if j == 0:
                    hist, bins = np.histogram(split.x[:, i], density=True)
                    us = split.x[:, i]
                else:
                    hist, _ = np.histogram(split.x[:, i], density=True, bins=bins)
                    vs = split.x[:, i]
                    ws = wasserstein_distance(us, vs)
                    ws_max = max(ws_max, wasserstein_distance(us, vs))
                    ws_ave += ws / (train.num_features * 2)
        print('Max wasserstein:', ws_max)
        print('Average wasserstein:', ws_ave)
        print()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('datasets_dir')
    args = parser.parse_args()
    wasserstein(args.datasets_dir)

