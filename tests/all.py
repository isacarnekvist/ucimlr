import os
import unittest

import numpy as np
import pandas as pd

from ucimlr.helpers import one_hot_encode_df_, clean_na_
from ucimlr import all_datasets, CLASSIFICATION, REGRESSION


class TestDatasets(unittest.TestCase):

    def test_all(self):
        for dataset_cls in all_datasets:
            dataset = dataset_cls('datasets')

            # Check numbers
            self.assertFalse(np.isnan(dataset.x).any())
            self.assertFalse(np.isnan(dataset.y).any())
            self.assertTrue(np.isfinite(dataset.x).any())
            self.assertTrue(np.isfinite(dataset.y).any())

            # Check shapes and number of classes
            self.assertTrue(dataset.x.shape[1], dataset.num_features)
            self.assertEqual(dataset.num_features, dataset.x.shape[1])
            if dataset.type_ is REGRESSION:
                self.assertEqual(len(dataset.y.shape),  2,
                                 msg='dataset.y should have two axis for regression')
                self.assertEqual(dataset.num_targets, dataset.y.shape[1])
            elif dataset.type_ is CLASSIFICATION:
                self.assertEqual(len(dataset.y.shape),  1,
                                 msg='dataset.y should have one axis for classification')
                classes = set(dataset.y)
                expected_classes = {c for c in range(dataset.num_classes)}
                self.assertEqual(classes.symmetric_difference(expected_classes), set(),
                                 msg='Class labels should be {0, ..., dataset.num_classes - 1}')


class TestHelpers(unittest.TestCase):

    def setUp(self) -> None:
        root = os.path.dirname(os.path.realpath(__file__))
        csv_path = os.path.join(root, 'data.csv')
        self.df_regression = pd.read_csv(csv_path, index_col='id')

    def test_one_hot(self):
        clean_na_(self.df_regression)
        one_hot_encode_df_(self.df_regression)
        self.assertTrue((self.df_regression.sex_0 == 1 - self.df_regression.sex_1).all())


if __name__ == '__main__':
    unittest.main()
