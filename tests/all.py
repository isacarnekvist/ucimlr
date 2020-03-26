import os
import sys
import unittest
import datetime

import numpy as np
import pandas as pd

from ucimlr.helpers import one_hot_encode_df_, clean_na_, label_encode_df_, normalize_df_
from ucimlr import all_datasets, CLASSIFICATION, REGRESSION

dataset_path = 'dataset_test_stubs'


class TestDatasets(unittest.TestCase):

    def test_general(self):
        for dataset_cls in all_datasets:
            for train in [False, True]:
                dataset = dataset_cls(dataset_path, train=train)

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
            end_time = datetime.datetime.now()

    def test_normalized(self):
        for dataset_cls in all_datasets:
            dataset = dataset_cls(dataset_path, train=True)
            self.assertAlmostEqual(dataset.x.mean(), 0, delta=1e-4, msg=f'Dataset name: {dataset.name}')

            # Check standard deviations
            std_is_1 = abs(dataset.x.std(axis=0) - 1) < 1e-2
            std_is_0 = abs(dataset.x.std(axis=0) - 0) < 1e-2
            self.assertTrue((std_is_0 | std_is_1).all(), msg='Standard deviation should be close to either 0 or 1'
                                                             f'Dataset: {dataset.name}')

            if dataset.type_ == REGRESSION:
                self.assertAlmostEqual(dataset.y.mean(), 0, delta=1e-4, msg=f'Dataset name: {dataset.name}')
                std_is_1 = abs(dataset.y.std(axis=0) - 1) < 1e-2
                std_is_0 = abs(dataset.y.std(axis=0) - 0) < 1e-2
                self.assertTrue((std_is_0 | std_is_1).all(), msg='Standard deviation should be close to either 0 or 1')


class TestHelpers(unittest.TestCase):

    def setUp(self) -> None:
        root = os.path.dirname(os.path.realpath(__file__))
        csv_path = os.path.join(root, 'data.csv')
        self.df = pd.read_csv(csv_path, index_col='id')

    def test_one_hot(self):
        clean_na_(self.df)
        one_hot_encode_df_(self.df)
        self.assertTrue((self.df.sex_0 == 1 - self.df.sex_1).all())

    def test_normalize_classification(self):
        clean_na_(self.df)
        label_encode_df_(self.df, 'sex')
        normalize_df_(self.df, skip_column='sex')

        # Label/omitted column should have mean > 0
        self.assertGreater(self.df.sex.mean(), 0)

        # Other columns should have mean = 0
        self.assertAlmostEqual(self.df.age.mean(), 0, delta=1e-6)

        # Columns with original std > 0 should now have std = 1
        self.assertAlmostEqual(self.df.age.std(), 1, delta=1e-6)

    def test_normalize_regression(self):
        clean_na_(self.df)
        one_hot_encode_df_(self.df)
        normalize_df_(self.df)

        # All columns should have mean = 0
        self.assertAlmostEqual(self.df.mean().mean(), 0, delta=1e-9)

        # Columns with original std > 0 should now have std = 1
        self.assertAlmostEqual(self.df.age.std(), 1, delta=1e-1)


if __name__ == '__main__':
    unittest.main()
