import os
import unittest

import numpy as np
import pandas as pd

from ucimlr import all_datasets
from ucimlr.constants import TRAIN, VALIDATION, TEST, REGRESSION, CLASSIFICATION
from ucimlr.helpers import one_hot_encode_df_, clean_na_, label_encode_df_, normalize_df_

dataset_path = 'dataset_test_stubs'


class TestDatasets(unittest.TestCase):

    def test_general(self):
        for dataset_cls in all_datasets:
            for split in [TRAIN, VALIDATION, TEST]:
                dataset = dataset_cls(dataset_path, split=split, validation_size=0.2)

                # Tests specific to test split
                if split == TEST:
                    dataset_2 = dataset_cls(dataset_path, split=TEST, validation_size=0.8)
                    self.check_deterministic_test_splits(dataset, dataset_2)

                # Check numbers
                self.assertFalse(np.isnan(dataset.x).any())
                self.assertFalse(np.isnan(dataset.y).any())
                self.assertTrue(np.isfinite(dataset.x).any(), msg=f'Dataset: {dataset.name}')
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
                                     msg=f'dataset.y should have one axis for classification. Dataset: {dataset.name}')

    def check_deterministic_test_splits(self, ds1, ds2):
        """
        We want the test sets to always be the same for comparability, while
        the validation sets and training sets can vary.
        """
        for dataset_cls in all_datasets:
            # We can't compare the numbers since normalization if different.
            # Re-normalizing is not guaranteed to work, but since it is an affine
            # transformation the ordering is preserved.
            for d in range(ds1.num_features):
                # Set to avoid arbitrary ordering of identical elements.
                indices1 = np.array(set(ds1.x[:, d])).argsort()
                indices2 = np.array(set(ds2.x[:, d])).argsort()
                self.assertTrue((indices1 == indices2).all())

    def test_normalized(self):
        for dataset_cls in all_datasets:
            dataset = dataset_cls(dataset_path, split=TRAIN)
            self.assertAlmostEqual(dataset.x.mean(), 0, delta=1e-4, msg=f'Dataset name: {dataset.name}')

            # Check standard deviations
            std_is_1 = abs(dataset.x.std(axis=0) - 1) < 0.05
            std_is_0 = abs(dataset.x.std(axis=0) - 0) < 0.05
            self.assertTrue((std_is_0 | std_is_1).all(), msg='Standard deviation should be close to either 0 or 1 '
                                                             f'Dataset: {dataset.name}')

            if dataset.type_ == REGRESSION:
                self.assertAlmostEqual(dataset.y.mean(), 0, delta=1e-4, msg=f'Dataset name: {dataset.name}')
                std_is_1 = abs(dataset.y.std(axis=0) - 1) < 0.05
                std_is_0 = abs(dataset.y.std(axis=0) - 0) < 0.05
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

    def test_split_df_deterministic(self):
        from ucimlr.helpers import split_df
        df = pd.DataFrame({
            'a': list(range(10))
        })
        df1, _, _ = split_df(df, [0.2, 0.4, 0.4])
        df2, _, _ = split_df(df, [0.2, 0.7, 0.1])
        df3, _, _ = split_df(df, [0.2, 0.1, 0.7])
        self.assertTrue((df1 == df2).all().bool())
        self.assertTrue((df2 == df3).all().bool())

    def test_split_df(self):
        from ucimlr.helpers import split_df
        df = pd.DataFrame({
            'val1': range(10),
        })
        df5, df3, df2 = split_df(df, [0.5, 0.3, 0.2])
        self.assertEqual(len(df5), 5)
        self.assertEqual(len(df3), 3)
        self.assertEqual(len(df2), 2)

        df0, df10 = split_df(df, [0.0, 1.0])
        self.assertEqual(len(df0), 0)
        self.assertEqual(len(df10), 10)

        df10, df0 = split_df(df, [1.0, 0.0])
        self.assertEqual(len(df0), 0)
        self.assertEqual(len(df10), 10)

        def not_summing_to_one():
            _, _ = split_df(df, [0.1, 0.2])
        self.assertRaises(ValueError, not_summing_to_one)

        # Check deterministic
        df5_, df3_, df2_ = split_df(df, [0.5, 0.3, 0.2])
        self.assertTrue((df5_ == df5).all().bool())
        self.assertTrue((df3_ == df3).all().bool())
        self.assertTrue((df2_ == df2).all().bool())

    def test_split_df_on_column(self):
        from ucimlr.helpers import split_df_on_column
        df = pd.DataFrame({
            'person': [0, 0, 1, 1, 2, 2, 3, 3],
        })
        df6, df2 = split_df_on_column(df, [0.75, 0.25], 'person')
        self.assertEqual(len(df6), 6)
        self.assertEqual(len(df2), 2)

        # Check person column is split such that splits are disjoint
        persons2 = set(df2.person.unique())
        persons6 = set(df6.person.unique())
        self.assertTrue(persons2.isdisjoint(persons6))

        # Check  deterministic
        df6_, df2_ = split_df_on_column(df, [0.75, 0.25], 'person')
        self.assertTrue((df2_ == df2).all().bool())


if __name__ == '__main__':
    unittest.main()
