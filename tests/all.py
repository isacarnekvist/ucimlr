import unittest

import numpy as np
import pandas as pd

from ucimlr.helpers import one_hot_encode_df_


class TestHelpers(unittest.TestCase):

    def test_one_hot(self):
        df = pd.DataFrame([['b'], ['b'], ['a']])
        one_hot_encode_df_(df)
        truth = np.array([[0, 1], [0, 1], [1, 0]])
        self.assertTrue((df.to_numpy() == truth).all())


if __name__ == '__main__':
    unittest.main()
