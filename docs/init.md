<a name=".ucimlr.__init__"></a>
## ucimlr.\_\_init\_\_

<a name=".ucimlr.__init__.constants"></a>
## ucimlr.\_\_init\_\_.constants

Dataset types

<a name=".ucimlr.__init__.regression_datasets"></a>
## ucimlr.\_\_init\_\_.regression\_datasets

<a name=".ucimlr.__init__.regression_datasets.Abalone"></a>
### Abalone

```python
class Abalone(RegressionDataset):
 |  Abalone(root, split=TRAIN, validation_size=0.2)
```

Link to the dataset [description](http://archive.ics.uci.edu/ml/datasets/Abalone).

__Parameters__

- __root__ (`str`): Local path for storing/reading dataset files.
- __split__ (`str`): One of {'train', 'validation', 'test'}
- __validation_size__ (`float`): How large fraction in (0, 1) of the training partition to use for validation.

<a name=".ucimlr.__init__.regression_datasets.AirFoil"></a>
### AirFoil

```python
class AirFoil(RegressionDataset):
 |  AirFoil(root, split=TRAIN, validation_size=0.2)
```

Link to the dataset [description](http://archive.ics.uci.edu/ml/datasets/Airfoil+Self-Noise).

__Parameters__

- __root__ (`str`): Local path for storing/reading dataset files.
- __split__ (`str`): One of {'train', 'validation', 'test'}
- __validation_size__ (`float`): How large fraction in (0, 1) of the training partition to use for validation.

<a name=".ucimlr.__init__.regression_datasets.AirQuality"></a>
### AirQuality

```python
class AirQuality(RegressionDataset):
 |  AirQuality(root, split=TRAIN, validation_size=0.2)
```

__Parameters__

- __root__ (`str`): Local path for storing/reading dataset files.
- __split__ (`str`): One of {'train', 'validation', 'test'}
- __validation_size__ (`float`): How large fraction in (0, 1) of the training partition to use for validation.

<a name=".ucimlr.__init__.regression_datasets.BlogFeedback"></a>
### BlogFeedback

```python
class BlogFeedback(RegressionDataset):
 |  BlogFeedback(root, split=TRAIN, validation_size=0.2)
```

Link to the dataset [description](http://archive.ics.uci.edu/ml/datasets/BlogFeedback).

__Parameters__

- __root__ (`str`): Local path for storing/reading dataset files.
- __split__ (`str`): One of {'train', 'validation', 'test'}
- __validation_size__ (`float`): How large fraction in (0, 1) of the training partition to use for validation.

<a name=".ucimlr.__init__.regression_datasets.CTSlices"></a>
### CTSlices

```python
class CTSlices(RegressionDataset):
 |  CTSlices(root, split=TRAIN, validation_size=0.2)
```

__Parameters__

- __root__ (`str`): Local path for storing/reading dataset files.
- __split__ (`str`): One of {'train', 'validation', 'test'}
- __validation_size__ (`float`): How large fraction in (0, 1) of the training partition to use for validation.

<a name=".ucimlr.__init__.regression_datasets.FacebookComments"></a>
### FacebookComments

```python
class FacebookComments(RegressionDataset):
 |  FacebookComments(root, split=TRAIN, validation_size=0.2)
```

Predict the number of likes on posts from a collection of Facebook pages.
Every page has multiple posts, making the number of pages less than the samples
in the dataset (each sample is one post).

__Note__

The provided test split has a relatively large discrepancy in terms
of distributions of the features and targets. Training and validation splits are
also made to ensure that the same page is not in both splits. This makes the distributions
of features in training and validation splits vary to a relatively large extent, possible
because the number of pages are not that many, while the features are many.

Link to the dataset [description](http://archive.ics.uci.edu/ml/datasets/Facebook+Comment+Volume+Dataset).

__Parameters__

- __root__ (`str`): Local path for storing/reading dataset files.
- __split__ (`str`): One of {'train', 'validation', 'test'}
- __validation_size__ (`float`): How large fraction in (0, 1) of the training partition to use for validation.

<a name=".ucimlr.__init__.regression_datasets.OnlineNews"></a>
### OnlineNews

```python
class OnlineNews(RegressionDataset):
 |  OnlineNews(root, split=TRAIN, validation_size=0.2)
```

__Parameters__

- __root__ (`str`): Local path for storing/reading dataset files.
- __split__ (`str`): One of {'train', 'validation', 'test'}
- __validation_size__ (`float`): How large fraction in (0, 1) of the training partition to use for validation.

<a name=".ucimlr.__init__.regression_datasets.Parkinson"></a>
### Parkinson

```python
class Parkinson(RegressionDataset):
 |  Parkinson(root, split=TRAIN, validation_size=0.2)
```

Link to the dataset [description](http://archive.ics.uci.edu/ml/datasets/parkinsons).

__Parameters__

- __root__ (`str`): Local path for storing/reading dataset files.
- __split__ (`str`): One of {'train', 'validation', 'test'}
- __validation_size__ (`float`): How large fraction in (0, 1) of the training partition to use for validation.

<a name=".ucimlr.__init__.regression_datasets.PowerPlant"></a>
### PowerPlant

```python
class PowerPlant(RegressionDataset):
 |  PowerPlant(root, split=TRAIN, validation_size=0.2)
```

__Parameters__

- __root__ (`str`): Local path for storing/reading dataset files.
- __split__ (`str`): One of {'train', 'validation', 'test'}
- __validation_size__ (`float`): How large fraction in (0, 1) of the training partition to use for validation.

<a name=".ucimlr.__init__.regression_datasets.RealEstate"></a>
### RealEstate

```python
class RealEstate(RegressionDataset):
 |  RealEstate(root, split=TRAIN, validation_size=0.2)
```

Link to the dataset [description](http://archive.ics.uci.edu/ml/datasets/Real+estate+valuation+data+set).

__Parameters__

- __root__ (`str`): Local path for storing/reading dataset files.
- __split__ (`str`): One of {'train', 'validation', 'test'}
- __validation_size__ (`float`): How large fraction in (0, 1) of the training partition to use for validation.

<a name=".ucimlr.__init__.regression_datasets.Superconductivity"></a>
### Superconductivity

```python
class Superconductivity(RegressionDataset):
 |  Superconductivity(root, split=TRAIN, validation_size=0.2)
```

__Parameters__

- __root__ (`str`): Local path for storing/reading dataset files.
- __split__ (`str`): One of {'train', 'validation', 'test'}
- __validation_size__ (`float`): How large fraction in (0, 1) of the training partition to use for validation.

<a name=".ucimlr.__init__.regression_datasets.WhiteWineQuality"></a>
### WhiteWineQuality

```python
class WhiteWineQuality(RegressionDataset):
 |  WhiteWineQuality(root, split=TRAIN, validation_size=0.2)
```

Description of dataset [here](http://archive.ics.uci.edu/ml/datasets/Wine+Quality).

Citation:
```
P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.
Modeling wine preferences by data mining from physicochemical properties.
In Decision Support Systems, Elsevier, 47(4):547-553, 2009.
```

__Parameters__

- __root__ (`str`): Local path for storing/reading dataset files.
- __split__ (`str`): One of {'train', 'validation', 'test'}
- __validation_size__ (`float`): How large fraction in (0, 1) of the training partition to use for validation.

<a name=".ucimlr.__init__.dataset"></a>
## ucimlr.\_\_init\_\_.dataset

<a name=".ucimlr.__init__.classification_datasets"></a>
## ucimlr.\_\_init\_\_.classification\_datasets

<a name=".ucimlr.__init__.classification_datasets.Adult"></a>
### Adult

```python
class Adult(ClassificationDataset):
 |  Adult(root, split=TRAIN, validation_size=0.2)
```

__Parameters__

- __root__ (`str`): Local path for storing/reading dataset files.
- __split__ (`str`): One of {'train', 'validation', 'test'}
- __validation_size__ (`float`): How large fraction in (0, 1) of the training partition to use for validation.

<a name=".ucimlr.__init__.classification_datasets.APSFailure"></a>
### APSFailure

```python
class APSFailure(ClassificationDataset):
 |  APSFailure(root, split=TRAIN, validation_size=0.2)
```

__Parameters__

- __root__ (`str`): Local path for storing/reading dataset files.
- __split__ (`str`): One of {'train', 'validation', 'test'}
- __validation_size__ (`float`): How large fraction in (0, 1) of the training partition to use for validation.

<a name=".ucimlr.__init__.classification_datasets.Avila"></a>
### Avila

```python
class Avila(ClassificationDataset):
 |  Avila(root, split=TRAIN, validation_size=0.2)
```

__Parameters__

- __root__ (`str`): Local path for storing/reading dataset files.
- __split__ (`str`): One of {'train', 'validation', 'test'}
- __validation_size__ (`float`): How large fraction in (0, 1) of the training partition to use for validation.

<a name=".ucimlr.__init__.classification_datasets.BankMarketing"></a>
### BankMarketing

```python
class BankMarketing(ClassificationDataset):
 |  BankMarketing(root, split=TRAIN, validation_size=0.2)
```

__Parameters__

- __root__ (`str`): Local path for storing/reading dataset files.
- __split__ (`str`): One of {'train', 'validation', 'test'}
- __validation_size__ (`float`): How large fraction in (0, 1) of the training partition to use for validation.

<a name=".ucimlr.__init__.classification_datasets.CardDefault"></a>
### CardDefault

```python
class CardDefault(ClassificationDataset):
 |  CardDefault(root, split=TRAIN, validation_size=0.2)
```

__Parameters__

- __root__ (`str`): Local path for storing/reading dataset files.
- __split__ (`str`): One of {'train', 'validation', 'test'}
- __validation_size__ (`float`): How large fraction in (0, 1) of the training partition to use for validation.

<a name=".ucimlr.__init__.classification_datasets.Landsat"></a>
### Landsat

```python
class Landsat(ClassificationDataset):
 |  Landsat(root, split=TRAIN, validation_size=0.2)
```

__Parameters__

- __root__ (`str`): Local path for storing/reading dataset files.
- __split__ (`str`): One of {'train', 'validation', 'test'}
- __validation_size__ (`float`): How large fraction in (0, 1) of the training partition to use for validation.

<a name=".ucimlr.__init__.classification_datasets.LetterRecognition"></a>
### LetterRecognition

```python
class LetterRecognition(ClassificationDataset):
 |  LetterRecognition(root, split=TRAIN, validation_size=0.2)
```

__Parameters__

- __root__ (`str`): Local path for storing/reading dataset files.
- __split__ (`str`): One of {'train', 'validation', 'test'}
- __validation_size__ (`float`): How large fraction in (0, 1) of the training partition to use for validation.

<a name=".ucimlr.__init__.classification_datasets.MagicGamma"></a>
### MagicGamma

```python
class MagicGamma(ClassificationDataset):
 |  MagicGamma(root, split=TRAIN, validation_size=0.2)
```

__Parameters__

- __root__ (`str`): Local path for storing/reading dataset files.
- __split__ (`str`): One of {'train', 'validation', 'test'}
- __validation_size__ (`float`): How large fraction in (0, 1) of the training partition to use for validation.

<a name=".ucimlr.__init__.classification_datasets.SensorLessDrive"></a>
### SensorLessDrive

```python
class SensorLessDrive(ClassificationDataset):
 |  SensorLessDrive(root, split=TRAIN, validation_size=0.2)
```

__Parameters__

- __root__ (`str`): Local path for storing/reading dataset files.
- __split__ (`str`): One of {'train', 'validation', 'test'}
- __validation_size__ (`float`): How large fraction in (0, 1) of the training partition to use for validation.

<a name=".ucimlr.__init__.classification_datasets.Shuttle"></a>
### Shuttle

```python
class Shuttle(ClassificationDataset):
 |  Shuttle(root, split=TRAIN, validation_size=0.2)
```

Description of dataset [here](http://archive.ics.uci.edu/ml/datasets/Statlog+(Shuttle)).

__Parameters__

- __root__ (`str`): Local path for storing/reading dataset files.
- __split__ (`str`): One of {'train', 'validation', 'test'}
- __validation_size__ (`float`): How large fraction in (0, 1) of the training partition to use for validation.

<a name=".ucimlr.__init__.helpers"></a>
## ucimlr.\_\_init\_\_.helpers

<a name=".ucimlr.__init__.helpers.download_file"></a>
#### download\_file

```python
download_file(url, dataset_path, filename)
```

Downloads the file at the url to the path 'dataset_path/filename'.

Returns True if dataset was downloaded, False if already existing

<a name=".ucimlr.__init__.helpers.download_unzip"></a>
#### download\_unzip

```python
download_unzip(url: str, dataset_path: str)
```

Downloads the file at the specified 'url' and unzips it
at the specified 'dataset_path'.

<a name=".ucimlr.__init__.helpers.one_hot_encode_df_"></a>
#### one\_hot\_encode\_df\_

```python
one_hot_encode_df_(df, skip_columns=None)
```

One-hot encodes all non-numeric columns and drops those columns.
This is done in place.

<a name=".ucimlr.__init__.helpers.normalize_df_"></a>
#### normalize\_df\_

```python
normalize_df_(df_train, *, other_dfs=None, skip_column=None)
```

Normalizes all columns of `df_train` to zero mean unit variance in place.
Optionally performs same transformation to `other_dfs`

__Parameters__

- __other_dfs [pd.DataFrame]__: List of other DataFrames to apply transformation to
- __skip_column__ (`str, int`): Column to omit, for example categorical targets.

<a name=".ucimlr.__init__.helpers.xy_split"></a>
#### xy\_split

```python
xy_split(df: pd.DataFrame, y_columns: list)
```

Takes a DataFrame and return X and Y numpy arrays where
Y is given by the 'y_columns' argument.

<a name=".ucimlr.__init__.helpers.label_encode_df_"></a>
#### label\_encode\_df\_

```python
label_encode_df_(dfs, y_column)
```

Label encodes in place.

__Parameters__

- __dfs__:  DataFrame or list of DataFrames
- __y_column__: The label column

<a name=".ucimlr.__init__.helpers.split_normalize_sequence"></a>
#### split\_normalize\_sequence

```python
split_normalize_sequence(df: pd.DataFrame, y_columns, validation_size: float, split, dataset_type)
```

Performs the common sequence of operations:
train_test_split -> normalize -> x, y split

<a name=".ucimlr.__init__.helpers.split_df"></a>
#### split\_df

```python
split_df(df: pd.DataFrame, fractions)
```

Randomly splits (always with same seed) the dataframe `df` into
`fractions`.

If you want one split to be always the same (given the same fraction of course),
let that split be the first one. For example:
```
df_test, df_train, df_valid = split_df(df, [0.2, 0.4, 0.4])
```
or
```
df_test, df_train, df_valid = split_df(df, [0.2, 0.7, 0.1])
```
These always produce the same df_test.

<a name=".ucimlr.__init__.helpers.split_classification_df"></a>
#### split\_classification\_df

```python
split_classification_df(df, fractions, y_column)
```

Like split_df but groups into respective classes and then samples
fractions of those groups. Makes sure that:
1) No labels/classes are missing from any of the splits
2) The proportions of classes are the same in all splits

<a name=".ucimlr.__init__.helpers.split_df_on_column"></a>
#### split\_df\_on\_column

```python
split_df_on_column(df, fractions, column)
```

Randomly splits (always with same seed) the dataframe `df` into
`fractions`. Values in the `column` are disjoint over the splits.

