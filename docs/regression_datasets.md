<a name=".ucimlr.regression_datasets"></a>
## ucimlr.regression\_datasets

<a name=".ucimlr.regression_datasets.all_datasets"></a>
#### all\_datasets

```python
all_datasets()
```

Returns a list of all RegressionDataset classes.

<a name=".ucimlr.regression_datasets.Abalone"></a>
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

<a name=".ucimlr.regression_datasets.AirFoil"></a>
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

<a name=".ucimlr.regression_datasets.AirQuality"></a>
### AirQuality

```python
class AirQuality(RegressionDataset):
 |  AirQuality(root, split=TRAIN, validation_size=0.2)
```

__Parameters__

- __root__ (`str`): Local path for storing/reading dataset files.
- __split__ (`str`): One of {'train', 'validation', 'test'}
- __validation_size__ (`float`): How large fraction in (0, 1) of the training partition to use for validation.

<a name=".ucimlr.regression_datasets.BlogFeedback"></a>
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

<a name=".ucimlr.regression_datasets.CTSlices"></a>
### CTSlices

```python
class CTSlices(RegressionDataset):
 |  CTSlices(root, split=TRAIN, validation_size=0.2)
```

__Parameters__

- __root__ (`str`): Local path for storing/reading dataset files.
- __split__ (`str`): One of {'train', 'validation', 'test'}
- __validation_size__ (`float`): How large fraction in (0, 1) of the training partition to use for validation.

<a name=".ucimlr.regression_datasets.FacebookComments"></a>
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

<a name=".ucimlr.regression_datasets.OnlineNews"></a>
### OnlineNews

```python
class OnlineNews(RegressionDataset):
 |  OnlineNews(root, split=TRAIN, validation_size=0.2)
```

__Parameters__

- __root__ (`str`): Local path for storing/reading dataset files.
- __split__ (`str`): One of {'train', 'validation', 'test'}
- __validation_size__ (`float`): How large fraction in (0, 1) of the training partition to use for validation.

<a name=".ucimlr.regression_datasets.Parkinson"></a>
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

<a name=".ucimlr.regression_datasets.PowerPlant"></a>
### PowerPlant

```python
class PowerPlant(RegressionDataset):
 |  PowerPlant(root, split=TRAIN, validation_size=0.2)
```

__Parameters__

- __root__ (`str`): Local path for storing/reading dataset files.
- __split__ (`str`): One of {'train', 'validation', 'test'}
- __validation_size__ (`float`): How large fraction in (0, 1) of the training partition to use for validation.

<a name=".ucimlr.regression_datasets.RealEstate"></a>
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

<a name=".ucimlr.regression_datasets.Superconductivity"></a>
### Superconductivity

```python
class Superconductivity(RegressionDataset):
 |  Superconductivity(root, split=TRAIN, validation_size=0.2)
```

__Parameters__

- __root__ (`str`): Local path for storing/reading dataset files.
- __split__ (`str`): One of {'train', 'validation', 'test'}
- __validation_size__ (`float`): How large fraction in (0, 1) of the training partition to use for validation.

<a name=".ucimlr.regression_datasets.WhiteWineQuality"></a>
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

