<a name=".ucimlr.datasets"></a>
## ucimlr.datasets

<a name=".ucimlr.datasets.Abalone"></a>
### Abalone

```python
class Abalone(Dataset):
 |  Abalone(root, split=TRAIN, validation_size=0.2)
```

Link to the dataset [description](https://archive.ics.uci.edu/ml/datasets/Abalone).

__Parameters__

- __root__ (`str`): Local path for storing/reading dataset files.
- __split__ (`str`): One of {'train', 'validation', 'test'}
- __validation_size__ (`float`): How large fraction in (0, 1) of the training partition to use for validation.

<a name=".ucimlr.datasets.Adult"></a>
### Adult

```python
class Adult(Dataset):
 |  Adult(root, split=TRAIN, validation_size=0.2)
```

__Parameters__

- __root__ (`str`): Local path for storing/reading dataset files.
- __split__ (`str`): One of {'train', 'validation', 'test'}
- __validation_size__ (`float`): How large fraction in (0, 1) of the training partition to use for validation.

<a name=".ucimlr.datasets.AirQuality"></a>
### AirQuality

```python
class AirQuality(Dataset):
 |  AirQuality(root, split=TRAIN, validation_size=0.2)
```

__Parameters__

- __root__ (`str`): Local path for storing/reading dataset files.
- __split__ (`str`): One of {'train', 'validation', 'test'}
- __validation_size__ (`float`): How large fraction in (0, 1) of the training partition to use for validation.

<a name=".ucimlr.datasets.APSFailure"></a>
### APSFailure

```python
class APSFailure(Dataset):
 |  APSFailure(root, split=TRAIN, validation_size=0.2)
```

__Parameters__

- __root__ (`str`): Local path for storing/reading dataset files.
- __split__ (`str`): One of {'train', 'validation', 'test'}
- __validation_size__ (`float`): How large fraction in (0, 1) of the training partition to use for validation.

<a name=".ucimlr.datasets.Avila"></a>
### Avila

```python
class Avila(Dataset):
 |  Avila(root, split=TRAIN, validation_size=0.2)
```

__Parameters__

- __root__ (`str`): Local path for storing/reading dataset files.
- __split__ (`str`): One of {'train', 'validation', 'test'}
- __validation_size__ (`float`): How large fraction in (0, 1) of the training partition to use for validation.

<a name=".ucimlr.datasets.BankMarketing"></a>
### BankMarketing

```python
class BankMarketing(Dataset):
 |  BankMarketing(root, split=TRAIN, validation_size=0.2)
```

__Parameters__

- __root__ (`str`): Local path for storing/reading dataset files.
- __split__ (`str`): One of {'train', 'validation', 'test'}
- __validation_size__ (`float`): How large fraction in (0, 1) of the training partition to use for validation.

<a name=".ucimlr.datasets.BlogFeedback"></a>
### BlogFeedback

```python
class BlogFeedback(Dataset):
 |  BlogFeedback(root, split=TRAIN, validation_size=0.2)
```

Link to the dataset [description](http://archive.ics.uci.edu/ml/datasets/BlogFeedback).

__Parameters__

- __root__ (`str`): Local path for storing/reading dataset files.
- __split__ (`str`): One of {'train', 'validation', 'test'}
- __validation_size__ (`float`): How large fraction in (0, 1) of the training partition to use for validation.

<a name=".ucimlr.datasets.CardDefault"></a>
### CardDefault

```python
class CardDefault(Dataset):
 |  CardDefault(root, split=TRAIN, validation_size=0.2)
```

__Parameters__

- __root__ (`str`): Local path for storing/reading dataset files.
- __split__ (`str`): One of {'train', 'validation', 'test'}
- __validation_size__ (`float`): How large fraction in (0, 1) of the training partition to use for validation.

<a name=".ucimlr.datasets.CTSlices"></a>
### CTSlices

```python
class CTSlices(Dataset):
 |  CTSlices(root, split=TRAIN, validation_size=0.2)
```

__Parameters__

- __root__ (`str`): Local path for storing/reading dataset files.
- __split__ (`str`): One of {'train', 'validation', 'test'}
- __validation_size__ (`float`): How large fraction in (0, 1) of the training partition to use for validation.

<a name=".ucimlr.datasets.FacebookComments"></a>
### FacebookComments

```python
class FacebookComments(Dataset):
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

<a name=".ucimlr.datasets.Landsat"></a>
### Landsat

```python
class Landsat(Dataset):
 |  Landsat(root, split=TRAIN, validation_size=0.2)
```

__Parameters__

- __root__ (`str`): Local path for storing/reading dataset files.
- __split__ (`str`): One of {'train', 'validation', 'test'}
- __validation_size__ (`float`): How large fraction in (0, 1) of the training partition to use for validation.

<a name=".ucimlr.datasets.LetterRecognition"></a>
### LetterRecognition

```python
class LetterRecognition(Dataset):
 |  LetterRecognition(root, split=TRAIN, validation_size=0.2)
```

__Parameters__

- __root__ (`str`): Local path for storing/reading dataset files.
- __split__ (`str`): One of {'train', 'validation', 'test'}
- __validation_size__ (`float`): How large fraction in (0, 1) of the training partition to use for validation.

<a name=".ucimlr.datasets.MagicGamma"></a>
### MagicGamma

```python
class MagicGamma(Dataset):
 |  MagicGamma(root, split=TRAIN, validation_size=0.2)
```

__Parameters__

- __root__ (`str`): Local path for storing/reading dataset files.
- __split__ (`str`): One of {'train', 'validation', 'test'}
- __validation_size__ (`float`): How large fraction in (0, 1) of the training partition to use for validation.

<a name=".ucimlr.datasets.OnlineNews"></a>
### OnlineNews

```python
class OnlineNews(Dataset):
 |  OnlineNews(root, split=TRAIN, validation_size=0.2)
```

__Parameters__

- __root__ (`str`): Local path for storing/reading dataset files.
- __split__ (`str`): One of {'train', 'validation', 'test'}
- __validation_size__ (`float`): How large fraction in (0, 1) of the training partition to use for validation.

<a name=".ucimlr.datasets.Parkinson"></a>
### Parkinson

```python
class Parkinson(Dataset):
 |  Parkinson(root, split=TRAIN, validation_size=0.2)
```

Link to the dataset [description](http://archive.ics.uci.edu/ml/datasets/parkinsons).

__Parameters__

- __root__ (`str`): Local path for storing/reading dataset files.
- __split__ (`str`): One of {'train', 'validation', 'test'}
- __validation_size__ (`float`): How large fraction in (0, 1) of the training partition to use for validation.

<a name=".ucimlr.datasets.PowerPlant"></a>
### PowerPlant

```python
class PowerPlant(Dataset):
 |  PowerPlant(root, split=TRAIN, validation_size=0.2)
```

__Parameters__

- __root__ (`str`): Local path for storing/reading dataset files.
- __split__ (`str`): One of {'train', 'validation', 'test'}
- __validation_size__ (`float`): How large fraction in (0, 1) of the training partition to use for validation.

<a name=".ucimlr.datasets.SensorLessDrive"></a>
### SensorLessDrive

```python
class SensorLessDrive(Dataset):
 |  SensorLessDrive(root, split=TRAIN, validation_size=0.2)
```

__Parameters__

- __root__ (`str`): Local path for storing/reading dataset files.
- __split__ (`str`): One of {'train', 'validation', 'test'}
- __validation_size__ (`float`): How large fraction in (0, 1) of the training partition to use for validation.

<a name=".ucimlr.datasets.Shuttle"></a>
### Shuttle

```python
class Shuttle(Dataset):
 |  Shuttle(root, split=TRAIN, validation_size=0.2)
```

Description of dataset [here](http://archive.ics.uci.edu/ml/datasets/Statlog+(Shuttle)).

__Parameters__

- __root__ (`str`): Local path for storing/reading dataset files.
- __split__ (`str`): One of {'train', 'validation', 'test'}
- __validation_size__ (`float`): How large fraction in (0, 1) of the training partition to use for validation.

<a name=".ucimlr.datasets.Superconductivity"></a>
### Superconductivity

```python
class Superconductivity(Dataset):
 |  Superconductivity(root, split=TRAIN, validation_size=0.2)
```

__Parameters__

- __root__ (`str`): Local path for storing/reading dataset files.
- __split__ (`str`): One of {'train', 'validation', 'test'}
- __validation_size__ (`float`): How large fraction in (0, 1) of the training partition to use for validation.

<a name=".ucimlr.datasets.WhiteWineQuality"></a>
### WhiteWineQuality

```python
class WhiteWineQuality(Dataset):
 |  WhiteWineQuality(root, split=TRAIN, validation_size=0.2)
```

Description of dataset [here](https://archive.ics.uci.edu/ml/datasets/Wine+Quality).

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

