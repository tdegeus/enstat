# enstat

[![CI](https://github.com/tdegeus/enstat/workflows/CI/badge.svg)](https://github.com/tdegeus/enstat/actions)
[![Documentation Status](https://readthedocs.org/projects/enstat/badge/?version=latest)](https://enstat.readthedocs.io)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/enstat.svg)](https://anaconda.org/conda-forge/enstat)
[![PyPi release](https://img.shields.io/pypi/v/enstat.svg)](https://pypi.org/project/enstat/)

Documentation: [enstat.readthedocs.io](https://enstat.readthedocs.io)

## Overview

*enstat* is a library to facilitate the computation of ensemble averages
(and their standard deviation and variance).
The key feature is that a class stored the sum of the first and second statistical moments
and the number of samples,
such that adding a sample can be done trivially, while giving access to the mean etc.
at all times.

*enstat* allows you to compute the average (and variance) or the histogram of chunked data,
without the need to load all data at once.
For the average (and variance) this is done by keeping the sum of the first (and second)
statistical moment in memory, together with the normalisation.
For the histogram, the bins are updated for every 'chunk' (sample).
A common practical application is computing the average of an ensemble of realisations.

### Ensemble average

Suppose that we have 100 realisations each with 1000 blocks, and we want to know the ensemble
average of each block:

```python
import enstat

ensemble = enstat.static()

for realisation in range(100):

    sample = np.random.random(1000)
    ensemble += sample

mean = ensemble.mean()
print(mean.shape)
```

which outputs ``[1000]``.

## Installation

-   Using conda

    ```bash
    conda install -c conda-forge enstat
    ```

-   Using PyPi

    ```bash
    pip install enstat
    ```

## Disclaimer

This library is free to use under the
[MIT license](https://github.com/tdegeus/enstat/blob/master/LICENSE).
Any additions are very much appreciated.
As always, the code comes with no guarantee.
None of the developers can be held responsible for possible mistakes.
