# enstat

[![CI](https://github.com/tdegeus/enstat/workflows/CI/badge.svg)](https://github.com/tdegeus/enstat/actions)
[![Documentation Status](https://readthedocs.org/projects/enstat/badge/?version=latest)](https://enstat.readthedocs.io)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/enstat.svg)](https://anaconda.org/conda-forge/enstat)
[![PyPi release](https://img.shields.io/pypi/v/enstat.svg)](https://pypi.org/project/enstat/)

Documentation: [enstat.readthedocs.io](https://enstat.readthedocs.io)

## Overview

*enstat* is a library to facilitate the computation of ensemble averages
(and their variances) or histograms.

The key feature is that a class stores the sum of the first and second statistical moments
and the number of samples.
This gives access to the mean (and variance) at all times, while you can keep adding samples.

For the histogram something similar holds, but this time the count per bin in stored.

### Ensemble average

Suppose that we have 100 realisations each with 1000 'blocks', and we want to know the ensemble
average of each block:

```python
import enstat

ensemble = enstat.static()

for realisation in range(100):

    sample = np.random.random(1000)
    ensemble += sample

print(ensemble.mean())
```

### Ensemble histogram

Same example, but now we want the histogram for a number of pre-defined bins:
```python
import enstat

bin_edges = np.linspace(0, 1, 11)
hist = enstat.histogram(bin_edges=bin_edges)

for realisation in range(100):

    sample = np.random.random(1000)
    hist += sample

print(hist.p)
```

which prints the probability density of each bin (so list of values around `0.1` for these bins).

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
