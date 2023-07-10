# enstat

[![CI](https://github.com/tdegeus/enstat/workflows/CI/badge.svg)](https://github.com/tdegeus/enstat/actions)
[![Documentation Status](https://readthedocs.org/projects/enstat/badge/?version=latest)](https://enstat.readthedocs.io)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/enstat.svg)](https://anaconda.org/conda-forge/enstat)
[![PyPi release](https://img.shields.io/pypi/v/enstat.svg)](https://pypi.org/project/enstat/)

Documentation: [enstat.readthedocs.io](https://enstat.readthedocs.io)

## Hallmark feature

*enstat* is a library to compute ensemble statistics **without storing the entire ensemble in memory**.
In particular, it allows you to compute:

*   [Ensemble averages (and their variance)](#readme-average);
*   [Ensemble averages (and their variance) based on a certain binning of a quantity](#readme-binned);
*   [Histograms of an ensemble](#readme-histogram).

Below you find a quick-start.
For more information, see the [documentation](https://enstat.readthedocs.io).

<div id="readme-average"></div>

## Ensemble average

The key feature is to store the sum of the first and second statistical moments and the number of samples.
This gives access to the mean (and variance) at all times, while you can keep adding samples.

Suppose that we have 100 realisations, each with 1000 'items', and we want to know the ensemble average of each item:

```python
import enstat
import numpy as np

ensemble = enstat.static()

for realisation in range(100):
    sample = np.random.random(1000)
    ensemble += sample

print(ensemble.mean())
```

which will print a list of 1000 values, each around `0.5`.

This is the equivalent of

```python
import numpy as np

container = np.empty((100, 1000))

for realisation in range(100):
    sample = np.random.random(1000)
    container[realisation, :] = sample

print(np.mean(container, axis=0))
```

The key difference is that *enstat* only requires you to have `4 * N` values in memory for a sample of size `N`: the sample itself, the sums of the first and second moment, and the normalisation.
Instead the solution with the container uses much more memory.

A nice feature is also that you can keep adding samples to `ensemble`.
You can even store it and continue later.

<div id="readme-histogram"></div>

## Ensemble histogram

Same example, but now we want the histogram for predefined bins:
```python
import enstat
import numpy as np

bin_edges = np.linspace(0, 1, 11)
hist = enstat.histogram(bin_edges=bin_edges)

for realisation in range(100):
    sample = np.random.random(1000)
    hist += sample

print(hist.p)
```

which prints the probability density of each bin (so list of values around `0.1` for these bins).

The `histogram` class contains two additional nice features.

1.  It has several bin algorithms that NumPy does not have.

2.  It can be used for plotting with an ultra-sort interface, for example:

    ```python
    import enstat
    import numpy as np
    import matplotlib.pyplot as plt

    data = np.random.random(1000)
    hist = enstat.histogram.from_data(data, bins=10, mode="log")

    fig, ax = plt.subplots()
    ax.plot(hist.x, hist.p)
    plt.show()
    ```

    You can even use `ax.plot(*hist.plot)`.

<div id="readme-binned"></div>

## Average per bin

Suppose you have some time series (`t`) with multiple observables (`a` and `b`); e.g.;
```python
import enstat
import numpy as np

t = np.linspace(0, 10, 100)
a = np.random.normal(loc=5, scale=0.1, size=t.size)
b = np.random.normal(loc=1, scale=0.5, size=t.size)
```
Now suppose that you want to compute the average `a`, `b`, and `t` based on a certain binning of `t`:
```python
bin_edges = np.linspace(0, 12, 12)
binned = enstat.binned.from_data(t, a, b, names=["t", "a", "b"]m bin_edges=bin_edges)
print(binned["a"].mean())
```

## Installation

-   Using conda

    ```bash
    conda install -c conda-forge enstat
    ```

-   Using PyPi

    ```bash
    python -m pip install enstat
    ```

## Disclaimer

This library is free to use under the [MIT license](https://github.com/tdegeus/enstat/blob/master/LICENSE).
Any additions are very much appreciated.
As always, the code comes with no guarantee.
None of the developers can be held responsible for possible mistakes.
