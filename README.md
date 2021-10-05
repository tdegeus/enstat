# enstat

[![CI](https://github.com/tdegeus/enstat/workflows/CI/badge.svg)](https://github.com/tdegeus/enstat/actions)
[![Documentation Status](https://readthedocs.org/projects/enstat/badge/?version=latest)](https://enstat.readthedocs.io)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/enstat.svg)](https://anaconda.org/conda-forge/enstat)
[![PyPi release](https://img.shields.io/pypi/v/enstat.svg)](https://pypi.org/project/enstat/)

Documentation: [enstat.readthedocs.io](https://enstat.readthedocs.io)

# Overview

*enstat* is a library to facilitate the computation of ensemble averages
(and their standard deviation and variance).
The key feature is that a class stored the sum of the first and second statistical moments
and the number of samples,
such that adding a sample can be done trivially, while giving access to the mean etc.
at all times.

**This library is being developed, so for the moment subject to arbitrary changes.**

# Disclaimer

This library is free to use under the
[MIT license](https://github.com/tdegeus/enstat/blob/master/LICENSE).
Any additions are very much appreciated, in terms of suggested functionality, code,
documentation, testimonials, word-of-mouth advertisement, etc.
Bug reports or feature requests can be filed on
[GitHub](https://github.com/tdegeus/enstat).
As always, the code comes with no guarantee.
None of the developers can be held responsible for possible mistakes.

Download:
[.zip file](https://github.com/tdegeus/enstat/zipball/master) |
[.tar.gz file](https://github.com/tdegeus/enstat/tarball/master).

(c - [MIT](https://github.com/tdegeus/enstat/blob/master/LICENSE))
T.W.J. de Geus (Tom) | tom@geus.me | www.geus.me |
[github.com/tdegeus/enstat](https://github.com/tdegeus/enstat)

# Installation

## Using conda

```bash
conda install -c conda-forge enstat
```

## Using PyPi

```bash
pip install enstat
```

# Change-log

## v0.3.0

*   Adding mask option to `enstat.static.StaticNd`.

## v0.2.0

*   Adding size and shape methods.
*   Various generalisations.
