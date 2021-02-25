# enstat

[![CI](https://github.com/tdegeus/enstat/workflows/CI/badge.svg)](https://github.com/tdegeus/enstat/actions)

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

