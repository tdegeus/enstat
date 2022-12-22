from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from numpy.typing import DTypeLike

from . import detail  # noqa: F401
from . import mean  # noqa: F401
from ._version import version  # noqa: F401
from ._version import version_tuple  # noqa: F401


class scalar:
    r"""
    Ensemble average of a scalar.
    Add samples to it using :py:func:`scalar.add_sample`.
    The mean, variance, and standard deviation can be obtained at any time.
    They are derived from the following members:

    *   :py:attr:`scalar.first`: Sum of the first statistical moment.
    *   :py:attr:`scalar.second`: Sum of the second statistical moment.
    *   :py:attr:`scalar.norm`: Number of samples.

    To continue an old average by specifying:

    :param float first: Sum of the first moment.
    :param float second: Sum of the second moment.
    :param int norm: Number of samples.
    """

    def __init__(self, first: float = 0, second: float = 0, norm: float = 0):

        self.first = first
        self.second = second
        self.norm = norm

    def __add__(self, datum: float):

        self.add_sample(datum)
        return self

    def add_sample(self, datum: float):
        """
        Add a sample.
        Internally changes the sums of the first and second statistical moments and normalisation.

        :param datum: Sample.
        """

        datum = np.array(datum)
        self.first += np.sum(datum)
        self.second += np.sum(datum**2)
        self.norm += datum.size

    def mean(self) -> float:
        r"""
        Current mean.
        Samples can be added afterwards without any problems.

        :return: Mean.
        """

        if self.norm == 0:
            return np.NaN

        return self.first / self.norm

    def variance(self) -> float:
        r"""
        Current variance.
        Samples can be added afterwards without any problems.

        :return: Variance.
        """

        if self.norm <= 1:
            return np.NaN

        d = (self.second / self.norm - (self.first / self.norm) ** 2) * self.norm
        n = self.norm - 1

        return d / n

    def std(self) -> float:
        r"""
        Current standard deviation.
        Samples can be added afterwards without any problems.

        :return: Standard deviation.
        """

        return np.sqrt(self.variance())


class static:
    r"""
    Ensemble average of an nd-array (of same size for all samples).
    Add samples to it using :py:func:`static.add_sample`.
    The mean, variance, and standard deviation can be obtained at any time.
    They are derived from the following members:

    *   :py:attr:`static.first`: Sum of the first statistical moment.
    *   :py:attr:`static.second`: Sum of the second statistical moment.
    *   :py:attr:`static.norm`: Number of samples.

    Furthermore, the following members are available:

    *   :py:attr:`static.shape`: Shape of the data.
    *   :py:attr:`static.size`: Size of the data (= prod(shape)).

    :param compute_variance:
        If set ``False`` no second moment will be computed (making things slightly faster).
        In that case, the variance an standard deviation will not be available.

    :param shape:
        The shape of the data.
        If not specified it is determined form the first sample.

    :param dtype:
        The type of the data.
        If not specified it is determined form the first sample.

    :param first: Continued computation: Sum of the first moment.
    :param second: Continued computation: Sum of the second moment.
    :param norm: Continued computation: Number of samples (integer).
    """

    def __init__(
        self,
        compute_variance: bool = True,
        shape: tuple[int] = None,
        dtype: DTypeLike = np.float64,
        first: ArrayLike = None,
        second: ArrayLike = None,
        norm: ArrayLike = None,
    ):

        self.compute_variance = compute_variance
        self.norm = norm
        self.first = first
        self.second = second

        if norm is not None:
            assert first is not None
            assert second is not None or not compute_variance

        if shape is None:
            return

        if norm is not None:
            assert shape == norm.shape
            assert shape == second.shape
            assert shape == norm.shape
        else:
            self.norm = np.zeros(shape, np.int64)
            self.first = np.zeros(shape, dtype)
            if compute_variance:
                self.second = np.zeros(shape, dtype)

    def _allocate(self, shape, dtype):

        self.norm = np.zeros(shape, np.int64)
        self.first = np.zeros(shape, dtype)
        if self.compute_variance:
            self.second = np.zeros(shape, dtype)

    @property
    def dtype(self):
        """
        The type of the data.
        """
        return self.first.dtype

    @property
    def shape(self):
        """
        The shape of the data.
        """
        return self.first.shape

    @property
    def size(self):
        """
        The size of the data.
        """
        return np.prod(self.first.shape)

    def ravel(self) -> scalar:
        r"""
        Return as :py:class:`scalar`: all entries are summed.

        :return: Ensemble average.
        """

        return scalar(
            first=np.sum(self.first),
            second=np.sum(self.second),
            norm=np.sum(self.norm),
        )

    def __add__(self, data: ArrayLike):

        self.add_sample(data)
        return self

    def add_sample(self, data: ArrayLike, mask: ArrayLike = None):
        r"""
        Add a sample.
        Internally changes the sums of the first and second statistical moments and normalisation.

        :param data: The sample.
        :param mask: Mask entries (boolean array).
        """

        if self.first is None:
            self._allocate(data.shape, data.dtype)

        # masked data
        if mask is not None:
            incl = np.logical_not(mask)
            self.norm[incl] += 1
            self.first[incl] += data[incl]
            if self.compute_variance:
                self.second[incl] += data[incl] ** 2
            return

        # unmasked data
        self.norm += 1
        self.first += data
        if self.compute_variance:
            self.second += data**2

    def add_point(self, datum: float | int, index: int):
        """
        Add a single point. Note that::

            ensemble.add_point(datum, index)

        Is equivalent to::

            data = np.empty(ensemble.shape)
            mask = np.ones(ensemble.shape, dtype=bool)
            data[index] = datum
            mask[index] = False
            ensemble.add_sample(data, mask)

        (but faster).
        """

        if self.first is None:
            raise OSError("shape should be pre-specified")

        self.norm[index] += 1
        self.first[index] += datum

        if self.compute_variance:
            self.second[index] += datum**2

    def mean(self, min_norm: int = 1) -> ArrayLike:
        r"""
        Current mean.
        Samples can be added afterwards without any problems.

        :param min_norm: Minimum number of samples to consider as value output.
        :return: Mean.
        """

        if self.norm is None:
            return None

        n = np.where(self.norm > 0, self.norm, 1)
        ret = self.first / n
        return np.where(self.norm >= min_norm, ret, np.NaN)

    def variance(self, min_norm: int = 2) -> ArrayLike:
        r"""
        Current variance.
        Samples can be added afterwards without any problems.

        :param min_norm: Minimum number of samples to consider as value output.
        :return: Variance.
        """

        if self.norm is None:
            return None

        assert self.compute_variance
        n = np.where(self.norm > 1, self.norm, 2)
        ret = (self.second / n - (self.first / n) ** 2) * n / (n - 1)
        return np.where(self.norm >= min_norm, ret, np.NaN)

    def std(self, min_norm: int = 2) -> ArrayLike:
        r"""
        Current standard deviation.
        Samples can be added afterwards without any problems.

        :param min_norm: Minimum number of samples to consider as value output.
        :return: Standard deviation.
        """

        if self.norm is None:
            return None

        return np.sqrt(self.variance(min_norm))


def _expand_array1d(data, size):

    tmp = np.zeros((size), data.dtype)
    tmp[: data.size] = data
    return tmp


class dynamic1d(static):
    r"""
    Ensemble average of an 1d-array (which grows depending of the size of the samples).
    Add samples to it using :py:func:`dynamic1d.add_sample`.
    The mean, variance, and standard deviation can be obtained at any time.
    Also the sums of the first and statistical moments, as well as the number of samples can be
    obtained at any time.

    Continue an old average by specifying:

    :param compute_variance:
        If set ``False`` no second moment will be computed.
        In that case, the variance an standard deviation will not be available.

    :param size:
        The initial size of the data.
        If not specified it is determined form the first sample.

    :param dtype:
        The type of the data.
        If not specified it is determined form the first sample.

    :param first: Continued computation: Sum of the first moment.
    :param second: Continued computation: Sum of the second moment.
    :param norm: Continued computation: Number of samples (integer).
    """

    def __init__(
        self,
        compute_variance: bool = True,
        size: int = None,
        dtype: DTypeLike = np.float64,
        first: ArrayLike = None,
        second: ArrayLike = None,
        norm: ArrayLike = None,
    ):
        super().__init__(
            compute_variance=compute_variance,
            shape=(size,) if size is not None else None,
            dtype=dtype,
            first=first,
            second=second,
            norm=norm,
        )

    def _expand(self, size: int):

        if size <= self.first.size:
            return

        self.norm = _expand_array1d(self.norm, size)
        self.first = _expand_array1d(self.first, size)

        if self.compute_variance:
            self.second = _expand_array1d(self.second, size)

    def __add__(self, data: ArrayLike):

        self.add_sample(data)
        return self

    def add_sample(self, data: ArrayLike):

        assert data.ndim == 1

        if self.first is None:
            super().add_sample(data)
            return

        self._expand(data.size)
        self.norm[: data.size] += 1
        self.first[: data.size] += data

        if self.compute_variance:
            self.second[: data.size] += data**2

    def add_point(self, datum: float | int, index: int):

        if self.first is None:
            self._allocate(index + 1, type(datum))
        else:
            self._expand(index + 1)

        return super().add_point(datum, index)


class Histogram:
    """
    Histogram.
    One can add samples to it using :py:func:`Histogram.add_sample`.

    :param bin_edges: The bin-edges.
    :param right: Whether the bin includes the right edge (or the left edge) see numpy.digitize.
    :param count: The initial count (default: zeros).
    """

    def __init__(self, bin_edges: ArrayLike, right: bool = False, count: ArrayLike = None):

        assert np.all(np.diff(bin_edges) > 0) or np.all(np.diff(bin_edges) < 0)

        self.right = right
        self.bin_edges = np.array(bin_edges)

        if count is not None:
            assert len(count) == len(bin_edges) - 1
            self.count = np.array(count).astype(np.uint64)
        else:
            self.count = np.zeros((len(bin_edges) - 1), np.uint64)

    def strip(self, min_count: int = 0):
        """
        Strip the histogram of empty bins to the left and the right.

        :param min_count: The minimum count for a bin to be considered non-empty.
        """

        self.lstrip(min_count)
        self.rstrip(min_count)

    def lstrip(self, min_count: int = 0):
        """
        Strip the histogram of empty bins to the left.

        :param min_count: The minimum count for a bin to be considered non-empty.
        """

        i = np.argmax(self.count > min_count)
        self.count = self.count[i:]
        self.bin_edges = self.bin_edges[i:]

    def rstrip(self, min_count: int = 0):
        """
        Strip the histogram of empty bins to the right.

        :param min_count: The minimum count for a bin to be considered non-empty.
        """

        i = len(self.count) - np.argmax(self.count[::-1] > min_count)
        self.count = self.count[:i]
        self.bin_edges = self.bin_edges[: i + 1]

    def interp(self, bin_edges: ArrayLike):
        """
        Interpolate the histogram to a new set of bin-edges.

        :param bin_edges: The new bin-edges.
        """

        assert np.all(np.diff(bin_edges) > 0) or np.all(np.diff(bin_edges) < 0)

        m = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        self.count = np.interp(m, self.x, self.count)
        self.bin_edges = bin_edges

    def squash(self, n: int):
        """
        Squash the histogram by combining ``n`` sequential bins into one
        (the last bin may be smaller).

        :param n: Number of bins to group.
        """

        if n >= self.count.size:
            self.count = np.sum(self.count).reshape(1)
            self.bin_edges = self.bin_edges[[0, -1]]
            return

        m = self.count.size // n
        s = m * n
        self.count = np.hstack((self.count[:s].reshape((m, n)).sum(axis=1), np.sum(self.count[s:])))

        if len(self.bin_edges) // n != self.count.size + 1:
            self.bin_edges = np.hstack((self.bin_edges[::n], self.bin_edges[-1]))
        else:
            self.bin_edges = self.bin_edges[::n]

    def merge_right(self, index: ArrayLike):
        """
        Merge the bins to the right of ``index`` into ``index``.

        :param index: The indices of the bin to merge into.
        """

        index = np.array(index)
        index = np.sort(np.where(index < 0, index + self.count.size, index))

        if index[-1] == self.count.size - 1:
            index = index[:-1]

        self.count[index] += self.count[index + 1]
        self.count = np.delete(self.count, index + 1)
        self.bin_edges = np.delete(self.bin_edges, index + 1)

    def merge_left(self, index: ArrayLike):
        """
        Merge the bins to the left of ``index`` into ``index``.

        :param index: The indices of the bin to merge into.
        """

        index = np.array(index)
        index = np.sort(np.where(index < 0, index + self.count.size, index))

        if index[0] == 0:
            index = index[1:]

        self.count[index - 1] += self.count[index]
        self.count = np.delete(self.count, index)
        self.bin_edges = np.delete(self.bin_edges, index)

    def as_integer(self):
        """
        Merge bins not encompassing an integer with the preceding bin.
        For example: a bin with edges ``[1.1, 1.9]`` is removed, but ``[0.9, 1.1]`` is not removed.
        """

        assert self.bin_edges.size > 1

        merge = np.argwhere(np.diff(np.floor(self.bin_edges)) == 0).ravel()
        self.merge_right(merge)
        self.bin_edges = np.floor(self.bin_edges)
        self.right = False

    def __add__(self, data: ArrayLike):

        self.add_sample(data)
        return self

    def add_sample(self, data: ArrayLike):
        """
        Add a sample to the histogram.
        You can also use the ``+`` operator.
        """

        bin = np.digitize(data, self.bin_edges, self.right) - 1
        self.count += np.bincount(bin, minlength=self.count.size).astype(np.uint64)

    @property
    def x(self) -> ArrayLike:
        """
        The bin centers.
        """

        return 0.5 * (self.bin_edges[:-1] + self.bin_edges[1:])

    @property
    def density(self) -> ArrayLike:
        """
        The probability density function at the bin.
        """

        return self.count / np.sum(np.diff(self.bin_edges) * self.count)


def auto_histogram(
    data: ArrayLike,
    bins: int = None,
    mode: str = "equal",
    min_count: int = None,
    min_width: float = None,
    integer: bool = False,
    bin_edges: ArrayLike = None,
):
    r"""
    Construct a histogram from data.

    :param data: Data (flattened).
    :param bins: Number of bins.
    :param mode:
        Mode with which to compute the bin-edges.

        -   ``'equal'``: each bin has equal width.
        -   ``'log'``: logarithmic spacing.
        -   ``'uniform'``: uniform number of data-points per bin.
        -   ``'voronoi'``: each bin is the region between two adjacent data-points.

    :param min_count: Minimum number of data-points per bin.
    :param min_width: Minimum width of a bin.

    :param integer:
        If ``True``, bins not encompassing an integer are removed
        (e.g. a bin with edges ``[1.1, 1.9]`` is removed, but ``[0.9, 1.1]`` is not removed).

    :param bin_edges: Specify the bin-edges (overrides ``bins`` and ``mode``).

    :return: The :py:class:`Histogram` object.
    """

    if hasattr(bins, "__len__"):
        raise OSError("Only the number of bins can be specified")

    if bin_edges is None:
        bin_edges = detail.histogram_bin_edges(data, bins, mode, min_count)

    if min_count is not None:
        bin_edges = detail.histogram_bin_edges_mincount(data, bin_edges, min_count)

    if min_width is not None:
        bin_edges = detail.histogram_bin_edges_minwidth(bin_edges, min_width)

    if integer:
        bin_edges = detail.histogram_bin_edges_integer(bin_edges)

    return Histogram(
        bin_edges, right=True, count=np.histogram(data, bins=bin_edges, density=False)[0]
    )


if __name__ == "__main__":
    pass
