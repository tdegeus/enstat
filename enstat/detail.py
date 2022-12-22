from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def histogram_bin_edges(data: ArrayLike, bins: int, mode: str, min_count: int = None) -> ArrayLike:
    """
    Determine the bin-edges.

    :param data: The input data.
    :param bins: The number of bins.
    :param mode: The binning mode.
    :param min_count: The minimum number of data-points per bin (only used for mode="uniform").
    """

    if mode == "equal":

        return np.linspace(np.min(data), np.max(data), bins + 1)

    if mode == "log":

        return np.logspace(np.log10(np.min(data)), np.log10(np.max(data)), bins + 1)

    if mode == "uniform":

        if min_count and bins is None:
            if not isinstance(min_count, int):
                raise OSError('"min_count" must be an integer number')
            bins = int(np.floor(float(len(data)) / float(min_count)))

        assert bins is not None

        # number of data-points in each bin (equal for each)
        count = int(np.floor(float(len(data)) / float(bins))) * np.ones(bins, dtype="int")

        # increase the number of data-points by one is an many bins as needed,
        # such that the total fits the total number of data-points
        count[np.linspace(0, bins - 1, len(data) - np.sum(count)).astype(int)] += 1

        # split the data
        idx = np.empty((bins + 1), dtype="int")
        idx[0] = 0
        idx[1:] = np.cumsum(count)
        idx[-1] = len(data) - 1

        # determine the bin-edges
        return np.unique(np.sort(data)[idx])

    if mode == "voronoi":

        mid_points = np.unique(data)
        diff = np.diff(mid_points)
        bin_edges = mid_points + np.hstack((0.5 * diff, 0.5 * diff[-1]))
        return np.hstack((mid_points[0] - 0.5 * diff[0], bin_edges))

    raise OSError("Unknown option")


def histogram_bin_edges_minwidth(bin_edges: ArrayLike, min_width: int) -> ArrayLike:
    r"""
    Merge bin_edges with right-neighbour until each bin has a minimum width.

    :param bin_edges: The bin-edges.
    :param min_width: The minimum bin width.
    :return: The bin-edges.
    """

    while True:

        idx = np.where(np.diff(bin_edges) < min_width)[0]

        if len(idx) == 0:
            return bin_edges

        idx = idx[0]

        if idx + 1 == len(bin_edges) - 1:
            bin_edges = np.hstack((bin_edges[:idx], bin_edges[-1]))
        else:
            j = idx + 1
            k = idx + 2
            bin_edges = np.hstack((bin_edges[:j], bin_edges[k:]))


def histogram_bin_edges_mincount(
    data: ArrayLike, bin_edges: ArrayLike, min_count: int
) -> ArrayLike:
    r"""
    Merge bins with right-neighbour until each bin has a minimum number of data-points.

    :param data: The input data.
    :param bin_edges: The bin-edges.
    :param min_count: The minimum number of data-points per bin.
    :return: The bin-edges.
    """

    assert isinstance(min_count, int)

    while True:

        count, _ = np.histogram(data, bins=bin_edges, density=False)

        idx = np.where(count < min_count)[0]

        if len(idx) == 0:
            return bin_edges

        idx = idx[0]

        if idx + 1 == len(count):
            bin_edges = np.hstack((bin_edges[:idx], bin_edges[-1]))
        else:
            j = idx + 1
            k = idx + 2
            bin_edges = np.hstack((bin_edges[:j], bin_edges[k:]))


def histogram_bin_edges_integer(bin_edges: ArrayLike) -> ArrayLike:
    r"""
    Merge bins not encompassing an integer with the preceding bin.
    For example: a bin with edges ``[1.1, 1.9]`` is removed, but ``[0.9, 1.1]`` is not removed.

    :param bin_edges: The bin-edges.
    :return: The bin-edges.
    """

    bin_edges = np.array(bin_edges)
    assert bin_edges.size > 1

    i = np.where(np.diff(np.floor(bin_edges)) >= 1)[0]

    if i[0] > 0:
        i[0] = 0

    i = list(i) + [bin_edges.size - 1]

    return bin_edges[i]
