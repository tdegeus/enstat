import numpy as np

class Scalar:

    def __init__(self, first=0, second=0, norm=0):
        r'''
Allocate the ensemble average of a scalar.
Add samples to it using :py:func:`Scalar.add_sample`.
The mean, variance, and standard deviations can be obtained at any time.
Also the sums of the first and statistical moments, as well as the number of samples can be
obtained at any time.

Continue an old average by specifying:

:param first float: Sum of the first moment.
:param second float: Sum of the second moment.
:param norm int: Number of samples.
        '''

        self.m_first = first
        self.m_second = second
        self.m_norm = norm

    def add_sample(self, data):
        r'''
Add a sample.
Internally changes the sums of the first and second statistical moments and normalisation.

:param data float: the scalar sample.
        '''

        self.m_first += np.sum(data.ravel())
        self.m_second += np.sum(data.ravel() ** 2.0)
        self.m_norm += data.size

    def mean(self):
        r'''
Obtain the current mean.
N.B. samples can be added afterwards without any problems.
        '''

        if self.m_norm == 0:
            return 0

        return self.m_first / self.m_norm

    def variance(self):
        r'''
Obtain the current variance.
N.B. samples can be added afterwards without any problems.
        '''

        if self.m_norm == 0:
            return 0

        return (self.m_second / self.m_norm - (self.m_first / self.m_norm) ** 2) * self.m_norm / (self.m_norm - 1)

    def std(self):
        r'''
Obtain the current standard deviation.
N.B. samples can be added afterwards without any problems.
        '''

        return np.sqrt(self.variance())

    def norm(self):
        r'''
Return normalisation: the number of samples.

:return: Scalar.
        '''

        return self.m_norm

    def first(self):
        r'''
Sum of the first statistical moment.

:return: Scalar.
        '''

        return self.m_first

    def second(self):
        r'''
Sum of the second statistical moment.

:return: Scalar.
        '''

        return self.m_second


class StaticNd:

    def __init__(self, shape=None, dtype=None):

        self.m_first = None
        self.m_norm = None
        self.shape = shape
        self.dtype = dtype

    def _allocate(self, data):

        if self.m_first is not None:
            assert data.shape == self.shape
            return

        self.shape = self.shape if self.shape is not None else data.shape
        self.dtype = self.dtype if self.dtype is not None else data.dtype

        self.m_first = np.zeros(self.shape, self.dtype)
        self.m_norm = np.zeros(self.shape, np.int64)

    def add_sample(self, data):

        self._allocate(data)

        self.m_first += data
        self.m_norm += 1

    def mean(self):

        return self.m_first / np.where(self.m_norm > 0, self.m_norm, 1)


def _expand_array1d(data, size):

    tmp = np.zeros((size), data.dtype)
    tmp[: data.size] = data
    return tmp

class Dynamic1d:

    def __init__(self, size=None, dtype=None):

        self.m_first = None
        self.m_norm = None
        self.size = size
        self.dtype = dtype

    def _allocate(self, data):

        if self.m_first is not None:
            return

        size = self.size if self.size is not None else data.size
        dtype = self.dtype if self.dtype is not None else data.dtype

        self.m_first = np.zeros((size), dtype)
        self.m_norm = np.zeros((size), np.int64)

    def _expand(self, data):

        if data.size <= self.m_first.size:
            return

        self.m_first = _expand_array1d(self.m_first, data.size)
        self.m_norm = _expand_array1d(self.m_norm, data.size)

    def add_sample(self, data):

        assert data.ndim == 1

        self._allocate(data)
        self._expand(data)

        self.m_first[: data.size] += data
        self.m_norm[: data.size] += 1

    def mean(self):

        return self.m_first / np.where(self.m_norm > 0, self.m_norm, 1)

if __name__ == "__main__":
    pass
