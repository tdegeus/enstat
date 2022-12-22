import unittest

import numpy as np

import enstat


class Test_norm(unittest.TestCase):
    """
    Histogram normalisation.
    """

    def test_density(self):

        data = [0, 0, 0, 1, 1, 2]
        bin_edges = [-0.5, 0.5, 1.5, 2.5]
        p, _ = np.histogram(data, bins=bin_edges, density=True)

        hist = enstat.histogram.from_data(data, bin_edges=bin_edges)

        self.assertTrue(np.allclose(hist.density, p))

    def test_mid(self):

        bin_edges = [-0.5, 0.5, 1.5, 2.5]
        mid = [0, 1, 2]

        hist = enstat.histogram(bin_edges=bin_edges)

        self.assertEqual(mid, hist.x.tolist())


class Test_accumulate(unittest.TestCase):
    """
    Histogram accumulation.
    """

    def test_accumulate(self):

        data = np.random.random([100, 100])
        bin_edges = np.linspace(0, 1, 21)
        p, _ = np.histogram(data.ravel(), bins=bin_edges, density=True)

        hist = enstat.histogram(bin_edges=bin_edges)

        for i in range(data.shape[0]):
            hist += data[i, :]

        self.assertTrue(np.allclose(hist.density, p))


class Test_strip(unittest.TestCase):
    """
    Strip empty bins.
    """

    def test_lstrip(self):

        data = np.random.random(100)
        bin_edges = np.linspace(-1, 2, 25)
        count, _ = np.histogram(data, bins=bin_edges, density=False)

        hist = enstat.histogram.from_data(data, bin_edges=bin_edges)
        hist.lstrip()

        self.assertTrue(np.allclose(hist.bin_edges, bin_edges[8:]))
        self.assertTrue(np.all(hist.count == count[8:]))

    def test_rstrip(self):

        data = np.random.random(100)
        bin_edges = np.linspace(-1, 2, 25)
        count, _ = np.histogram(data, bins=bin_edges, density=False)

        hist = enstat.histogram.from_data(data, bin_edges=bin_edges)
        hist.rstrip()

        self.assertTrue(np.allclose(hist.bin_edges, bin_edges[:-8]))
        self.assertTrue(np.all(hist.count == count[:-8]))

    def test_strip(self):

        data = np.random.random(100)
        bin_edges = np.linspace(-1, 2, 25)
        count, _ = np.histogram(data, bins=bin_edges, density=False)

        hist = enstat.histogram.from_data(data, bin_edges=bin_edges)
        hist.strip()

        self.assertTrue(np.allclose(hist.bin_edges, bin_edges[8:-8]))
        self.assertTrue(np.all(hist.count == count[8:-8]))


class Test_merge(unittest.TestCase):
    """
    Merge bins.
    """

    def test_merge_right(self):

        bin_edges = [0, 1, 2, 3, 4, 5]
        count = [1, 2, 1, 2, 3]

        hist = enstat.histogram(bin_edges=bin_edges, count=count)
        hist.merge_right([0, 2])

        self.assertEqual([0, 2, 4, 5], hist.bin_edges.tolist())
        self.assertEqual([3, 3, 3], hist.count.tolist())

    def test_merge_right_first(self):

        bin_edges = [0, 1, 2, 3, 4, 5]
        count = [1, 2, 1, 2, 3]

        hist = enstat.histogram(bin_edges=bin_edges, count=count)
        hist.merge_right([0])

        self.assertEqual([0, 2, 3, 4, 5], hist.bin_edges.tolist())
        self.assertEqual([3, 1, 2, 3], hist.count.tolist())

    def test_merge_right_last(self):

        bin_edges = [0, 1, 2, 3, 4, 5]
        count = [1, 2, 1, 2, 3]

        hist = enstat.histogram(bin_edges=bin_edges, count=count)
        hist.merge_right([-1])

        self.assertEqual(bin_edges, hist.bin_edges.tolist())
        self.assertEqual(count, hist.count.tolist())

    def test_merge_left(self):

        bin_edges = [0, 1, 2, 3, 4, 5]
        count = [1, 2, 1, 2, 3]

        hist = enstat.histogram(bin_edges=bin_edges, count=count)
        hist.merge_left([0, 2])

        self.assertEqual([0, 1, 3, 4, 5], hist.bin_edges.tolist())
        self.assertEqual([1, 3, 2, 3], hist.count.tolist())

    def test_merge_left_first(self):

        bin_edges = [0, 1, 2, 3, 4, 5]
        count = [1, 2, 1, 2, 3]

        hist = enstat.histogram(bin_edges=bin_edges, count=count)
        hist.merge_left([0])

        self.assertEqual(bin_edges, hist.bin_edges.tolist())
        self.assertEqual(count, hist.count.tolist())

    def test_merge_left_last(self):

        bin_edges = [0, 1, 2, 3, 4, 5]
        count = [1, 2, 1, 2, 3]

        hist = enstat.histogram(bin_edges=bin_edges, count=count)
        hist.merge_left([-1])

        self.assertEqual([0, 1, 2, 3, 5], hist.bin_edges.tolist())
        self.assertEqual([1, 2, 1, 5], hist.count.tolist())

    def test_as_integer(self):

        bin_edges = [0, 1, 2, 2.1, 3, 3.1, 4, 5]
        count = [1, 2, 1, 1, 1, 2, 3]

        hist = enstat.histogram(bin_edges=bin_edges, count=count)
        hist.as_integer()

        self.assertEqual([0, 1, 2, 3, 4, 5], hist.bin_edges.tolist())
        self.assertEqual([1, 2, 2, 3, 3], hist.count.tolist())

    def test_as_integer_first(self):

        bin_edges = [0, 0.9, 1, 2, 3, 4, 5]
        count = [1, 2, 1, 1, 2, 3]

        hist = enstat.histogram(bin_edges=bin_edges, count=count)
        hist.as_integer()

        self.assertEqual([0, 1, 2, 3, 4, 5], hist.bin_edges.tolist())
        self.assertEqual([3, 1, 1, 2, 3], hist.count.tolist())

    def test_as_integer_last(self):

        bin_edges = [0, 1, 2, 3, 4, 4.5, 5]
        count = [1, 2, 1, 2, 1, 3]

        hist = enstat.histogram(bin_edges=bin_edges, count=count)
        hist.as_integer()

        self.assertEqual([0, 1, 2, 3, 4, 5], hist.bin_edges.tolist())
        self.assertEqual([1, 2, 1, 2, 4], hist.count.tolist())


class Test_squash(unittest.TestCase):
    """
    Squash ``n`` bins into one.
    """

    def test_squash_fit(self):

        bin_edges = [0, 1, 2, 3, 4, 6]
        count = [1, 2, 1, 2, 3]

        hist = enstat.histogram(bin_edges=bin_edges, count=count)
        hist.squash(2)

        self.assertEqual([3, 3, 3], hist.count.tolist())
        self.assertEqual([0, 2, 4, 6], hist.bin_edges.tolist())

    def test_squash_nofit(self):

        bin_edges = [0, 1, 2, 3, 4, 6]
        count = [1, 2, 1, 2, 3]

        hist = enstat.histogram(bin_edges=bin_edges, count=count)
        hist.squash(3)

        self.assertEqual([4, 5], hist.count.tolist())
        self.assertEqual([0, 3, 6], hist.bin_edges.tolist())

        hist = enstat.histogram(bin_edges=bin_edges, count=count)
        hist.squash(4)

        self.assertEqual([6, 3], hist.count.tolist())
        self.assertEqual([0, 4, 6], hist.bin_edges.tolist())

        hist = enstat.histogram(bin_edges=bin_edges, count=count)
        hist.squash(5)

        self.assertEqual([9], hist.count.tolist())
        self.assertEqual([0, 6], hist.bin_edges.tolist())

        hist = enstat.histogram(bin_edges=bin_edges, count=count)
        hist.squash(6)

        self.assertEqual([9], hist.count.tolist())
        self.assertEqual([0, 6], hist.bin_edges.tolist())

    def test_strip_squash(self):

        bin_edges = [-1, 0, 1, 2, 3, 4, 6, 7]
        count = [0, 1, 2, 1, 2, 3, 0]

        hist = enstat.histogram(bin_edges=bin_edges, count=count)
        hist.strip()
        hist.squash(2)

        self.assertEqual([3, 3, 3], hist.count.tolist())
        self.assertEqual([0, 2, 4, 6], hist.bin_edges.tolist())


class Test_histogram_bin_edges_integer(unittest.TestCase):
    """
    Bin edges: integer
    """

    def test_front(self):

        a = [0, 0.5, 1.5, 2.5]
        b = [0, 1.5, 2.5]
        self.assertTrue(np.allclose(enstat.detail.histogram_bin_edges_integer(a), b))

    def test_middle(self):

        a = [0, 1.5, 1.6, 2.5]
        b = [0, 1.6, 2.5]
        self.assertTrue(np.allclose(enstat.detail.histogram_bin_edges_integer(a), b))

    def test_back(self):

        a = [0, 1.5, 2.5, 2.6]
        b = [0, 1.5, 2.6]
        self.assertTrue(np.allclose(enstat.detail.histogram_bin_edges_integer(a), b))


class Test_histogram_bin_edges_voronoi(unittest.TestCase):
    """
    Bin edges: Voronoi
    """

    def test_integer(self):

        data = np.array([0, 0, 1, 1, 1, 2, 2, 3, 4, 5, 6])
        hist = enstat.histogram.from_data(data, bins=10, mode="voronoi")

        self.assertTrue(
            np.allclose(hist.bin_edges, np.array([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5]))
        )

    def test_integer2(self):

        data = np.array([8, 8, 1, 1, 1, 2, 2, 3, 4, 5, 6])
        hist = enstat.histogram.from_data(data, bins=10, mode="voronoi")

        self.assertTrue(np.allclose(hist.bin_edges, np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 7, 9])))


if __name__ == "__main__":

    unittest.main()
