import unittest

import numpy as np

import enstat.mean


class Test_mean(unittest.TestCase):
    def test_Scalar(self):

        average = enstat.mean.Scalar()

        a = np.random.random(50 * 20).reshape(50, 20)

        for i in range(a.shape[0]):
            average.add_sample(a[i, :])

        self.assertTrue(np.isclose(average.mean(), np.mean(a)))
        self.assertTrue(np.isclose(average.std(), np.std(a), rtol=1e-3))

    def test_StaticNd(self):

        average = enstat.mean.StaticNd()

        a = np.random.random(35 * 50 * 20).reshape(35, 50, 20)

        for i in range(a.shape[0]):
            average.add_sample(a[i, :, :])

        self.assertTrue(np.allclose(average.mean(), np.mean(a, axis=0)))
        self.assertTrue(
            np.allclose(average.std(), np.std(a, axis=0), rtol=5e-1, atol=1e-3)
        )
        self.assertTrue(average.shape() == a.shape[1:])
        self.assertTrue(average.size() == np.prod(a.shape[1:]))

    def test_StaticNd_mask(self):

        average = enstat.mean.StaticNd()

        a = np.random.random(35 * 50 * 20).reshape(35, 50, 20)
        m = np.random.random(35 * 50 * 20).reshape(35, 50, 20) > 0.8

        for i in range(a.shape[0]):
            average.add_sample(a[i, :, :], m[i, :, :])

        self.assertTrue(
            np.isclose(
                np.sum(average.first()) / np.sum(average.norm()),
                np.mean(a[np.logical_not(m)]),
            )
        )

        self.assertTrue(
            np.isclose(
                np.sum(average.first()) / np.sum(average.norm()),
                np.mean(a[np.logical_not(m)]),
            )
        )

        self.assertTrue(
            np.all(np.equal(average.norm(), np.sum(np.logical_not(m), axis=0)))
        )

    def test_Dynamic1d(self):

        average = enstat.mean.Dynamic1d()

        average.add_sample(np.array([1, 2, 3]))
        average.add_sample(np.array([1, 2, 3]))
        average.add_sample(np.array([1, 2]))
        average.add_sample(np.array([1]))

        self.assertTrue(np.allclose(average.mean(), np.array([1, 2, 3])))
        self.assertTrue(np.allclose(average.std(), np.array([0, 0, 0])))
        self.assertTrue(average.shape() == [3])
        self.assertTrue(average.size() == 3)


if __name__ == "__main__":

    unittest.main()
