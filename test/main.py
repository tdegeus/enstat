import enstat.mean
import numpy as np
import unittest

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


    def test_Dynamic1d(self):

        average = enstat.mean.Dynamic1d()

        average.add_sample(np.array([1, 2, 3]))
        average.add_sample(np.array([1, 2]))
        average.add_sample(np.array([1]))

        self.assertTrue(np.allclose(average.mean(), np.array([1, 2, 3])))


if __name__ == '__main__':

    unittest.main()
