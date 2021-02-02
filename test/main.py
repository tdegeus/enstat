import enstat.mean
import numpy as np
import unittest

class Test_mean(unittest.TestCase):

    def test_Dynamic1d(self):

        average = enstat.mean.Dynamic1d()

        average.add_sample(np.array([1, 2, 3]))
        average.add_sample(np.array([1, 2]))
        average.add_sample(np.array([1]))

        self.assertTrue(np.allclose(average.mean(), np.array([1, 2, 3])))

if __name__ == '__main__':

    unittest.main()
