import unittest

import numpy as np

import enstat


class Test_binned(unittest.TestCase):
    """
    Binned data.
    """

    def test_simple(self):
        """
        Plain data, one variables.
        """
        x = np.random.random([1234])

        bin_edges = np.linspace(0, 1, 13)

        bin = np.digitize(x.ravel(), bin_edges) - 1
        n = bin_edges.size - 1

        xmean = np.NaN * np.ones(n, dtype=float)
        xerr = np.NaN * np.ones(n, dtype=float)
        n = np.zeros(n, dtype=int)

        for ibin in range(np.max(bin) + 1):
            sel = bin == ibin
            n[ibin] = np.sum(sel)

            xi = x[sel]
            xmean[ibin] = np.mean(xi)
            xerr[ibin] = np.std(xi)

        binned = enstat.binned.from_data(x, bin_edges=bin_edges)

        self.assertTrue(np.allclose(binned[0].mean(), xmean))
        self.assertTrue(np.allclose(binned[0].std(), xerr, rtol=1e-2, atol=1e-5))

    def test_simple_three(self):
        """
        Plain data, three variables.
        """
        x = np.random.random([1234])
        y = np.random.random(x.shape)
        z = np.random.random(x.shape)

        bin_edges = np.linspace(0, 1, 13)

        bin = np.digitize(x.ravel(), bin_edges) - 1
        n = bin_edges.size - 1

        xmean = np.NaN * np.ones(n, dtype=float)
        ymean = np.NaN * np.ones(n, dtype=float)
        zmean = np.NaN * np.ones(n, dtype=float)
        xerr = np.NaN * np.ones(n, dtype=float)
        yerr = np.NaN * np.ones(n, dtype=float)
        zerr = np.NaN * np.ones(n, dtype=float)
        n = np.zeros(n, dtype=int)

        for ibin in range(np.max(bin) + 1):
            sel = bin == ibin
            n[ibin] = np.sum(sel)

            xi = x[sel]
            yi = y[sel]
            zi = z[sel]
            xmean[ibin] = np.mean(xi)
            ymean[ibin] = np.mean(yi)
            zmean[ibin] = np.mean(zi)
            xerr[ibin] = np.std(xi)
            yerr[ibin] = np.std(yi)
            zerr[ibin] = np.std(zi)

        binned = enstat.binned.from_data(x, y, z, bin_edges=bin_edges)

        self.assertTrue(np.allclose(binned[0].mean(), xmean))
        self.assertTrue(np.allclose(binned[1].mean(), ymean))
        self.assertTrue(np.allclose(binned[2].mean(), zmean))
        self.assertTrue(np.allclose(binned[0].std(), xerr, rtol=1e-2, atol=1e-5))
        self.assertTrue(np.allclose(binned[1].std(), yerr, rtol=1e-2, atol=1e-5))
        self.assertTrue(np.allclose(binned[2].std(), zerr, rtol=1e-2, atol=1e-5))

    def test_ensemble(self):
        """
        Ensemble data, two variables.
        """
        x = np.random.random([123, 10])
        y = np.random.random(x.shape)
        bin_edges = np.linspace(0, 1, 13)
        bin = np.digitize(x.ravel(), bin_edges) - 1
        n = bin_edges.size - 1

        xmean = np.NaN * np.ones(n, dtype=float)
        ymean = np.NaN * np.ones(n, dtype=float)
        xerr = np.NaN * np.ones(n, dtype=float)
        yerr = np.NaN * np.ones(n, dtype=float)
        n = np.zeros(n, dtype=int)

        for ibin in range(np.max(bin) + 1):
            sel = bin == ibin
            n[ibin] = np.sum(sel)

            xi = x.ravel()[sel]
            yi = y.ravel()[sel]
            xmean[ibin] = np.mean(xi)
            ymean[ibin] = np.mean(yi)
            xerr[ibin] = np.std(xi)
            yerr[ibin] = np.std(yi)

        binned = enstat.binned(bin_edges)
        for i in range(x.shape[0]):
            binned.add_sample(x[i, :], y[i, :])

        self.assertTrue(np.allclose(binned[0].mean(), xmean))
        self.assertTrue(np.allclose(binned[1].mean(), ymean))
        self.assertTrue(np.allclose(binned[0].std(), xerr, rtol=1e-2, atol=1e-5))
        self.assertTrue(np.allclose(binned[1].std(), yerr, rtol=1e-2, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
