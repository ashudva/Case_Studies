import numpy as np
import scipy.stats as ss


def make_synth_data(n=50):
    """Returns 2n bivariate random variates and labels """
    labels = np.concatenate((np.repeat(0, n), np.repeat(1, n)))
    points = np.concatenate((ss.norm(0, 1).rvs((n, 2)), ss.norm(1, 1).rvs(
        (n, 2))),
        axis=0)
    return points, labels
