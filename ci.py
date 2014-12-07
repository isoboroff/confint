#!/usr/bin/env python3.4

import math
import numpy as np
import scipy.stats as stats

def t_ci_mean(data, alpha=0.05):
    m = data.mean()
    s = data.std()
    n = data.size
    moe = s / math.sqrt(n)
    ta = stats.t.ppf(1 - alpha/2, df=n-1)
    return m - (ta * moe), m + (ta * moe)

def bootstrap_t_ci_mean(data, alpha=0.05, samples=5000):
    # Get stats on the data
    m = data.mean()
    s = data.std()
    # Create the resampling index matrix: samples x data.size
    resample = np.random.randint(0, data.size, (samples, data.size))

    # Compute the row means of the resampled data
    resampled_means = np.mean(data[resample], axis=-1)
    # ... and the std error too
    resampled_std = np.std(data[resample], axis=-1)

    # The studentized margin of error
    resampled_moe = (resampled_means - m) / resampled_std

    # The CI is the [m - t^(1-alpha) * s, m - t^alpha * s
    # Yes, the alpha terms are the opposite of what you'd expect
    # and we subtract both times.  See Efron & Tibshirani p160
    return m - np.percentile(resampled_moe, 100-(100*alpha)) * s, m - np.percentile(resampled_moe, 100*alpha) * s

def bootstrap_pct_ci_mean(data, alpha=0.05, samples=5000):
    resample = np.random.randint(0, data.size, (samples, data.size))
    resampled_means = np.mean(data[resample], axis=-1)
    return np.percentile(resampled_means, 100*alpha/2.0), np.percentile(resampled_means, 100-(100*alpha/2.0))

# This method is to test if we can speed up the coverage calculation by
# vectorizing it.  The answer is no.
def boot_pci_ci_mean_and_cov(data, alpha=0.05, samples=5000, coverage_samples=1000):
    m = np.mean(data)
    resample = np.random.randint(0, data.size, (coverage_samples, samples, data.size))
    for i in range(1, 1000):
        resample[i, ...] = np.random.choice(resample[i, 0, ...], size=(5000, data.size), replace=True)
    rmeans = np.mean(data[resample], axis=-1)
    lo = np.percentile(rmeans, 100*alpha/2.0, axis=-1)
    hi = np.percentile(rmeans, 100-(100*alpha/2.0), axis=-1)
    coverage = sum(np.logical_and(m >= lo, m <= hi)) / coverage_samples
    return lo[0], hi[0], coverage

def shape(mean, lo, hi):
    if (mean - lo == 0):
        return float('nan')
    else:
        return (hi - mean) / (mean - lo)

def coverage(data, ci_fn, samples=1000):
    m = data.mean()
    resample_indexes = np.random.randint(0, data.size, (samples, data.size))
    resample = data[resample_indexes]
    ci_arr = np.apply_along_axis(ci_fn, 1, resample)

    # Check if m is inside each interval
    in_lo = np.greater_equal(m, ci_arr[:,0])
    in_hi = np.less_equal(m, ci_arr[:,1])
    count = sum(np.logical_and(in_lo, in_hi))
    return count / samples

