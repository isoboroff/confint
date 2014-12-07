#!/usr/bin/env python3

import sys
import collections
import fileinput
import numpy as np
import ci
from scikits import bootstrap


sc = {}
measures = ['MSnDCG_1000','nERR_1000','Q_1000']

for line in fileinput.input():
    (measure, topic, score) = line.split()
    if measure not in measures:
        continue
    if topic == 'all':
        break
    if measure not in sc:
        sc[measure] = {}
    sc[measure][topic] = float(score)

for measure in measures:
    values = np.fromiter(sc[measure].values(), np.float)
    mean = values.mean()

    lo, hi = ci.t_ci_mean(values)
    shap = ci.shape(mean, lo, hi)
    cover = ci.coverage(values, ci.t_ci_mean)
    print('{} t {:.4f} [{:.4f},{:.4f}] {:.2f} {:.3f}'.format(measure, values.mean(), lo, hi, shap, cover))

    lo, hi = ci.bootstrap_t_ci_mean(values)
    shap = ci.shape(mean, lo, hi)
    cover = ci.coverage(values, ci.bootstrap_t_ci_mean)
    print('{} bootstrap_t {:.4f} [{:.4f},{:.4f}] {:.2f} {:.3f}'.format(measure, values.mean(), lo, hi, shap, cover))

    lo, hi = ci.bootstrap_pct_ci_mean(values)
    shap = ci.shape(mean, lo, hi)
    cover = ci.coverage(values, ci.bootstrap_pct_ci_mean)
    print('{} bootstrap_pct {:.4f} [{:.4f},{:.4f}] {:.2f} {:.3f}'.format(measure, values.mean(), lo, hi, shap, cover))

    lo, hi = bootstrap.ci(values, n_samples=2000)
    shap = ci.shape(mean, lo, hi)
    cover = ci.coverage(values, lambda x: bootstrap.ci(x, n_samples=2000))
    print('{} sk.bs-bca {:.4f} [{:.4f},{:.4f}] {:.2f} {:.3f}'.format(measure, values.mean(), lo, hi, shap, cover))
