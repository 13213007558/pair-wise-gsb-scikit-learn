r"""
=================================================================================
Gaussian process regression with derivative observations (gradient-augmented GP)
=================================================================================

This example shows how :class:`~sklearn.gaussian_process.GaussianProcessRegressor`
can incorporate measurements of partial derivatives alongside function values.

We observe the function :math:`f(x) = \sin(x)` at a small number of positions
and, at a few of the positions, we also measured the slope
:math:`f'(x) = \cos(x)`. Passing the slopes in as derivative observations
(with the ``kind`` argument of
:func:`~sklearn.gaussian_process.GaussianProcessRegressor.fit`) makes the
regressor build the joint covariance of values and derivatives. A first
version of this feature supports an isotropic or anisotropic
:class:`~sklearn.gaussian_process.kernels.RBF` kernel multiplied by a
:class:`~sklearn.gaussian_process.kernels.ConstantKernel`.

Treating the slopes as ordinary ``y`` samples cannot express this model: an
ordinary observation at :math:`x_0` is assumed to equal :math:`f(x_0)`, while
here the measurement equals :math:`f'(x_0)`.

The bottom panel shows that the derivative-augmented GP also predicts the
derivative, using the same ``kind`` argument of
:func:`~sklearn.gaussian_process.GaussianProcessRegressor.predict`.
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# %%
import warnings

import matplotlib.pyplot as plt
import numpy as np

from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

# A handful of function values ...
x_values = np.linspace(-3, 3, 8)
y_values = np.sin(x_values)

# ... plus three measured slopes at selected positions.
x_slopes = np.array([-1.5, 0.0, 1.5])
y_slopes = np.cos(x_slopes)

# Stack everything together and tag each observation: -1 marks a function
# value, while j >= 0 marks the partial derivative with respect to feature j.
X_train = np.concatenate([x_values, x_slopes])[:, None]
y_train = np.concatenate([y_values, y_slopes])
kind = np.array([-1] * len(x_values) + [0] * len(x_slopes))

kernel = ConstantKernel(1.0, (1e-2, 1e2)) * RBF(1.0, (1e-3, 1e1))
gpr = GaussianProcessRegressor(kernel=kernel, alpha=1e-8, random_state=0)
gpr.fit(X_train, y_train, kind=kind)

# For comparison, misuse the slopes as ordinary function-value observations.
# This fit drives the length scale to its lower bound to try to interpolate
# the slope magnitudes; the warning raised by the optimizer is expected.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", ConvergenceWarning)
    gpr_naive = GaussianProcessRegressor(
        kernel=kernel, alpha=1e-8, random_state=0
    ).fit(X_train, y_train)

X_test = np.linspace(-4, 4, 201)[:, None]
y_mean, y_std = gpr.predict(X_test, return_std=True)
d_kind = np.zeros(len(X_test), dtype=int)
d_mean, d_std = gpr.predict(X_test, kind=d_kind, return_std=True)
y_naive = gpr_naive.predict(X_test)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

ax1.plot(X_test, np.sin(X_test), "k--", label="True function")
ax1.plot(X_test, y_naive, color="tab:orange", alpha=0.6,
         label="Slopes treated as ordinary y")
ax1.plot(X_test, y_mean, color="tab:blue", label="Derivative observations")
ax1.fill_between(
    X_test[:, 0], y_mean - 1.96 * y_std, y_mean + 1.96 * y_std,
    color="tab:blue", alpha=0.2, label="95% confidence interval",
)
ax1.scatter(x_values, y_values, color="tab:blue", zorder=5, label="Values")
ax1.scatter(x_slopes, y_slopes, marker="x", s=80, color="tab:red",
            zorder=5, label="Measured slopes (input as f-values by the naive fit)")
ax1.set_ylabel("f(x)")
ax1.set_title("Function values and slopes of a sine")
ax1.legend(loc="upper left", fontsize=9)

ax2.plot(X_test, np.cos(X_test), "k--", label="True derivative")
ax2.plot(X_test, d_mean, color="tab:green", label="Predicted derivative")
ax2.fill_between(
    X_test[:, 0], d_mean - 1.96 * d_std, d_mean + 1.96 * d_std,
    color="tab:green", alpha=0.2, label="95% confidence interval",
)
ax2.scatter(x_slopes, y_slopes, marker="x", s=80, color="tab:red",
            zorder=5, label="Measured slopes")
ax2.set_xlabel("x")
ax2.set_ylabel("f'(x)")
ax2.legend(loc="upper left", fontsize=9)
_ = fig.suptitle(
    f"Fitted kernel: {gpr.kernel_}", y=0.99,
)
plt.show()
