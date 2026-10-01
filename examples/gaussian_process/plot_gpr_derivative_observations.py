"""
=========================================================================
Gaussian process regression with derivative observations
=========================================================================

Gaussian processes can be conditioned on measured slopes in addition to
function values. Each observation is tagged as a value (``-1``) or as a
partial derivative (feature index), using a constant kernel times an RBF.
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import matplotlib.pyplot as plt
import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

# %%
# We reconstruct ``f(x) = sin(x)`` on ``[0, 2 * pi]`` from a handful of
# function values together with a few measured slopes ``f'(x) = cos(x)``.
rng = np.random.RandomState(1)

x_values = rng.uniform(0, 2 * np.pi, size=7).reshape(-1, 1)
x_slopes = np.array([[0.6], [2.8], [5.2]])

X_train = np.vstack([x_values, x_slopes])
y_train = np.concatenate([np.sin(x_values[:, 0]), np.cos(x_slopes[:, 0])])

# -1 marks function-value observations, 0 marks d/dx_0 observations.
derivative = np.array([-1] * len(x_values) + [0] * len(x_slopes))

kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(2.0, (1e-2, 1e2))
gpr = GaussianProcessRegressor(
    kernel=kernel,
    alpha=1e-4,
    derivative_alpha=1e-4,
    n_restarts_optimizer=5,
    random_state=0,
).fit(X_train, y_train, derivative)

x = np.linspace(0, 2 * np.pi, 200).reshape(-1, 1)
mean, std = gpr.predict(x, return_std=True)
slope_mean, slope_std = gpr.predict(
    x, return_std=True, derivative=np.zeros(len(x), dtype=int)
)

# %%
# Both the function and its derivative are recovered, together with their
# uncertainty.
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))

ax1.plot(x, np.sin(x), "k--", label="true f(x) = sin(x)")
ax1.plot(x, mean, label="posterior mean")
ax1.fill_between(
    x[:, 0],
    mean - 1.96 * std,
    mean + 1.96 * std,
    alpha=0.3,
    label="95% confidence interval",
)
ax1.scatter(
    x_values[:, 0],
    np.sin(x_values[:, 0]),
    c="k",
    zorder=5,
    label="function observations",
)
ax1.set_ylabel("f(x)")
ax1.legend(loc="upper right")

ax2.plot(x, np.cos(x), "k--", label="true f'(x) = cos(x)")
ax2.plot(x, slope_mean, label="posterior mean of f'")
ax2.fill_between(
    x[:, 0],
    slope_mean - 1.96 * slope_std,
    slope_mean + 1.96 * slope_std,
    alpha=0.3,
    label="95% confidence interval",
)
ax2.scatter(
    x_slopes[:, 0],
    np.cos(x_slopes[:, 0]),
    c="tab:red",
    marker="^",
    zorder=5,
    label="derivative observations",
)
ax2.set_xlabel("x")
ax2.set_ylabel("f'(x)")
ax2.legend(loc="upper right")
_ = fig.suptitle("GPR with function-value and derivative observations")

# %%
# What happens if the slopes are treated as ordinary y samples? The model
# tries to interpolate the slope values as if they were function values and
# produces a distorted surface, in particular near the slope points.
gpr_naive = GaussianProcessRegressor(
    kernel=ConstantKernel(1.0) * RBF(2.0), alpha=1e-8, optimizer=None
).fit(X_train, y_train)

fig2, ax = plt.subplots(figsize=(8, 4))
ax.plot(x, np.sin(x), "k--", label="true f(x) = sin(x)")
ax.plot(x, gpr_naive.predict(x), label="slopes treated as plain y samples")
ax.scatter(
    x_values[:, 0],
    np.sin(x_values[:, 0]),
    c="k",
    zorder=5,
    label="function observations",
)
ax.scatter(
    x_slopes[:, 0],
    np.cos(x_slopes[:, 0]),
    c="tab:red",
    marker="^",
    zorder=5,
    label="slope observations (misinterpreted as values)",
)
ax.set_xlabel("x")
ax.set_ylabel("f(x)")
ax.legend(loc="upper right")
_ = ax.set_title("Derivative observations cannot be used as ordinary targets")

# %%
# Predicting values and derivatives jointly yields their cross-covariance,
# which is required for instance when propagating uncertainty from both
# kinds of predictions.
X_query = np.array([[1.0], [2.0], [3.0]])
kinds = np.array([-1, 0, -1])
joint_mean, joint_cov = gpr.predict(X_query, return_cov=True, derivative=kinds)
print("Joint query rows: f(1.0), f'(2.0), f(3.0)")
print(np.round(joint_mean, 3))
print("Joint predictive covariance:")
print(np.round(joint_cov, 4))

plt.show()
