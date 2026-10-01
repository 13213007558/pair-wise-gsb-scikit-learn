"""Tests for Gaussian process regression with partial-derivative observations."""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process._derivative import (
    FUNCTION_TAG,
    _check_derivative,
    _mixed_covariance,
    _mixed_covariance_gradient,
    _mixed_diag,
    _unpack_constant_rbf,
)
from sklearn.gaussian_process.kernels import (
    RBF,
    ConstantKernel,
    DotProduct,
    WhiteKernel,
)

# A reusable tag convention: -1 marks function-value observations.
F = FUNCTION_TAG


def _finite_difference_block(kernel, x, ix, y, jy, eps=1e-5):
    """Covariance between observation ``ix`` at x and ``jy`` at y via FD."""

    def k(a, b):
        return kernel(a[np.newaxis], b[np.newaxis])[0, 0]

    if ix == F and jy == F:
        return k(x, y)
    if ix == F:
        yp = y.copy()
        yp[jy] += eps
        ym = y.copy()
        ym[jy] -= eps
        return (k(x, yp) - k(x, ym)) / (2 * eps)
    if jy == F:
        xp = x.copy()
        xp[ix] += eps
        xm = x.copy()
        xm[ix] -= eps
        return (k(xp, y) - k(xm, y)) / (2 * eps)

    xp = x.copy()
    xp[ix] += eps
    xm = x.copy()
    xm[ix] -= eps

    def dy(xx):
        yp = y.copy()
        yp[jy] += eps
        ym = y.copy()
        ym[jy] -= eps
        return (k(xx, yp) - k(xx, ym)) / (2 * eps)

    return (dy(xp) - dy(xm)) / (2 * eps)


def _reference_mixed(kernel, X, dx, Y=None, dy=None, eps=1e-5):
    """Finite-difference reference for the full mixed covariance matrix."""
    if Y is None:
        Y, dy = X, dx
    return np.array(
        [
            [
                _finite_difference_block(kernel, X[i], dx[i], Y[j], dy[j], eps)
                for j in range(len(Y))
            ]
            for i in range(len(X))
        ]
    )


@pytest.mark.parametrize("anisotropic", [False, True])
@pytest.mark.parametrize("reversed_order", [False, True])
def test_mixed_covariance_blocks(anisotropic, reversed_order):
    """All four blocks agree with finite differences of the kernel."""
    rng = np.random.RandomState(0)
    n_features = 3
    length_scale = rng.uniform(0.5, 2.0, n_features) if anisotropic else 1.3
    constant = ConstantKernel(1.7)
    rbf = RBF(length_scale)
    kernel = rbf * constant if reversed_order else constant * rbf
    ck, rbf_ = _unpack_constant_rbf(kernel)

    X = rng.randn(9, n_features)
    dx = np.array([F, 0, 1, 2, F, 1, 0, 2, F])

    K = _mixed_covariance(ck, rbf_, X, X, dx, dx)
    K_ref = _reference_mixed(kernel, X, dx)
    assert K.shape == (9, 9)
    assert np.allclose(K, K.T)
    np.testing.assert_allclose(K, K_ref, atol=2e-5)

    # Diagonal: function obs carry sigma**2, derivatives sigma**2 / l_d**2
    diag = _mixed_diag(ck, rbf_, X, dx)
    np.testing.assert_allclose(diag, np.diag(K))

    # Rectangular cross-covariance, mixing function and derivative queries
    Y = rng.randn(5, n_features)
    dy = np.array([F, 2, F, 0, 1])
    Ks = _mixed_covariance(ck, rbf_, Y, X, dy, dx)
    Ks_ref = _reference_mixed(kernel, Y, dy, X, dx)
    np.testing.assert_allclose(Ks, Ks_ref, atol=2e-5)


@pytest.mark.parametrize("anisotropic", [False, True])
@pytest.mark.parametrize("reversed_order", [False, True])
def test_mixed_covariance_gradient(anisotropic, reversed_order):
    """Analytic hyperparameter gradients agree with central differences."""
    rng = np.random.RandomState(0)
    n_features = 3
    length_scale = rng.uniform(0.5, 2.0, n_features) if anisotropic else 1.3
    constant = ConstantKernel(1.7)
    rbf = RBF(length_scale)
    kernel = rbf * constant if reversed_order else constant * rbf
    ck, rbf_ = _unpack_constant_rbf(kernel)

    X = rng.randn(8, n_features)
    dx = np.array([F, 0, 1, 2, 0, 2, 1, F])

    K, K_gradient = _mixed_covariance_gradient(kernel, X, dx)
    assert K_gradient.shape == (8, 8, kernel.n_dims)
    K_ref = _reference_mixed(kernel, X, dx)
    np.testing.assert_allclose(K, K_ref, atol=2e-5)

    theta = kernel.theta
    eye = np.eye(len(theta))
    eps = 1e-6
    for t in range(len(theta)):
        kp = kernel.clone_with_theta(theta + eps * eye[t])
        km = kernel.clone_with_theta(theta - eps * eye[t])
        cp, rp = _unpack_constant_rbf(kp)
        cm, rm = _unpack_constant_rbf(km)
        grad_ref = (
            _mixed_covariance(cp, rp, X, X, dx, dx)
            - _mixed_covariance(cm, rm, X, X, dx, dx)
        ) / (2 * eps)
        np.testing.assert_allclose(K_gradient[..., t], grad_ref, atol=2e-4)


@pytest.mark.parametrize(
    "kernel",
    [
        RBF(),
        ConstantKernel() * DotProduct(),
        ConstantKernel() * RBF() + WhiteKernel(),
    ],
)
def test_unsupported_kernel_raises(kernel):
    X = np.zeros((2, 1))
    y = np.zeros(2)
    derivative = np.array([0, 0])
    gpr = GaussianProcessRegressor(kernel=kernel)
    with pytest.raises(ValueError, match="ConstantKernel and an RBF"):
        gpr.fit(X, y, derivative)


@pytest.mark.parametrize(
    "derivative,expected_message",
    [
        (np.zeros((2, 2)), "shape"),
        (np.array([0, 5]), "neither -1"),
        (np.array([0.0, -1.0]), "integers"),
        (np.array([0]), "shape"),
    ],
)
def test_check_derivative_validation(derivative, expected_message):
    X = np.zeros((2, 3))
    with pytest.raises(ValueError, match=expected_message):
        _check_derivative(derivative, X)


def test_check_derivative_none():
    X = np.zeros((3, 2))
    np.testing.assert_array_equal(_check_derivative(None, X), np.full(3, F))


def test_derivative_prediction_requires_derivative_fit():
    X = np.linspace(0, 1, 5).reshape(-1, 1)
    y = X[:, 0]
    gpr = GaussianProcessRegressor(kernel=ConstantKernel() * RBF(), optimizer=None).fit(
        X, y
    )
    with pytest.raises(ValueError, match="derivative observations"):
        gpr.predict(X, derivative=np.zeros(5, dtype=int))


def test_sine_with_slopes():
    """A few slopes let the GPR recover f and f' on the sine function."""
    rng = np.random.RandomState(0)
    Xf = rng.uniform(0, 2 * np.pi, size=(12, 1))
    Xd = np.array([[0.7], [2.5], [4.5]])
    X = np.vstack([Xf, Xd])
    y = np.concatenate([np.sin(Xf[:, 0]), np.cos(Xd[:, 0])])
    derivative = np.array([F] * 12 + [0] * 3)

    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(2.0, (1e-2, 1e2))
    gpr = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-8,
        derivative_alpha=1e-8,
        n_restarts_optimizer=2,
        random_state=0,
    ).fit(X, y, derivative)

    X_test = np.linspace(0, 2 * np.pi, 40).reshape(-1, 1)
    f_mean, f_std = gpr.predict(X_test, return_std=True)
    df_mean, df_std = gpr.predict(
        X_test, return_std=True, derivative=np.zeros(40, dtype=int)
    )

    np.testing.assert_allclose(f_mean, np.sin(X_test[:, 0]), atol=0.1)
    np.testing.assert_allclose(df_mean, np.cos(X_test[:, 0]), atol=0.2)
    assert np.all(f_std >= 0)
    assert np.all(df_std >= 0)


def test_slopes_as_plain_samples_fail():
    """Treating slopes as ordinary y samples does not reproduce f'."""
    rng = np.random.RandomState(0)
    Xf = rng.uniform(0, 2 * np.pi, size=(12, 1))
    Xd = np.array([[0.7], [2.5], [4.5]])
    X = np.vstack([Xf, Xd])
    y = np.concatenate([np.sin(Xf[:, 0]), np.cos(Xd[:, 0])])

    # Slopes handed over as if they were function values.
    gpr_naive = GaussianProcessRegressor(
        kernel=ConstantKernel(1.0) * RBF(2.0),
        alpha=1e-8,
        optimizer=None,
    ).fit(X, y)

    h = 1e-5
    slope_estimate = (gpr_naive.predict(Xd + h) - gpr_naive.predict(Xd - h)) / (2 * h)
    # The naive model distorts the surface around the slope points and
    # cannot recover the true derivatives.
    assert np.max(np.abs(slope_estimate - np.cos(Xd[:, 0]))) > 0.5

    # Whereas the derivative-aware model fits both.
    derivative = np.array([F] * 12 + [0] * 3)
    gpr = GaussianProcessRegressor(
        kernel=ConstantKernel(1.0, (1e-3, 1e3)) * RBF(2.0, (1e-2, 1e2)),
        alpha=1e-8,
        derivative_alpha=1e-8,
        n_restarts_optimizer=2,
        random_state=0,
    ).fit(X, y, derivative)
    np.testing.assert_allclose(
        gpr.predict(Xd, derivative=np.zeros(3, dtype=int)),
        np.cos(Xd[:, 0]),
        atol=0.15,
    )


def test_duplicate_locations():
    """Function value and derivative may be observed at the same point."""
    X = np.array([[0.0], [1.0], [1.0], [2.0]])
    y = np.array([0.0, 1.0, 0.5, 0.0])
    derivative = np.array([F, F, 0, F])

    gpr = GaussianProcessRegressor(
        kernel=ConstantKernel(1.0) * RBF(0.8),
        alpha=1e-10,
        derivative_alpha=1e-10,
        optimizer=None,
    ).fit(X, y, derivative)

    mean, cov = gpr.predict(X[[0, 1]], return_cov=True, derivative=np.array([F, 0]))
    assert np.all(np.isfinite(mean))
    assert np.all(np.linalg.eigvalsh(cov) > -1e-8)

    # Exactly repeated function-value observations collapse their
    # predictive variance as noise tends to zero.
    X_rep = np.array([[0.5], [0.5]])
    y_rep = np.array([0.3, 0.3])
    gpr_rep = GaussianProcessRegressor(
        kernel=ConstantKernel() * RBF(),
        alpha=1e-12,
        optimizer=None,
    ).fit(X_rep, y_rep)
    _, std = gpr_rep.predict(X_rep[:1], return_std=True)
    assert std[0] < 1e-3


def test_separate_noise_levels():
    """Function and derivative observations use independent noise."""
    X = np.array([[0.0], [1.0], [1.0]])
    y = np.array([0.0, 1.0, 0.5])
    derivative = np.array([F, F, 0])

    gpr = GaussianProcessRegressor(
        kernel=ConstantKernel() * RBF(1.0),
        alpha=1e-2,
        derivative_alpha=1.0,
        optimizer=None,
    ).fit(X, y, derivative)

    # The derivative observation must land on the diagonal with its own
    # noise: rebuild the Cholesky factor argument and check the diagonal.
    assert gpr.alpha_diag_[0] == pytest.approx(1e-2)
    assert gpr.alpha_diag_[2] == pytest.approx(1.0)

    # Array-valued noise levels are also accepted, one entry per kind.
    gpr_array = GaussianProcessRegressor(
        kernel=ConstantKernel() * RBF(1.0),
        alpha=np.array([1e-2, 1e-3]),
        derivative_alpha=np.array([2.0]),
        optimizer=None,
    ).fit(X, y, derivative)
    np.testing.assert_allclose(gpr_array.alpha_diag_, [1e-2, 1e-3, 2.0])

    with pytest.raises(ValueError, match="one entry per"):
        GaussianProcessRegressor(
            kernel=ConstantKernel() * RBF(1.0),
            derivative_alpha=np.array([1.0, 2.0]),
            optimizer=None,
        ).fit(X, y, derivative)

    # A length-1 array broadcasts to all observations of the given kind.
    gpr_broadcast = GaussianProcessRegressor(
        kernel=ConstantKernel() * RBF(1.0),
        alpha=np.array([1e-2]),
        derivative_alpha=np.array([2.0]),
        optimizer=None,
    ).fit(X, y, derivative)
    np.testing.assert_allclose(gpr_broadcast.alpha_diag_, [1e-2, 1e-2, 2.0])


def test_mixed_prediction_joint_covariance():
    """Joint function/derivative predictions use the mixed covariance."""
    rng = np.random.RandomState(2)
    X = rng.uniform(-1, 1, size=(8, 2))
    y = X[:, 0] ** 2 + X[:, 1]
    Xd = rng.uniform(-1, 1, size=(4, 2))
    yd = np.zeros(4)  # derivative along feature 0 at origin-like constraint
    derivative_y = np.zeros(4, dtype=int)

    X_all = np.vstack([X, Xd])
    y_all = np.concatenate([y, yd])
    tags = np.array([F] * 8 + [0] * 4)

    gpr = GaussianProcessRegressor(
        kernel=ConstantKernel(1.0, (1e-3, 1e3)) * RBF([0.8, 0.8], (1e-2, 1e2)),
        alpha=1e-7,
        derivative_alpha=1e-7,
        optimizer=None,
    ).fit(X_all, y_all, tags)

    X_test = rng.uniform(-1, 1, size=(6, 2))
    kinds = np.array([F, 0, 1, F, 0, 1])
    mean, cov = gpr.predict(X_test, return_cov=True, derivative=kinds)
    assert cov.shape == (6, 6)
    assert np.allclose(cov, cov.T)
    assert np.all(np.linalg.eigvalsh(cov) > -1e-8)

    # return_std diagonal agrees with the joint covariance
    _, std = gpr.predict(X_test, return_std=True, derivative=kinds)
    np.testing.assert_allclose(np.diag(cov), std**2, atol=1e-10)


def test_normalize_y_with_derivatives():
    """normalize_y centers values but does not subtract a mean from slopes."""
    rng = np.random.RandomState(3)
    offset, scale = 5.0, 2.0
    Xf = rng.uniform(0, 1, size=(10, 1))
    yf = offset + scale * Xf[:, 0]
    Xd = np.array([[0.2], [0.8]])
    yd = np.full(2, scale)
    X = np.vstack([Xf, Xd])
    y = np.concatenate([yf, yd])
    tags = np.array([F] * 10 + [0] * 2)

    gpr = GaussianProcessRegressor(
        kernel=ConstantKernel(1.0, (1e-3, 1e3)) * RBF(0.5, (1e-2, 1e2)),
        normalize_y=True,
        alpha=1e-7,
        derivative_alpha=1e-7,
        n_restarts_optimizer=1,
        random_state=0,
    ).fit(X, y, tags)

    X_test = np.linspace(0, 1, 5).reshape(-1, 1)
    f_mean = gpr.predict(X_test)
    df_mean = gpr.predict(X_test, derivative=np.zeros(5, dtype=int))
    np.testing.assert_allclose(f_mean, offset + scale * X_test[:, 0], atol=0.1)
    np.testing.assert_allclose(df_mean, np.full(5, scale), atol=0.2)


def test_log_marginal_likelihood_gradient():
    """The LML gradient through the mixed blocks matches finite differences."""
    rng = np.random.RandomState(4)
    Xf = rng.uniform(0, 2, size=(5, 1))
    Xd = rng.uniform(0, 2, size=(3, 1))
    X = np.vstack([Xf, Xd])
    y = np.concatenate([Xf[:, 0] ** 2, 2 * Xd[:, 0]])
    tags = np.array([F] * 5 + [0] * 3)

    gpr = GaussianProcessRegressor(
        kernel=ConstantKernel(1.2) * RBF(0.7),
        alpha=1e-8,
        derivative_alpha=1e-8,
        optimizer=None,
    ).fit(X, y, tags)

    theta = gpr.kernel_.theta
    _, grad = gpr.log_marginal_likelihood(theta, eval_gradient=True)
    eps = 1e-6
    eye = np.eye(len(theta))
    grad_ref = np.empty_like(grad)
    for t in range(len(theta)):
        lp = gpr.log_marginal_likelihood(theta + eps * eye[t])
        lm = gpr.log_marginal_likelihood(theta - eps * eye[t])
        grad_ref[t] = (lp - lm) / (2 * eps)
    np.testing.assert_allclose(grad, grad_ref, atol=2e-3)


def test_multioutput_derivatives():
    rng = np.random.RandomState(5)
    Xf = rng.uniform(-1, 1, size=(8, 1))
    Yf = np.column_stack([Xf[:, 0], 3 * Xf[:, 0]])
    Xd = np.array([[0.0], [0.5]])
    Yd = np.array([[1.0, 3.0], [1.0, 3.0]])
    X = np.vstack([Xf, Xd])
    Y = np.vstack([Yf, Yd])
    tags = np.array([F] * 8 + [0, 0])

    gpr = GaussianProcessRegressor(
        kernel=ConstantKernel() * RBF(0.6),
        alpha=1e-7,
        derivative_alpha=1e-7,
        optimizer=None,
    ).fit(X, Y, tags)
    slope = gpr.predict(Xd, derivative=np.zeros(2, dtype=int))
    np.testing.assert_allclose(slope, [[1, 3], [1, 3]], atol=1e-4)


def test_sample_y_with_derivatives():
    X = np.linspace(0, 2, 6).reshape(-1, 1)
    y = X[:, 0] ** 2
    Xd = np.array([[1.0]])
    X_all = np.vstack([X, Xd])
    y_all = np.concatenate([y, [2.0]])
    tags = np.array([F] * 6 + [0])

    gpr = GaussianProcessRegressor(
        kernel=ConstantKernel() * RBF(0.7),
        alpha=1e-8,
        derivative_alpha=1e-8,
        optimizer=None,
    ).fit(X_all, y_all, tags)

    X_test = np.linspace(0, 2, 4).reshape(-1, 1)
    samples = gpr.sample_y(
        X_test,
        n_samples=5,
        random_state=0,
        derivative=np.zeros(4, dtype=int),
    )
    assert samples.shape == (4, 5)
    assert np.all(np.isfinite(samples))


def test_anisotropic_length_scale_dimension():
    """Anisotropic length scales must match the number of features."""
    X = np.zeros((4, 3))
    y = np.zeros(4)
    derivative = np.array([0, 1, 2, 0])
    gpr = GaussianProcessRegressor(
        kernel=ConstantKernel() * RBF([1.0, 2.0]), optimizer=None
    )
    with pytest.raises(ValueError, match="as many length scales"):
        gpr.fit(X, y, derivative)


def test_derivative_only_observations():
    """A model may be fit on derivative observations alone."""
    X = np.linspace(0, 1, 5).reshape(-1, 1)
    y = np.full(5, 2.0)  # df/dx = 2 everywhere
    tags = np.zeros(5, dtype=int)

    gpr = GaussianProcessRegressor(
        kernel=ConstantKernel(1.0) * RBF(0.4),
        derivative_alpha=1e-8,
        optimizer=None,
    ).fit(X, y, tags)
    X_test = np.linspace(0, 1, 3).reshape(-1, 1)
    slope = gpr.predict(X_test, derivative=np.zeros(3, dtype=int))
    np.testing.assert_allclose(slope, 2.0, atol=0.1)

    # Function values are only identified up to a constant; predictions
    # remain finite.
    assert np.all(np.isfinite(gpr.predict(X_test)))


def test_backward_compatible_fit_without_derivative():
    """Existing calls without the derivative argument are unchanged."""
    X = np.linspace(0, 1, 6).reshape(-1, 1)
    y = np.sin(2 * np.pi * X[:, 0])
    gpr_a = GaussianProcessRegressor(
        kernel=ConstantKernel(1.5) * RBF(0.3),
        alpha=1e-8,
        optimizer=None,
    ).fit(X, y)
    gpr_b = GaussianProcessRegressor(
        kernel=ConstantKernel(1.5) * RBF(0.3),
        alpha=1e-8,
        optimizer=None,
    ).fit(X, y, np.full(6, F))
    np.testing.assert_allclose(gpr_a.predict(X), gpr_b.predict(X), atol=1e-10)
