"""Gaussian processes regression."""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import warnings
from numbers import Integral, Real
from operator import itemgetter

import numpy as np
import scipy.optimize
from scipy.linalg import cho_solve, cholesky, solve_triangular

from sklearn.base import (
    BaseEstimator,
    MultiOutputMixin,
    RegressorMixin,
    _fit_context,
    clone,
)
from sklearn.gaussian_process.kernels import RBF, Kernel, Product
from sklearn.gaussian_process.kernels import ConstantKernel as C
from sklearn.preprocessing._data import _handle_zeros_in_scale
from sklearn.utils import check_random_state
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.optimize import _check_optimize_result
from sklearn.utils.validation import validate_data

GPR_CHOLESKY_LOWER = True

# Sentinel used in the `kind` array to mark an observation of the function
# value. Entries ``j >= 0`` mark an observation of the partial derivative
# with respect to feature ``j``.
_FUNCTION_OBSERVATION = -1


def _check_obs_kind(kind, n_samples, n_features, name):
    """Validate the array tagging observations as values or derivatives.

    Entries equal to -1 denote function values while entries ``j`` in
    ``[0, n_features)`` denote observations of the partial derivative with
    respect to feature ``j``. ``n_features`` may be None for non-vector
    inputs, in which case only function values (-1) are allowed.
    """
    if kind is None:
        return np.full(n_samples, _FUNCTION_OBSERVATION, dtype=np.intp)
    kind = np.asarray(kind, dtype=np.intp)
    if kind.ndim != 1 or kind.shape[0] != n_samples:
        raise ValueError(
            f"`{name}` must be a 1D array with {n_samples} entries, one per "
            f"observation; got shape {kind.shape}."
        )
    if n_features is not None:
        valid = (kind >= _FUNCTION_OBSERVATION) & (kind < n_features)
        if not np.all(valid):
            raise ValueError(
                f"`{name}` entries must be -1 for a function value or an "
                "integer in [0, n_features) indexing a feature; got "
                f"{kind.tolist()} for {n_features} feature(s)."
            )
    elif np.any(kind >= 0):
        raise ValueError(
            f"`{name}` derivative observations require vector-valued "
            "inputs."
        )
    return kind


def _unpack_constant_rbf_kernel(kernel):
    """Return ``(constant_kernel, rbf_kernel)`` for a ``C() * RBF()`` kernel.

    Returns ``None`` if ``kernel`` is not a product of exactly one
    :class:`ConstantKernel` and one :class:`RBF` kernel (subclasses such as
    :class:`Matern` are not accepted by the first derivative-aware version).
    """
    if not isinstance(kernel, Product):
        return None
    k1, k2 = kernel.k1, kernel.k2
    if type(k1) is C and type(k2) is RBF:
        return k1, k2
    if type(k2) is C and type(k1) is RBF:
        return k2, k1
    return None


def _rbf_derivative_cov(
    constant_value, length_scale, X, kind_x, Y, kind_y, eval_gradient=False
):
    """Covariance between function/derivative observations for ``C * RBF``.

    Each row of ``X`` (resp. ``Y``) is tagged through ``kind_x`` (resp.
    ``kind_y``): -1 for a function value and ``j >= 0`` for a partial
    derivative with respect to feature ``j``. The four blocks of the joint
    covariance follow by differentiating ``k(x, y) = C * RBF(x, y)``. With
    ``u = (x - y) / l**2``, ``h = (x - y)^2 / l**2`` and
    :math:`\\partial_{x_i}\\partial_{y_j}` the mixed derivative::

        d2k/dx_i dy_j = C * exp(-h/2) * (delta_ij / l_i**2 - u_i u_j)

    while ``dk/dy_j = +C * exp(-h/2) * u_j`` and
    ``dk/dx_i = -C * exp(-h/2) * u_i``.

    The optional gradient is given with respect to the log of the constant
    value and of each length scale.
    """
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    Y = np.atleast_2d(np.asarray(Y, dtype=np.float64))
    n_samples, n_features = X.shape
    n_samples_y = Y.shape[0]
    ls = np.asarray(length_scale, dtype=np.float64).ravel()
    anisotropic = ls.size > 1
    length_scale = (
        ls.copy()
        if anisotropic
        else np.broadcast_to(ls[0], (n_features,)).copy()
    )

    kind_x = np.asarray(kind_x, dtype=np.intp)
    kind_y = np.asarray(kind_y, dtype=np.intp)

    length_scale_sq = length_scale**2
    # diff[i, j, q] = X[i, q] - Y[j, q]
    diff = X[:, np.newaxis, :] - Y[np.newaxis, :, :]
    # u[i, j, q] = diff_q / l_q**2, with h = sum(diff * u)
    u = diff / length_scale_sq
    h = np.sum(diff * u, axis=-1)
    k0 = float(constant_value) * np.exp(-0.5 * h)

    is_value_x = (kind_x == _FUNCTION_OBSERVATION)[:, np.newaxis]
    is_value_y = (kind_y == _FUNCTION_OBSERVATION)[np.newaxis, :]
    is_deriv_x = ~is_value_x
    is_deriv_y = ~is_value_y

    # Component of u along the derivative dimension of each row / column;
    # this is zero wherever the corresponding observation is a value.
    row_index = np.arange(n_samples)[:, np.newaxis]
    col_index = np.arange(n_samples_y)[np.newaxis, :]
    u_x = u[row_index, col_index, np.clip(kind_x, 0, n_features - 1)[:, np.newaxis]]
    u_x = np.where(is_value_x, 0.0, u_x)
    u_y = u[row_index, col_index, np.clip(kind_y, 0, n_features - 1)[np.newaxis, :]]
    u_y = np.where(is_value_y, 0.0, u_y)
    # Length scale squared along the row derivative dimension.
    l_sq_x = length_scale_sq[np.clip(kind_x, 0, n_features - 1)][:, np.newaxis]

    same_dim = (
        (kind_x[:, np.newaxis] == kind_y[np.newaxis, :])
        & is_deriv_x
        & is_deriv_y
    )

    mask_vv = is_value_x & is_value_y
    mask_vd = is_value_x & is_deriv_y
    mask_dv = is_deriv_x & is_value_y
    mask_dd = is_deriv_x & is_deriv_y

    K = np.zeros((n_samples, n_samples_y))
    with np.errstate(invalid="ignore"):
        K += mask_vv * k0
        # dk/dy_j = +C * exp(-h/2) * u_j  (u = (x - y)/l**2)
        K += mask_vd * (k0 * u_y)
        # dk/dx_i = -C * exp(-h/2) * u_i
        K += mask_dv * (-k0 * u_x)
        # d2k/dx_i dy_j = C * exp(-h/2) * (delta_ij / l_i**2 - u_i u_j)
        K += mask_dd * k0 * (same_dim / l_sq_x - u_x * u_y)

    if not eval_gradient:
        return K

    # Gradient with respect to log(C): every block scales linearly with C,
    # so dK/d(log C) = C * dK/dC = K.
    grad_constant = K[..., np.newaxis]

    # s_q = diff_q**2 / l_q**2, the gradient of the exponent wrt log(l_q).
    s = diff**2 / length_scale_sq
    grad_length = K[..., np.newaxis] * s

    row_dim = (kind_x[:, np.newaxis] == np.arange(n_features))[:, np.newaxis, :]
    col_dim = (kind_y[:, np.newaxis] == np.arange(n_features))[np.newaxis, :, :]

    # value/derivative block: -2 C exp(-h/2) u_j when q == j
    grad_length += mask_vd[..., np.newaxis] * (
        -2.0 * k0[..., np.newaxis] * u_y[..., np.newaxis] * col_dim
    )
    # derivative/value block: +2 C exp(-h/2) u_i when q == i
    grad_length += mask_dv[..., np.newaxis] * (
        2.0 * k0[..., np.newaxis] * u_x[..., np.newaxis] * row_dim
    )
    # derivative/derivative block corrections. The two indicators are added
    # (not OR-ed): for same-dimension blocks both contribute.
    product = k0 * u_x * u_y
    grad_length += mask_dd[..., np.newaxis] * (
        2.0
        * product[..., np.newaxis]
        * (row_dim.astype(np.float64) + col_dim.astype(np.float64))
    )
    grad_length += mask_dd[..., np.newaxis] * (
        -2.0
        * k0[..., np.newaxis]
        * (same_dim.astype(np.float64) / l_sq_x)[..., np.newaxis]
        * row_dim
    )

    if not anisotropic:
        # A single scalar length scale is shared by all dimensions.
        grad_length = grad_length.sum(axis=-1, keepdims=True)

    K_gradient = np.concatenate([grad_constant, grad_length], axis=-1)
    return K, K_gradient


class GaussianProcessRegressor(MultiOutputMixin, RegressorMixin, BaseEstimator):
    """Gaussian process regression (GPR).

    The implementation is based on Algorithm 2.1 of [RW2006]_.

    In addition to standard scikit-learn estimator API,
    :class:`GaussianProcessRegressor`:

    * allows prediction without prior fitting (based on the GP prior)
    * provides an additional method `sample_y(X)`, which evaluates samples
      drawn from the GPR (prior or posterior) at given inputs
    * exposes a method `log_marginal_likelihood(theta)`, which can be used
      externally for other ways of selecting hyperparameters, e.g., via
      Markov chain Monte Carlo.

    To learn the difference between a point-estimate approach vs. a more
    Bayesian modelling approach, refer to the example entitled
    :ref:`sphx_glr_auto_examples_gaussian_process_plot_compare_gpr_krr.py`.

    Read more in the :ref:`User Guide <gaussian_process>`.

    .. versionadded:: 0.18

    Parameters
    ----------
    kernel : kernel instance, default=None
        The kernel specifying the covariance function of the GP.
        If `None` is passed,
        the kernel `ConstantKernel() * RBF()` is used as default.
        Note that
        the kernel hyperparameters are optimized during fitting
        unless the bounds are marked as `"fixed"`
        or the argument `optimizer` is set to `None`.

    alpha : float or ndarray of shape (n_samples,), default=1e-10
        Value added to the diagonal of the kernel matrix during fitting.
        This can prevent a potential numerical issue during fitting, by
        ensuring that the calculated values form a positive definite matrix.
        It can also be interpreted as the variance of additional Gaussian
        measurement noise on the training observations. Note that this is
        different from using a `WhiteKernel`. If an array is passed, it must
        have the same number of entries as the data used for fitting and is
        used as datapoint-dependent noise level. Allowing to specify the
        noise level directly as a parameter is mainly for convenience and
        for consistency with :class:`~sklearn.linear_model.Ridge`.
        For an example illustrating how the alpha parameter controls
        the noise variance in Gaussian Process Regression, see
        :ref:`sphx_glr_auto_examples_gaussian_process_plot_gpr_noisy_targets.py`.

    optimizer : "fmin_l_bfgs_b", callable or None, default="fmin_l_bfgs_b"
        Can either be one of the internally supported optimizers for optimizing
        the kernel's parameters, specified by a string, or an externally
        defined optimizer passed as a callable. If a callable is passed, it
        must have the signature::

            def optimizer(obj_func, initial_theta, bounds):
                # * 'obj_func': the objective function to be minimized, which
                #   takes the hyperparameters theta as a parameter and an
                #   optional flag eval_gradient, which determines if the
                #   gradient is returned additionally to the function value
                # * 'initial_theta': the initial value for theta, which can be
                #   used by local optimizers
                # * 'bounds': the bounds on the values of theta
                ....
                # Returned are the best found hyperparameters theta and
                # the corresponding value of the target function.
                return theta_opt, func_min

        Per default, the L-BFGS-B algorithm from `scipy.optimize.minimize`
        is used. If None is passed, the kernel's parameters are kept fixed.
        Available internal optimizers are: `{'fmin_l_bfgs_b'}`.

    n_restarts_optimizer : int, default=0
        The number of restarts of the optimizer for finding the kernel's
        parameters which maximize the log-marginal likelihood. The first run
        of the optimizer is performed from the kernel's initial parameters,
        the remaining ones (if any) from thetas sampled log-uniform randomly
        from the space of allowed theta-values. If greater than 0, all bounds
        must be finite. Note that `n_restarts_optimizer == 0` implies that one
        run is performed.

    normalize_y : bool, default=False
        Whether or not to normalize the target values `y` by removing the mean
        and scaling to unit-variance. This is recommended for cases where
        zero-mean, unit-variance priors are used. Note that, in this
        implementation, the normalisation is reversed before the GP predictions
        are reported.

        .. versionchanged:: 0.23

    copy_X_train : bool, default=True
        If True, a persistent copy of the training data is stored in the
        object. Otherwise, just a reference to the training data is stored,
        which might cause predictions to change if the data is modified
        externally.

    n_targets : int, default=None
        The number of dimensions of the target values. Used to decide the number
        of outputs when sampling from the prior distributions (i.e. calling
        :meth:`sample_y` before :meth:`fit`). This parameter is ignored once
        :meth:`fit` has been called.

        .. versionadded:: 1.3

    random_state : int, RandomState instance or None, default=None
        Determines random number generation used to initialize the centers.
        Pass an int for reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.

    Attributes
    ----------
    X_train_ : array-like of shape (n_samples, n_features) or list of object
        Feature vectors or other representations of training data (also
        required for prediction).

    y_train_ : array-like of shape (n_samples,) or (n_samples, n_targets)
        Target values in training data (also required for prediction).

    obs_kind_ : ndarray of shape (n_samples,)
        Tag of each training observation. Entries are -1 for a function
        value observation and ``j >= 0`` for a partial derivative with
        respect to feature ``j``. Defined after :meth:`fit`.

        .. versionadded:: 1.10

    kernel_ : kernel instance
        The kernel used for prediction. The structure of the kernel is the
        same as the one passed as parameter but with optimized hyperparameters.

    L_ : array-like of shape (n_samples, n_samples)
        Lower-triangular Cholesky decomposition of the kernel in ``X_train_``.

    alpha_ : array-like of shape (n_samples,)
        Dual coefficients of training data points in kernel space.

    log_marginal_likelihood_value_ : float
        The log-marginal-likelihood of ``self.kernel_.theta``.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    GaussianProcessClassifier : Gaussian process classification (GPC)
        based on Laplace approximation.

    References
    ----------
    .. [RW2006] `Carl E. Rasmussen and Christopher K.I. Williams,
       "Gaussian Processes for Machine Learning",
       MIT Press 2006 <https://www.gaussianprocess.org/gpml/chapters/RW.pdf>`_

    Examples
    --------
    >>> from sklearn.datasets import make_friedman2
    >>> from sklearn.gaussian_process import GaussianProcessRegressor
    >>> from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
    >>> X, y = make_friedman2(n_samples=500, noise=0, random_state=0)
    >>> kernel = DotProduct() + WhiteKernel()
    >>> gpr = GaussianProcessRegressor(kernel=kernel,
    ...         random_state=0).fit(X, y)
    >>> gpr.score(X, y)
    0.3680...
    >>> gpr.predict(X[:2,:], return_std=True)
    (array([653.0, 592.1]), array([316.6, 316.6]))

    Notes
    -----
    When derivative observations are provided through the ``kind`` argument of
    :meth:`fit` (partial derivatives measured alongside function values), the
    kernel is restricted to an isotropic or anisotropic
    :class:`~sklearn.gaussian_process.kernels.RBF` multiplied by a
    :class:`~sklearn.gaussian_process.kernels.ConstantKernel`. The joint
    covariance of values and derivatives, the hyperparameter optimization and
    the predictive covariance all take the derivative observations into
    account. Partial derivatives can also be requested at prediction time
    through the ``kind`` argument of :meth:`predict`.
    """

    _parameter_constraints: dict = {
        "kernel": [None, Kernel],
        "alpha": [Interval(Real, 0, None, closed="left"), np.ndarray],
        "optimizer": [StrOptions({"fmin_l_bfgs_b"}), callable, None],
        "n_restarts_optimizer": [Interval(Integral, 0, None, closed="left")],
        "normalize_y": ["boolean"],
        "copy_X_train": ["boolean"],
        "n_targets": [Interval(Integral, 1, None, closed="left"), None],
        "random_state": ["random_state"],
    }

    def __init__(
        self,
        kernel=None,
        *,
        alpha=1e-10,
        optimizer="fmin_l_bfgs_b",
        n_restarts_optimizer=0,
        normalize_y=False,
        copy_X_train=True,
        n_targets=None,
        random_state=None,
    ):
        self.kernel = kernel
        self.alpha = alpha
        self.optimizer = optimizer
        self.n_restarts_optimizer = n_restarts_optimizer
        self.normalize_y = normalize_y
        self.copy_X_train = copy_X_train
        self.n_targets = n_targets
        self.random_state = random_state

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y, *, kind=None):
        """Fit Gaussian process regression model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or list of object
            Feature vectors or other representations of training data.

        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values. When a row of ``kind`` marks a derivative, the
            corresponding ``y`` entry is the measured partial derivative.

        kind : array-like of shape (n_samples,), default=None
            Tags indicating whether each observation is a function value or a
            partial derivative. Entries are:

            - ``-1`` (or ``None`` for all rows): the observed function value
              ``f(x)``,
            - ``j >= 0``: the observed partial derivative
              :math:`\\partial f / \\partial x_j` at the corresponding
              position.

            Mixing function values and derivatives requires the kernel to be
            an isotropic or anisotropic
            :class:`~sklearn.gaussian_process.kernels.RBF` multiplied by a
            :class:`~sklearn.gaussian_process.kernels.ConstantKernel`.
            Per-observation noise for the two kinds of measurements can be
            set independently through the ``alpha`` array.

            .. versionadded:: 1.10

        Returns
        -------
        self : object
            GaussianProcessRegressor class instance.
        """
        self.kernel_ = C() * RBF() if self.kernel is None else clone(self.kernel)

        self._rng = check_random_state(self.random_state)

        if self.kernel_.requires_vector_input:
            dtype, ensure_2d = "numeric", True
        else:
            dtype, ensure_2d = None, False
        X, y = validate_data(
            self,
            X,
            y,
            multi_output=True,
            y_numeric=True,
            ensure_2d=ensure_2d,
            dtype=dtype,
        )

        n_targets_seen = y.shape[1] if y.ndim > 1 else 1
        if self.n_targets is not None and n_targets_seen != self.n_targets:
            raise ValueError(
                "The number of targets seen in `y` is different from the parameter "
                f"`n_targets`. Got {n_targets_seen} != {self.n_targets}."
            )

        n_features_in = X.shape[1] if X.ndim == 2 else None
        kind = _check_obs_kind(kind, X.shape[0], n_features_in, "kind")
        self._has_derivative_observations = bool(np.any(kind >= 0))

        if self._has_derivative_observations:
            parts = _unpack_constant_rbf_kernel(self.kernel_)
            if parts is None:
                raise ValueError(
                    "Fitting with derivative observations (`kind` contains "
                    "entries >= 0) requires a kernel that is a product of a "
                    "ConstantKernel and an isotropic or anisotropic RBF, "
                    f"got {self.kernel_!r}. Subclasses of RBF such as Matern "
                    "are not supported."
                )
            return self._fit_derivative_observations(parts, X, y, kind)
        self.obs_kind_ = kind

        # Normalize target value
        if self.normalize_y:
            self._y_train_mean = np.mean(y, axis=0)
            self._y_train_std = _handle_zeros_in_scale(np.std(y, axis=0), copy=False)

            # Remove mean and make unit variance
            y = (y - self._y_train_mean) / self._y_train_std

        else:
            shape_y_stats = (y.shape[1],) if y.ndim == 2 else 1
            self._y_train_mean = np.zeros(shape=shape_y_stats)
            self._y_train_std = np.ones(shape=shape_y_stats)

        if np.iterable(self.alpha) and self.alpha.shape[0] != y.shape[0]:
            if self.alpha.shape[0] == 1:
                self.alpha = self.alpha[0]
            else:
                raise ValueError(
                    "alpha must be a scalar or an array with same number of "
                    f"entries as y. ({self.alpha.shape[0]} != {y.shape[0]})"
                )

        self.X_train_ = np.copy(X) if self.copy_X_train else X
        self.y_train_ = np.copy(y) if self.copy_X_train else y

        if self.optimizer is not None and self.kernel_.n_dims > 0:
            # Choose hyperparameters based on maximizing the log-marginal
            # likelihood (potentially starting from several initial values)
            def obj_func(theta, eval_gradient=True):
                if eval_gradient:
                    lml, grad = self.log_marginal_likelihood(
                        theta, eval_gradient=True, clone_kernel=False
                    )
                    return -lml, -grad
                else:
                    return -self.log_marginal_likelihood(theta, clone_kernel=False)

            # First optimize starting from theta specified in kernel
            optima = [
                (
                    self._constrained_optimization(
                        obj_func, self.kernel_.theta, self.kernel_.bounds
                    )
                )
            ]

            # Additional runs are performed from log-uniform chosen initial
            # theta
            if self.n_restarts_optimizer > 0:
                if not np.isfinite(self.kernel_.bounds).all():
                    raise ValueError(
                        "Multiple optimizer restarts (n_restarts_optimizer>0) "
                        "requires that all bounds are finite."
                    )
                bounds = self.kernel_.bounds
                for iteration in range(self.n_restarts_optimizer):
                    theta_initial = self._rng.uniform(bounds[:, 0], bounds[:, 1])
                    optima.append(
                        self._constrained_optimization(obj_func, theta_initial, bounds)
                    )
            # Select result from run with minimal (negative) log-marginal
            # likelihood
            lml_values = list(map(itemgetter(1), optima))
            self.kernel_.theta = optima[np.argmin(lml_values)][0]
            self.kernel_._check_bounds_params()

            self.log_marginal_likelihood_value_ = -np.min(lml_values)
        else:
            self.log_marginal_likelihood_value_ = self.log_marginal_likelihood(
                self.kernel_.theta, clone_kernel=False
            )

        # Precompute quantities required for predictions which are independent
        # of actual query points
        # Alg. 2.1, page 19, line 2 -> L = cholesky(K + sigma^2 I)
        K = self.kernel_(self.X_train_)
        K[np.diag_indices_from(K)] += self.alpha
        try:
            self.L_ = cholesky(K, lower=GPR_CHOLESKY_LOWER, check_finite=False)
        except np.linalg.LinAlgError as exc:
            exc.args = (
                (
                    f"The kernel, {self.kernel_}, is not returning a positive "
                    "definite matrix. Try gradually increasing the 'alpha' "
                    "parameter of your GaussianProcessRegressor estimator."
                ),
            ) + exc.args
            raise
        # Alg 2.1, page 19, line 3 -> alpha = L^T \ (L \ y)
        self.alpha_ = cho_solve(
            (self.L_, GPR_CHOLESKY_LOWER),
            self.y_train_,
            check_finite=False,
        )
        return self

    # ------------------------------------------------------------------
    # Derivative-aware fitting (observations of f and of its gradients)
    # ------------------------------------------------------------------

    def _derivative_canonical_theta(self, parts):
        """Return the canonical theta ``[log(C), log(l_1), ...]``."""
        const_kernel, rbf_kernel = parts
        ls = np.asarray(rbf_kernel.length_scale, dtype=np.float64).ravel()
        length_theta = np.log(ls) if ls.size > 1 else np.log(ls[:1])
        return np.concatenate(
            [np.log([const_kernel.constant_value]), length_theta]
        )

    def _derivative_bounds(self, parts):
        """Return bounds on the canonical theta, finite or ``(-inf, inf)``."""
        const_kernel, rbf_kernel = parts
        const_hp = const_kernel.hyperparameter_constant_value
        if const_hp.fixed:
            const_bounds = np.full(
                (1, 2), np.log(float(const_kernel.constant_value))
            )
        else:
            const_bounds = np.log(np.asarray(const_hp.bounds, dtype=np.float64))

        ls_hp = rbf_kernel.hyperparameter_length_scale
        ls = np.asarray(rbf_kernel.length_scale, dtype=np.float64).ravel()
        n_length = ls.size if ls.size > 1 else 1
        if ls_hp.fixed:
            length_bounds = np.column_stack(
                [np.log(ls) if ls.size > 1 else np.full(1, np.log(ls[0]))] * 2
            )
        else:
            length_bounds = np.log(np.asarray(ls_hp.bounds, dtype=np.float64))
        return np.vstack([const_bounds, length_bounds])

    def _set_derivative_kernel_theta(self, parts, theta):
        """Put canonical theta back onto ``self.kernel_`` in place."""
        const_kernel, rbf_kernel = parts
        const_kernel.constant_value = float(np.exp(theta[0]))
        if np.asarray(rbf_kernel.length_scale, dtype=np.float64).ravel().size > 1:
            rbf_kernel.length_scale = np.exp(theta[1:])
        else:
            rbf_kernel.length_scale = float(np.exp(theta[1]))

    def _derivative_cov(
        self, X, kind_x, Y, kind_y, eval_gradient=False, kernel=None
    ):
        """Joint value/derivative covariance using the current kernel_."""
        if kernel is None:
            kernel = self.kernel_
        parts = _unpack_constant_rbf_kernel(kernel)
        const_kernel, rbf_kernel = parts
        return _rbf_derivative_cov(
            const_kernel.constant_value,
            rbf_kernel.length_scale,
            X,
            kind_x,
            Y,
            kind_y,
            eval_gradient=eval_gradient,
        )

    def _fit_derivative_observations(self, parts, X, y, kind):
        """Fit using the joint covariance of values and derivatives."""
        const_kernel, rbf_kernel = parts
        ls = np.asarray(rbf_kernel.length_scale, dtype=np.float64).ravel()
        self._derivative_n_length = ls.size if ls.size > 1 else 1
        if ls.size > 1 and ls.shape[0] != X.shape[1]:
            raise ValueError(
                "An anisotropic RBF length_scale must have as many entries as "
                f"features. Got {ls.shape[0]} length scales for {X.shape[1]} "
                "features."
            )

        if np.iterable(self.alpha):
            alpha = np.asarray(self.alpha)
            if alpha.shape[0] == 1:
                self.alpha = alpha[0]
            elif alpha.shape[0] != y.shape[0]:
                raise ValueError(
                    "alpha must be a scalar or an array with same number of "
                    f"entries as y. ({alpha.shape[0]} != {y.shape[0]})"
                )

        # Normalize targets with a statistics defined on the value rows only:
        # a zero-mean GP for f implies a zero-mean GP for its derivatives.
        if self.normalize_y:
            value_rows = kind == _FUNCTION_OBSERVATION
            y_values = y[value_rows]
            if np.any(value_rows):
                mean = np.mean(y_values, axis=0)
                std = _handle_zeros_in_scale(np.std(y_values, axis=0), copy=False)
            else:
                # No function values: fall back to statistics over derivatives
                mean = np.zeros(y.shape[1] if y.ndim == 2 else 1)
                std = _handle_zeros_in_scale(np.std(y, axis=0), copy=False)
            self._y_train_mean = np.atleast_1d(mean)
            self._y_train_std = np.atleast_1d(std)
            y = (y - self._y_train_mean) / self._y_train_std
        else:
            shape_y_stats = (y.shape[1],) if y.ndim == 2 else 1
            self._y_train_mean = np.zeros(shape=shape_y_stats)
            self._y_train_std = np.ones(shape=shape_y_stats)

        self.X_train_ = np.copy(X) if self.copy_X_train else X
        self.y_train_ = np.copy(y) if self.copy_X_train else y
        self.obs_kind_ = np.copy(kind) if self.copy_X_train else kind

        initial_theta = self._derivative_canonical_theta(parts)
        bounds = self._derivative_bounds(parts)

        if self.optimizer is not None:
            def obj_func(theta, eval_gradient=True):
                if eval_gradient:
                    lml, grad = self._log_marginal_likelihood_derivative(
                        theta, eval_gradient=True
                    )
                    return -lml, -grad
                return -self._log_marginal_likelihood_derivative(theta)

            optima = [
                self._constrained_optimization(obj_func, initial_theta, bounds)
            ]
            if self.n_restarts_optimizer > 0:
                if not np.isfinite(bounds).all():
                    raise ValueError(
                        "Multiple optimizer restarts (n_restarts_optimizer>0) "
                        "requires that all bounds are finite."
                    )
                for _ in range(self.n_restarts_optimizer):
                    theta_initial = self._rng.uniform(bounds[:, 0], bounds[:, 1])
                    optima.append(
                        self._constrained_optimization(obj_func, theta_initial, bounds)
                    )
            lml_values = list(map(itemgetter(1), optima))
            best_theta = optima[np.argmin(lml_values)][0]
            self.log_marginal_likelihood_value_ = -np.min(lml_values)
        else:
            best_theta = initial_theta
            self.log_marginal_likelihood_value_ = (
                self._log_marginal_likelihood_derivative(best_theta)
            )

        self._set_derivative_kernel_theta(parts, best_theta)

        K = self._derivative_cov(
            self.X_train_, self.obs_kind_, self.X_train_, self.obs_kind_
        )
        K[np.diag_indices_from(K)] += np.asarray(self.alpha, dtype=K.dtype)
        try:
            self.L_ = cholesky(K, lower=GPR_CHOLESKY_LOWER, check_finite=False)
        except np.linalg.LinAlgError as exc:
            exc.args = (
                (
                    f"The kernel, {self.kernel_}, is not returning a positive "
                    "definite matrix. Try gradually increasing the 'alpha' "
                    "parameter of your GaussianProcessRegressor estimator."
                ),
            ) + exc.args
            raise
        self.alpha_ = cho_solve(
            (self.L_, GPR_CHOLESKY_LOWER),
            self.y_train_,
            check_finite=False,
        )
        return self

    def _log_marginal_likelihood_derivative(
        self, theta, eval_gradient=False, clone_kernel=False
    ):
        """Log-marginal likelihood for mixed value/derivative observations."""
        if clone_kernel:
            kernel = clone(self.kernel_)
            parts = _unpack_constant_rbf_kernel(kernel)
            self._set_derivative_kernel_theta(parts, theta)
        else:
            kernel = self.kernel_
            parts = _unpack_constant_rbf_kernel(kernel)
            self._set_derivative_kernel_theta(parts, theta)
        if eval_gradient:
            K, K_gradient = self._derivative_cov(
                self.X_train_,
                self.obs_kind_,
                self.X_train_,
                self.obs_kind_,
                eval_gradient=True,
                kernel=kernel,
            )
        else:
            K = self._derivative_cov(
                self.X_train_,
                self.obs_kind_,
                self.X_train_,
                self.obs_kind_,
                kernel=kernel,
            )

        K[np.diag_indices_from(K)] += np.asarray(self.alpha, dtype=K.dtype)
        try:
            L = cholesky(K, lower=GPR_CHOLESKY_LOWER, check_finite=False)
        except np.linalg.LinAlgError:
            return (-np.inf, np.zeros_like(theta)) if eval_gradient else -np.inf

        y_train = self.y_train_
        if y_train.ndim == 1:
            y_train = y_train[:, np.newaxis]

        alpha = cho_solve((L, GPR_CHOLESKY_LOWER), y_train, check_finite=False)

        log_likelihood_dims = -0.5 * np.einsum("ik,ik->k", y_train, alpha)
        log_likelihood_dims -= np.log(np.diag(L)).sum()
        log_likelihood_dims -= K.shape[0] / 2 * np.log(2 * np.pi)
        log_likelihood = log_likelihood_dims.sum(axis=-1)

        if eval_gradient:
            inner_term = np.einsum("ik,jk->ijk", alpha, alpha)
            K_inv = cho_solve(
                (L, GPR_CHOLESKY_LOWER), np.eye(K.shape[0]), check_finite=False
            )
            inner_term -= K_inv[..., np.newaxis]
            log_likelihood_gradient_dims = 0.5 * np.einsum(
                "ijl,jik->kl", inner_term, K_gradient
            )
            log_likelihood_gradient = log_likelihood_gradient_dims.sum(axis=-1)
            return log_likelihood, log_likelihood_gradient
        return log_likelihood

    def _predict_derivative_prior(
        self, kernel, X, predict_kind, return_std, return_cov
    ):
        """Prior predictions (zero mean) mixing values and derivatives."""
        n_targets = self.n_targets if self.n_targets is not None else 1
        y_mean = np.zeros(shape=(X.shape[0], n_targets)).squeeze()

        if not return_cov and not return_std:
            return y_mean

        y_cov = self._derivative_cov(
            X, predict_kind, X, predict_kind, kernel=kernel
        )
        if return_cov:
            if n_targets > 1:
                y_cov = np.repeat(np.expand_dims(y_cov, -1), n_targets, axis=-1)
            return y_mean, y_cov
        y_var = np.diag(y_cov).copy()
        if n_targets > 1:
            y_var = np.repeat(np.expand_dims(y_var, -1), n_targets, axis=-1)
        return y_mean, np.sqrt(y_var)

    def _predict_derivative(self, X, predict_kind, return_std, return_cov):
        """Posterior predictions mixing values and partial derivatives."""
        if predict_kind is None:
            predict_kind = np.full(X.shape[0], _FUNCTION_OBSERVATION, dtype=np.intp)

        # Cross-covariance between the requested quantities and the
        # observations; Alg. 2.1 line 4 with derivative operator blocks.
        K_trans = self._derivative_cov(
            X, predict_kind, self.X_train_, self.obs_kind_
        )
        y_mean = K_trans @ self.alpha_

        # Undo normalisation: derivatives scale with the value std but have
        # no mean offset.
        y_mean = y_mean * self._y_train_std
        value_rows = predict_kind == _FUNCTION_OBSERVATION
        if y_mean.ndim == 1:
            y_mean = y_mean.copy()
            y_mean[value_rows] += self._y_train_mean
        else:
            y_mean = y_mean.copy()
            y_mean[value_rows, :] += self._y_train_mean
            if y_mean.shape[1] == 1:
                y_mean = np.squeeze(y_mean, axis=1)

        if not return_cov and not return_std:
            return y_mean

        V = solve_triangular(
            self.L_, K_trans.T, lower=GPR_CHOLESKY_LOWER, check_finite=False
        )
        # Joint covariance of the mixed query quantities.
        y_cov = self._derivative_cov(X, predict_kind, X, predict_kind)
        y_cov -= V.T @ V

        if return_cov:
            y_cov = np.outer(
                y_cov, self._y_train_std**2
            ).reshape(*y_cov.shape, -1)
            if y_cov.shape[2] == 1:
                y_cov = np.squeeze(y_cov, axis=2)
            return y_mean, y_cov

        y_var = np.diag(y_cov).copy()
        y_var_negative = y_var < 0
        if np.any(y_var_negative):
            warnings.warn(
                "Predicted variances smaller than 0. "
                "Setting those variances to 0."
            )
            y_var[y_var_negative] = 0.0
        y_var = np.outer(y_var, self._y_train_std**2).reshape(
            y_var.shape[0], -1
        )
        if y_var.shape[1] == 1:
            y_var = np.squeeze(y_var, axis=1)
        return y_mean, np.sqrt(y_var)

    def predict(self, X, return_std=False, return_cov=False, *, kind=None):
        """Predict using the Gaussian process regression model.

        We can also predict based on an unfitted model by using the GP prior.
        In addition to the mean of the predictive distribution, optionally also
        returns its standard deviation (`return_std=True`) or covariance
        (`return_cov=True`). Note that at most one of the two can be requested.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or list of object
            Query points where the GP is evaluated.

        return_std : bool, default=False
            If True, the standard-deviation of the predictive distribution at
            the query points is returned along with the mean.

        return_cov : bool, default=False
            If True, the covariance of the joint predictive distribution at
            the query points is returned along with the mean.

        kind : array-like of shape (n_samples,), default=None
            Tags indicating whether to predict the function value or a partial
            derivative at each query point. Entries are ``-1`` (or ``None`` for
            all rows) for the function value and ``j >= 0`` for
            :math:`\\partial f / \\partial x_j`. Derivative predictions are
            supported after fitting with (or, for the prior, with a kernel
            restricted to) a ``ConstantKernel * RBF`` kernel.

            .. versionadded:: 1.10

        Returns
        -------
        y_mean : ndarray of shape (n_samples,) or (n_samples, n_targets)
            Mean of predictive distribution at query points.

        y_std : ndarray of shape (n_samples,) or (n_samples, n_targets), optional
            Standard deviation of predictive distribution at query points.
            Only returned when `return_std` is True.

        y_cov : ndarray of shape (n_samples, n_samples) or \
                (n_samples, n_samples, n_targets), optional
            Covariance of joint predictive distribution at query points.
            Only returned when `return_cov` is True.
        """
        if return_std and return_cov:
            raise RuntimeError(
                "At most one of return_std or return_cov can be requested."
            )

        if self.kernel is None or self.kernel.requires_vector_input:
            dtype, ensure_2d = "numeric", True
        else:
            dtype, ensure_2d = None, False

        X = validate_data(self, X, ensure_2d=ensure_2d, dtype=dtype, reset=False)

        n_features_in = X.shape[1] if X.ndim == 2 else None
        if kind is not None:
            predict_kind = _check_obs_kind(
                kind, X.shape[0], n_features_in, "kind"
            )
        else:
            predict_kind = None
        derivative_predictions = predict_kind is not None and bool(
            np.any(predict_kind >= 0)
        )

        if not hasattr(self, "X_train_"):  # Unfitted;predict based on GP prior
            if self.kernel is None:
                kernel = C(1.0, constant_value_bounds="fixed") * RBF(
                    1.0, length_scale_bounds="fixed"
                )
            else:
                kernel = self.kernel

            if predict_kind is not None:
                if _unpack_constant_rbf_kernel(kernel) is None:
                    raise ValueError(
                        "Predicting derivatives requires a kernel that is a "
                        "product of a ConstantKernel and an isotropic or "
                        f"anisotropic RBF, got {kernel!r}."
                    )
                return self._predict_derivative_prior(
                    kernel, X, predict_kind, return_std, return_cov
                )

            n_targets = self.n_targets if self.n_targets is not None else 1
            y_mean = np.zeros(shape=(X.shape[0], n_targets)).squeeze()

            if return_cov:
                y_cov = kernel(X)
                if n_targets > 1:
                    y_cov = np.repeat(
                        np.expand_dims(y_cov, -1), repeats=n_targets, axis=-1
                    )
                return y_mean, y_cov
            elif return_std:
                y_var = kernel.diag(X)
                if n_targets > 1:
                    y_var = np.repeat(
                        np.expand_dims(y_var, -1), repeats=n_targets, axis=-1
                    )
                return y_mean, np.sqrt(y_var)
            else:
                return y_mean
        elif derivative_predictions or self._has_derivative_observations:
            if _unpack_constant_rbf_kernel(self.kernel_) is None:
                raise ValueError(
                    "Predicting derivatives requires a kernel that is a "
                    "product of a ConstantKernel and an isotropic or "
                    f"anisotropic RBF, got {self.kernel_!r}."
                )
            return self._predict_derivative(
                X, predict_kind, return_std, return_cov
            )
        else:  # Predict based on GP posterior
            # Alg 2.1, page 19, line 4 -> f*_bar = K(X_test, X_train) . alpha
            K_trans = self.kernel_(X, self.X_train_)
            y_mean = K_trans @ self.alpha_

            # undo normalisation
            y_mean = self._y_train_std * y_mean + self._y_train_mean

            # if y_mean has shape (n_samples, 1), reshape to (n_samples,)
            if y_mean.ndim > 1 and y_mean.shape[1] == 1:
                y_mean = np.squeeze(y_mean, axis=1)

            if not return_cov and not return_std:
                return y_mean

            # Alg 2.1, page 19, line 5 -> v = L \ K(X_test, X_train)^T
            V = solve_triangular(
                self.L_, K_trans.T, lower=GPR_CHOLESKY_LOWER, check_finite=False
            )

            if return_cov:
                # Alg 2.1, page 19, line 6 -> K(X_test, X_test) - v^T. v
                y_cov = self.kernel_(X) - V.T @ V

                # undo normalisation
                y_cov = np.outer(y_cov, self._y_train_std**2).reshape(*y_cov.shape, -1)
                # if y_cov has shape (n_samples, n_samples, 1), reshape to
                # (n_samples, n_samples)
                if y_cov.shape[2] == 1:
                    y_cov = np.squeeze(y_cov, axis=2)

                return y_mean, y_cov
            else:  # return_std
                # Compute variance of predictive distribution
                # Use einsum to avoid explicitly forming the large matrix
                # V^T @ V just to extract its diagonal afterward.
                y_var = self.kernel_.diag(X).copy()
                y_var -= np.einsum("ij,ji->i", V.T, V)

                # Check if any of the variances is negative because of
                # numerical issues. If yes: set the variance to 0.
                y_var_negative = y_var < 0
                if np.any(y_var_negative):
                    warnings.warn(
                        "Predicted variances smaller than 0. "
                        "Setting those variances to 0."
                    )
                    y_var[y_var_negative] = 0.0

                # undo normalisation
                y_var = np.outer(y_var, self._y_train_std**2).reshape(*y_var.shape, -1)

                # if y_var has shape (n_samples, 1), reshape to (n_samples,)
                if y_var.shape[1] == 1:
                    y_var = np.squeeze(y_var, axis=1)

                return y_mean, np.sqrt(y_var)

    def sample_y(self, X, n_samples=1, random_state=0, *, kind=None):
        """Draw samples from Gaussian process and evaluate at X.

        Parameters
        ----------
        X : array-like of shape (n_samples_X, n_features) or list of object
            Query points where the GP is evaluated.

        n_samples : int, default=1
            Number of samples drawn from the Gaussian process per query point.

        random_state : int, RandomState instance or None, default=0
            Determines random number generation to randomly draw samples.
            Pass an int for reproducible results across multiple function
            calls.
            See :term:`Glossary <random_state>`.

        kind : array-like of shape (n_samples_X,), default=None
            Tags indicating whether to sample the function value (-1) or a
            partial derivative (``j >= 0``) at each query point, with the
            same convention as the ``kind`` parameter of :meth:`predict`.

            .. versionadded:: 1.10

        Returns
        -------
        y_samples : ndarray of shape (n_samples_X, n_samples), or \
            (n_samples_X, n_targets, n_samples)
            Values of n_samples samples drawn from Gaussian process and
            evaluated at query points.
        """
        rng = check_random_state(random_state)

        y_mean, y_cov = self.predict(X, return_cov=True, kind=kind)
        if y_mean.ndim == 1:
            y_samples = rng.multivariate_normal(y_mean, y_cov, n_samples).T
        else:
            y_samples = [
                rng.multivariate_normal(
                    y_mean[:, target], y_cov[..., target], n_samples
                ).T[:, np.newaxis]
                for target in range(y_mean.shape[1])
            ]
            y_samples = np.hstack(y_samples)
        return y_samples

    def log_marginal_likelihood(
        self, theta=None, eval_gradient=False, clone_kernel=True
    ):
        """Return log-marginal likelihood of theta for training data.

        Parameters
        ----------
        theta : array-like of shape (n_kernel_params,) default=None
            Kernel hyperparameters for which the log-marginal likelihood is
            evaluated. If None, the precomputed log_marginal_likelihood
            of ``self.kernel_.theta`` is returned. When fitting with
            derivative observations, ``theta`` follows the canonical ordering
            ``[log(constant_value), log(length_scale_1), ...]``.

        eval_gradient : bool, default=False
            If True, the gradient of the log-marginal likelihood with respect
            to the kernel hyperparameters at position theta is returned
            additionally. If True, theta must not be None.

        clone_kernel : bool, default=True
            If True, the kernel attribute is copied. If False, the kernel
            attribute is modified, but may result in a performance improvement.

        Returns
        -------
        log_likelihood : float
            Log-marginal likelihood of theta for training data.

        log_likelihood_gradient : ndarray of shape (n_kernel_params,), optional
            Gradient of the log-marginal likelihood with respect to the kernel
            hyperparameters at position theta.
            Only returned when eval_gradient is True.
        """
        if theta is None:
            if eval_gradient:
                raise ValueError("Gradient can only be evaluated for theta!=None")
            return self.log_marginal_likelihood_value_

        if getattr(self, "_has_derivative_observations", False):
            # In the derivative-aware case, ``theta`` follows the canonical
            # ordering ``[log(constant_value), log(length_scale)...]``.
            theta = np.asarray(theta, dtype=np.float64)
            n_expected = 1 + self._derivative_n_length
            if theta.shape[0] != n_expected:
                raise ValueError(
                    "theta must have shape (n_kernel_params,) with "
                    f"{n_expected} entries [log(constant_value), "
                    f"log(length_scale)...]; got {theta.shape[0]} entries."
                )
            return self._log_marginal_likelihood_derivative(
                theta, eval_gradient=eval_gradient, clone_kernel=clone_kernel
            )

        if clone_kernel:
            kernel = self.kernel_.clone_with_theta(theta)
        else:
            kernel = self.kernel_
            kernel.theta = theta

        if eval_gradient:
            K, K_gradient = kernel(self.X_train_, eval_gradient=True)
        else:
            K = kernel(self.X_train_)

        # Alg. 2.1, page 19, line 2 -> L = cholesky(K + sigma^2 I)
        K[np.diag_indices_from(K)] += self.alpha
        try:
            L = cholesky(K, lower=GPR_CHOLESKY_LOWER, check_finite=False)
        except np.linalg.LinAlgError:
            return (-np.inf, np.zeros_like(theta)) if eval_gradient else -np.inf

        # Support multi-dimensional output of self.y_train_
        y_train = self.y_train_
        if y_train.ndim == 1:
            y_train = y_train[:, np.newaxis]

        # Alg 2.1, page 19, line 3 -> alpha = L^T \ (L \ y)
        alpha = cho_solve((L, GPR_CHOLESKY_LOWER), y_train, check_finite=False)

        # Alg 2.1, page 19, line 7
        # -0.5 . y^T . alpha - sum(log(diag(L))) - n_samples / 2 log(2*pi)
        # y is originally thought to be a (1, n_samples) row vector. However,
        # in multioutputs, y is of shape (n_samples, 2) and we need to compute
        # y^T . alpha for each output, independently using einsum. Thus, it
        # is equivalent to:
        # for output_idx in range(n_outputs):
        #     log_likelihood_dims[output_idx] = (
        #         y_train[:, [output_idx]] @ alpha[:, [output_idx]]
        #     )
        log_likelihood_dims = -0.5 * np.einsum("ik,ik->k", y_train, alpha)
        log_likelihood_dims -= np.log(np.diag(L)).sum()
        log_likelihood_dims -= K.shape[0] / 2 * np.log(2 * np.pi)
        # the log likelihood is sum-up across the outputs
        log_likelihood = log_likelihood_dims.sum(axis=-1)

        if eval_gradient:
            # Eq. 5.9, p. 114, and footnote 5 in p. 114
            # 0.5 * trace((alpha . alpha^T - K^-1) . K_gradient)
            # alpha is supposed to be a vector of (n_samples,) elements. With
            # multioutputs, alpha is a matrix of size (n_samples, n_outputs).
            # Therefore, we want to construct a matrix of
            # (n_samples, n_samples, n_outputs) equivalent to
            # for output_idx in range(n_outputs):
            #     output_alpha = alpha[:, [output_idx]]
            #     inner_term[..., output_idx] = output_alpha @ output_alpha.T
            inner_term = np.einsum("ik,jk->ijk", alpha, alpha)
            # compute K^-1 of shape (n_samples, n_samples)
            K_inv = cho_solve(
                (L, GPR_CHOLESKY_LOWER), np.eye(K.shape[0]), check_finite=False
            )
            # create a new axis to use broadcasting between inner_term and
            # K_inv
            inner_term -= K_inv[..., np.newaxis]
            # Since we are interested about the trace of
            # inner_term @ K_gradient, we don't explicitly compute the
            # matrix-by-matrix operation and instead use an einsum. Therefore
            # it is equivalent to:
            # for param_idx in range(n_kernel_params):
            #     for output_idx in range(n_output):
            #         log_likelihood_gradient_dims[param_idx, output_idx] = (
            #             inner_term[..., output_idx] @
            #             K_gradient[..., param_idx]
            #         )
            log_likelihood_gradient_dims = 0.5 * np.einsum(
                "ijl,jik->kl", inner_term, K_gradient
            )
            # the log likelihood gradient is the sum-up across the outputs
            log_likelihood_gradient = log_likelihood_gradient_dims.sum(axis=-1)

        if eval_gradient:
            return log_likelihood, log_likelihood_gradient
        else:
            return log_likelihood

    def _constrained_optimization(self, obj_func, initial_theta, bounds):
        if self.optimizer == "fmin_l_bfgs_b":
            opt_res = scipy.optimize.minimize(
                obj_func,
                initial_theta,
                method="L-BFGS-B",
                jac=True,
                bounds=bounds,
            )
            _check_optimize_result("lbfgs", opt_res)
            theta_opt, func_min = opt_res.x, opt_res.fun
        elif callable(self.optimizer):
            theta_opt, func_min = self.optimizer(obj_func, initial_theta, bounds=bounds)
        else:
            raise ValueError(f"Unknown optimizer {self.optimizer}.")

        return theta_opt, func_min

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.requires_fit = False
        return tags
