"""Covariances between function-value and partial-derivative observations.

Gaussian processes can be conditioned not only on observations of the
function itself but also on observations of its partial derivatives.  This
module builds the (mixed) covariance matrices that are needed in this case,
for the product of a :class:`ConstantKernel` with an (an)isotropic
:class:`RBF` kernel::

    k(x, y) = sigma**2 * exp(-0.5 * sum_d (x_d - y_d)**2 / l_d**2)

For observations of partial derivatives the covariance follows from the
derivative property of Gaussian processes, writing ``d_i f`` for the
partial derivative of ``f`` along feature ``i``:

* ``Cov(  f(x),   d_i f(y)) = d_{y_i} k(x, y)``
* ``Cov(d_j f(x),   f(y) ) = d_{x_j} k(x, y)``
* ``Cov(d_j f(x), d_i f(y)) = d_{x_j} d_{y_i} k(x, y)``

See Chapter 9.4 of Rasmussen & Williams (2006) for details.
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Product

# Tag used to denote an observation of the function value itself (as opposed
# to an observation of a partial derivative, which is tagged with the index of
# the differentiated feature).
FUNCTION_TAG = -1


def _unpack_constant_rbf(kernel):
    """Return the ``(ConstantKernel, RBF)`` factors of a product kernel.

    Parameters
    ----------
    kernel : kernel instance
        The kernel that is expected to be a product of a
        :class:`ConstantKernel` and an :class:`RBF`.

    Returns
    -------
    constant_kernel : ConstantKernel instance
        The constant factor of the product.

    rbf : RBF instance
        The RBF factor of the product.

    Raises
    ------
    ValueError
        If ``kernel`` is not a product of a :class:`ConstantKernel` and an
        :class:`RBF`.
    """
    if not isinstance(kernel, Product):
        raise ValueError(
            "Observations of partial derivatives are only supported with a "
            "kernel that is the product of a ConstantKernel and an RBF "
            f"kernel, got {kernel!r}."
        )
    k1, k2 = kernel.k1, kernel.k2
    if isinstance(k1, ConstantKernel) and isinstance(k2, RBF):
        return k1, k2
    if isinstance(k2, ConstantKernel) and isinstance(k1, RBF):
        return k2, k1
    raise ValueError(
        "Observations of partial derivatives are only supported with a "
        "kernel that is the product of a ConstantKernel and an RBF kernel, "
        f"got {kernel!r}."
    )


def _check_derivative(derivative, X, estimator_name="GaussianProcessRegressor"):
    """Validate and canonicalize an array of observation kinds.

    Parameters
    ----------
    derivative : array-like of shape (n_samples,) or None
        For every row of ``X`` the feature index whose partial derivative
        has been observed at this location, or -1 if the function value
        itself has been observed. ``None`` means that all rows are
        function-value observations.

    X : ndarray of shape (n_samples, n_features)
        The locations at which the observations were made.

    estimator_name : str, default="GaussianProcessRegressor"
        Name used in error messages.

    Returns
    -------
    derivative : ndarray of shape (n_samples,) of int
        Canonical array of feature indices, with -1 denoting function
        values.
    """
    n_samples = X.shape[0]
    if derivative is None:
        return np.full(n_samples, FUNCTION_TAG, dtype=np.int64)

    derivative = np.asarray(derivative)
    if derivative.ndim != 1 or derivative.shape[0] != n_samples:
        raise ValueError(
            "`derivative` must be an array of shape (n_samples,), got shape "
            f"{derivative.shape} for {n_samples} samples."
        )
    if not np.issubdtype(derivative.dtype, np.integer):
        raise ValueError(
            "`derivative` must contain integers only (-1 for function-value "
            "observations, non-negative feature indices for partial "
            f"derivative observations), got dtype {derivative.dtype}."
        )
    n_features = X.shape[1]
    valid = (derivative == FUNCTION_TAG) | (
        (derivative >= 0) & (derivative < n_features)
    )
    if not np.all(valid):
        raise ValueError(
            "`derivative` contains an entry that is neither -1 (function "
            "value) nor a feature index in "
            f"[0, {n_features - 1}]."
        )
    return derivative.astype(np.int64, copy=False)


def _length_scales(rbf, n_features):
    """Return the per-feature length scales as a 1d float array."""
    ell = np.atleast_1d(rbf.length_scale).astype(float)
    if ell.shape[0] == 1:
        ell = np.full(n_features, ell[0])
    elif ell.shape[0] != n_features:
        raise ValueError(
            "An anisotropic RBF used with partial-derivative observations "
            "must have as many length scales as input features, got "
            f"{ell.shape[0]} length scales for {n_features} features."
        )
    return ell


def _mixed_blocks(constant_kernel, rbf, X, Y, dx, dy):
    """Build the quantities shared by covariance/gradient evaluations.

    Returns a dictionary with the four blocks of the mixed covariance
    (function/function, function/derivative, derivative/function and
    derivative/derivative), together with the intermediate quantities
    required to differentiate them with respect to the hyperparameters.
    Blocks that are absent are reported as ``None``.
    """
    sigma2 = constant_kernel.constant_value
    ell = _length_scales(rbf, X.shape[1])

    # u[i, j, d] = (x_i_d - y_j_d) / l_d
    u = (X[:, np.newaxis, :] - Y[np.newaxis, :, :]) / ell[np.newaxis, :]
    k = sigma2 * np.exp(-0.5 * np.sum(u**2, axis=-1))

    fx = dx < 0
    fy = dy < 0
    f_i = np.where(fx)[0]
    f_j = np.where(fy)[0]
    d_i = np.where(~fx)[0]
    d_j = np.where(~fy)[0]
    di = dx[d_i]
    dj = dy[d_j]

    out = dict(
        k=k,
        u=u,
        ell=ell,
        f_i=f_i,
        f_j=f_j,
        d_i=d_i,
        d_j=d_j,
        di=di,
        dj=dj,
        K_ff=None,
        K_fd=None,
        K_df=None,
        K_dd=None,
    )

    if f_i.size and f_j.size:
        # Cov(f, f) = k
        out["K_ff"] = k[np.ix_(f_i, f_j)]

    if f_i.size and d_j.size:
        # Cov(f(x), d_d f(y)) = k * u_d / l_d
        out["K_fd"] = (
            k[np.ix_(f_i, d_j)]
            * u[np.ix_(f_i, d_j)][
                np.arange(f_i.size)[:, np.newaxis],
                np.arange(d_j.size)[np.newaxis, :],
                dj[np.newaxis, :],
            ]
            / ell[dj][np.newaxis, :]
        )

    if d_i.size and f_j.size:
        # Cov(d_d f(x), f(y)) = -k * u_d / l_d
        out["K_df"] = (
            -k[np.ix_(d_i, f_j)]
            * u[np.ix_(d_i, f_j)][
                np.arange(d_i.size)[:, np.newaxis],
                np.arange(f_j.size)[np.newaxis, :],
                di[:, np.newaxis],
            ]
            / ell[di][:, np.newaxis]
        )

    if d_i.size and d_j.size:
        # Cov(d_e f(x), d_d f(y)) =
        #   k / (l_e l_d) * (delta_ed - u_e * u_d)
        # where u_d = (x_d - y_d) / l_d.
        ue = u[np.ix_(d_i, d_j)][
            np.arange(d_i.size)[:, np.newaxis],
            np.arange(d_j.size)[np.newaxis, :],
            di[:, np.newaxis],
        ]
        ud = u[np.ix_(d_i, d_j)][
            np.arange(d_i.size)[:, np.newaxis],
            np.arange(d_j.size)[np.newaxis, :],
            dj[np.newaxis, :],
        ]
        same = (di[:, np.newaxis] == dj[np.newaxis, :]).astype(float)
        kdd = k[np.ix_(d_i, d_j)]
        out["K_dd"] = (
            kdd / (ell[di][:, np.newaxis] * ell[dj][np.newaxis, :]) * (same - ue * ud)
        )
        out["ue"] = ue
        out["ud"] = ud
        out["same"] = same
        out["kdd"] = kdd

    return out


def _assemble(K_shape, blocks, key):
    """Scatter the four blocks of ``key`` into a full matrix."""
    M = np.zeros(K_shape, dtype=float)
    f_i, f_j = blocks["f_i"], blocks["f_j"]
    d_i, d_j = blocks["d_i"], blocks["d_j"]
    ff, fd, df, dd = (blocks[f"{key}_{s}"] for s in ("ff", "fd", "df", "dd"))
    if ff is not None:
        M[np.ix_(f_i, f_j)] = ff
    if fd is not None:
        M[np.ix_(f_i, d_j)] = fd
    if df is not None:
        M[np.ix_(d_i, f_j)] = df
    if dd is not None:
        M[np.ix_(d_i, d_j)] = dd
    return M


def _mixed_covariance(constant_kernel, rbf, X, Y, dx, dy):
    """Evaluate the mixed covariance between (derivative) observations.

    Each row of ``X`` and ``Y`` is tagged with a feature index (or -1):
    a tag of -1 denotes an observation of the function value, a non-negative
    index ``d`` denotes an observation of the partial derivative along
    feature ``d``.  The returned matrix contains the covariance between the
    corresponding (derivatives of the) GP.

    Parameters
    ----------
    constant_kernel : ConstantKernel instance
        The constant factor ``sigma**2`` of the kernel.

    rbf : RBF instance
        The RBF factor of the kernel (isotropic or anisotropic).

    X : ndarray of shape (n_samples_X, n_features)
        Left input locations.

    Y : ndarray of shape (n_samples_Y, n_features)
        Right input locations.

    dx : ndarray of shape (n_samples_X,) of int
        Observation kinds of the rows of ``X`` (-1 for function values).

    dy : ndarray of shape (n_samples_Y,) of int
        Observation kinds of the rows of ``Y`` (-1 for function values).

    Returns
    -------
    K : ndarray of shape (n_samples_X, n_samples_Y)
        Mixed covariance matrix.
    """
    blocks = _mixed_blocks(constant_kernel, rbf, X, Y, dx, dy)
    return _assemble((X.shape[0], Y.shape[0]), blocks, "K")


def _mixed_diag(constant_kernel, rbf, X, dx):
    """Diagonal of the mixed covariance matrix of observations at ``X``.

    Parameters
    ----------
    constant_kernel : ConstantKernel instance
        The constant factor ``sigma**2`` of the kernel.

    rbf : RBF instance
        The RBF factor of the kernel (isotropic or anisotropic).

    X : ndarray of shape (n_samples, n_features)
        Input locations.

    dx : ndarray of shape (n_samples,) of int
        Observation kinds (-1 for function values).

    Returns
    -------
    diag : ndarray of shape (n_samples,)
        Diagonal entries of the mixed covariance matrix. Function-value
        observations have prior variance ``sigma**2`` and observations of
        the derivative along feature ``d`` have prior variance
        ``sigma**2 / l_d**2``.
    """
    sigma2 = constant_kernel.constant_value
    ell = _length_scales(rbf, X.shape[1])
    diag = np.full(X.shape[0], sigma2, dtype=float)
    deriv = dx >= 0
    if np.any(deriv):
        d = dx[deriv]
        diag[deriv] = sigma2 / ell[d] ** 2
    return diag


def _length_scale_gradient_sheets(blocks):
    """Gradient of the four blocks w.r.t. log length-scales.

    Returns a list of four block arrays (ff, fd, df, dd), each either
    ``None`` or an array with a trailing axis:

    * one sheet per feature for an anisotropic RBF,
    * a single sheet (the gradient w.r.t. the shared log length-scale) for
      an isotropic RBF.

    The closed-form expressions used below follow from
    ``u_d = (x_d - y_d) / l_d`` and ``k = sigma**2 exp(-sum u**2 / 2)``:

    * ``ff: dk/d log l_m = k u_m**2``
    * ``fd``: d(k u_d / l_d)/d log l_m =
      (k u_d / l_d) (u_m**2 - 2 delta_md)
    * ``df``: with ``B = -k u_e / l_e`` the value of the block,
      ``dB / d log l_m = B (u_m**2 - 2 delta_me)``
    * ``dd``: with the block ``H = k C`` where
      ``C = (delta_ed - u_e u_d) / (l_e l_d)`` and
      ``count_m = delta_me + delta_md``::

          dH / d log l_m =
              k C u_m**2 - count_m k C
              + 2 delta_me delta_md k / l_m**2

      In the isotropic case this reduces to
      ``H sum_d u_d**2 + 2 k u_e u_d / l**2``.
    """
    u = blocks["u"]
    ell = blocks["ell"]
    f_i, f_j = blocks["f_i"], blocks["f_j"]
    d_i, d_j = blocks["d_i"], blocks["d_j"]
    di, dj = blocks["di"], blocks["dj"]
    n_features = u.shape[-1]

    # Whether the RBF exposes one shared log-length-scale hyperparameter.
    isotropic = blocks["isotropic"]
    n_sheets = 1 if isotropic else n_features

    def empty(shape):
        return np.zeros(shape + (n_sheets,), dtype=float)

    # ff block
    g_ff = None
    if blocks["K_ff"] is not None:
        uu = u[np.ix_(f_i, f_j)] ** 2
        g_ff = empty((f_i.size, f_j.size))
        if isotropic:
            g_ff[..., 0] = blocks["K_ff"] * uu.sum(axis=-1)
        else:
            g_ff[:] = blocks["K_ff"][..., np.newaxis] * uu

    # fd block: B = k u_d / l_d
    g_fd = None
    if blocks["K_fd"] is not None:
        uu_f = u[np.ix_(f_i, d_j)] ** 2
        g_fd = empty((f_i.size, d_j.size))
        if isotropic:
            # the only length-scale also enters u_d, hence the -2 term
            g_fd[..., 0] = blocks["K_fd"] * (uu_f.sum(axis=-1) - 2.0)
        else:
            g_fd[:] = blocks["K_fd"][..., np.newaxis] * uu_f
            for m in range(n_features):
                g_fd[..., m] -= 2.0 * (dj[np.newaxis, :] == m) * blocks["K_fd"]

    # df block: B = -k u_e / l_e
    g_df = None
    if blocks["K_df"] is not None:
        uu_d = u[np.ix_(d_i, f_j)] ** 2
        g_df = empty((d_i.size, f_j.size))
        if isotropic:
            # B = -k u_e / l_e ; dB = B (u**2 - 2) since the only
            # length-scale also enters u_e
            g_df[..., 0] = blocks["K_df"] * (uu_d.sum(axis=-1) - 2.0)
        else:
            g_df[:] = blocks["K_df"][..., np.newaxis] * uu_d
            for m in range(n_features):
                # dB = B (u_m**2 - 2 delta_me)
                g_df[..., m] -= 2.0 * (di[:, np.newaxis] == m) * blocks["K_df"]

    # dd block
    g_dd = None
    if blocks["K_dd"] is not None:
        ue, ud, same = blocks["ue"], blocks["ud"], blocks["same"]
        kdd = blocks["kdd"]
        C = (same - ue * ud) / (ell[di][:, np.newaxis] * ell[dj][np.newaxis, :])
        uu = u[np.ix_(d_i, d_j)] ** 2
        g_dd = empty((d_i.size, d_j.size))
        if isotropic:
            ell2 = ell[0] ** 2
            # dH/dlog l = H sum u**2 - 2 k delta_ed / l**2
            #            + 4 k u_e u_d / l**2
            g_dd[..., 0] = (
                blocks["K_dd"] * uu.sum(axis=-1)
                + kdd * (-2.0 * same + 4.0 * ue * ud) / ell2
            )
        else:
            for m in range(n_features):
                count = (di[:, np.newaxis] == m).astype(float) + (
                    dj[np.newaxis, :] == m
                ).astype(float)
                g_dd[..., m] = (
                    blocks["K_dd"] * uu[..., m]
                    - count * kdd * C
                    + count
                    * kdd
                    * ue
                    * ud
                    / (ell[di][:, np.newaxis] * ell[dj][np.newaxis, :])
                )

    return g_ff, g_fd, g_df, g_dd


def _mixed_covariance_gradient(kernel, X, dx):
    """Mixed covariance and its gradient w.r.t. the log hyperparameters.

    Only the square case ``K(X, X)`` is supported, as the marginal
    likelihood gradient is only required there.

    Parameters
    ----------
    kernel : Product of ConstantKernel and RBF
        Kernel at which the covariance is evaluated. The order of the two
        factors and which hyperparameters are fixed determine the ordering
        of the returned gradient sheets (identical to ``kernel.theta``).

    X : ndarray of shape (n_samples, n_features)
        Input locations.

    dx : ndarray of shape (n_samples,) of int
        Observation kinds (-1 for function values).

    Returns
    -------
    K : ndarray of shape (n_samples, n_samples)
        Mixed covariance matrix.

    K_gradient : ndarray of shape (n_samples, n_samples, n_dims)
        Gradient with respect to the non-fixed, log-transformed
        hyperparameters of the product kernel.
    """
    constant_kernel, rbf = _unpack_constant_rbf(kernel)
    isotropic = not (np.iterable(rbf.length_scale) and len(rbf.length_scale) > 1)

    blocks = _mixed_blocks(constant_kernel, rbf, X, X, dx, dx)
    blocks["isotropic"] = isotropic
    f_i, f_j = blocks["f_i"], blocks["f_j"]
    d_i, d_j = blocks["d_i"], blocks["d_j"]
    n = X.shape[0]
    K = _assemble((n, n), blocks, "K")
    # Enforce exact symmetry (the two triangular halves are computed from
    # the same formulas but may differ at the level of floating-point noise)
    K = 0.5 * (K + K.T)

    g_ff, g_fd, g_df, g_dd = _length_scale_gradient_sheets(blocks)
    if g_ff is not None or g_fd is not None or g_df is not None or g_dd is not None:
        n_sheets = next(b.shape[-1] for b in (g_ff, g_fd, g_df, g_dd) if b is not None)
        g_ell = np.zeros((n, n, n_sheets), dtype=float)
        if g_ff is not None:
            g_ell[np.ix_(f_i, f_j, np.arange(n_sheets))] = g_ff
        if g_fd is not None:
            g_ell[np.ix_(f_i, d_j, np.arange(n_sheets))] = g_fd
        if g_df is not None:
            g_ell[np.ix_(d_i, f_j, np.arange(n_sheets))] = g_df
        if g_dd is not None:
            g_ell[np.ix_(d_i, d_j, np.arange(n_sheets))] = g_dd
    else:
        g_ell = np.zeros((n, n, 0))
    g_ell = 0.5 * (g_ell + np.transpose(g_ell, (1, 0, 2)))

    constant_free = not constant_kernel.hyperparameter_constant_value.fixed
    length_free = not rbf.hyperparameter_length_scale.fixed
    # dK / d log(sigma**2) = K (every block is linear in sigma**2)
    g_constant = K[..., np.newaxis] if constant_free else None

    if not length_free:
        g_ell = g_ell[..., :0]

    # Order the sheets exactly like ``kernel.theta``: the hyperparameters of
    # the first factor followed by those of the second factor.
    constant_is_first = kernel.k1 is constant_kernel
    sheets = []
    if g_constant is not None and constant_is_first:
        sheets.append(g_constant)
    if g_ell.shape[-1]:
        sheets.append(g_ell)
    if g_constant is not None and not constant_is_first:
        sheets.append(g_constant)

    if sheets:
        K_gradient = np.concatenate(sheets, axis=-1)
    else:
        K_gradient = np.empty((n, n, 0))
    return K, K_gradient
