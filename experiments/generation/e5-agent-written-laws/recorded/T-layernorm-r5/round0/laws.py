import numpy as np

EPS32 = 2.0 ** -23
VAR_EPS = 1e-5


def _invalid(x, y):
    if not isinstance(y, np.ndarray):
        return "output is not an ndarray"
    if y.shape != x.shape:
        return "shape %s != %s" % (y.shape, x.shape)
    if y.dtype != np.float32:
        return "dtype %s != float32" % y.dtype
    if not np.all(np.isfinite(y)):
        return "non-finite output"
    return None


def _scales(x):
    """Per-row rounding allowance e (in units of y) and expected mean(y^2), from x only."""
    x64 = np.asarray(x, dtype=np.float64)
    n = x64.shape[1]
    m = x64.mean(axis=1, keepdims=True)
    xc = x64 - m
    v = (xc * xc).mean(axis=1)
    s = np.sqrt(v + VAR_EPS)
    a = np.abs(x64).max(axis=1)
    q = a / s
    q2 = (x64 * x64).mean(axis=1) / (s * s)
    z = np.maximum(np.abs(xc).max(axis=1) / s, 1.0)
    g = 1.0 + np.sqrt(n) / 16.0
    e = 4.0 * EPS32 * (2.0 + g * q + z * q2)
    exp_ms = v / (v + VAR_EPS)
    return x64, xc, e, exp_ms


def shape_dtype_finite(x, y):
    msg = _invalid(x, y)
    if msg:
        return False, msg
    return True, "ok"


def row_mean_zero(x, y):
    msg = _invalid(x, y)
    if msg:
        return False, msg
    _, _, e, _ = _scales(x)
    mu = np.abs(y.astype(np.float64).mean(axis=1))
    bad = mu > e
    if bad.any():
        i = int(np.argmax(mu - e))
        return False, "row %d mean(y)=%.3g exceeds %.3g" % (i, mu[i], e[i])
    return True, "ok"


def row_mean_square(x, y):
    # mean(y^2) = v/(v+1e-5) for a correct output; checks scale, the 1e-5, and population (not sample) variance
    msg = _invalid(x, y)
    if msg:
        return False, msg
    _, _, e, exp_ms = _scales(x)
    ms = (y.astype(np.float64) ** 2).mean(axis=1)
    tol = 2.0 * e * np.sqrt(exp_ms) + e * e
    d = np.abs(ms - exp_ms)
    if (d > tol).any():
        i = int(np.argmax(d - tol))
        return False, "row %d mean(y^2)=%.6g expected %.6g (tol %.3g)" % (i, ms[i], exp_ms[i], tol[i])
    return True, "ok"


def row_affine_in_x(x, y):
    # each output row must be a positive-slope affine function of the input row
    msg = _invalid(x, y)
    if msg:
        return False, msg
    x64, xc, e, _ = _scales(x)
    y64 = y.astype(np.float64)
    yc = y64 - y64.mean(axis=1, keepdims=True)
    sxx = (xc * xc).sum(axis=1)
    sxy = (xc * yc).sum(axis=1)
    alpha = np.where(sxx > 0, sxy / np.where(sxx > 0, sxx, 1.0), 0.0)
    res = np.abs(yc - alpha[:, None] * xc).max(axis=1)
    tol = 3.0 * e
    if (res > tol).any():
        i = int(np.argmax(res - tol))
        return False, "row %d not affine in x: residual %.3g > %.3g" % (i, res[i], tol[i])
    if (alpha < 0).any() and (np.abs(alpha) * np.sqrt(sxx) > tol).any():
        i = int(np.argmax((alpha < 0) & (np.abs(alpha) * np.sqrt(sxx) > tol)))
        return False, "row %d has negative slope" % i
    return True, "ok"


def row_monotone(x, y):
    msg = _invalid(x, y)
    if msg:
        return False, msg
    x64, _, e, _ = _scales(x)
    order = np.argsort(x64, axis=1, kind="stable")
    ys = np.take_along_axis(y.astype(np.float64), order, axis=1)
    if ys.shape[1] < 2:
        return True, "ok"
    drop = (ys[:, :-1] - ys[:, 1:]).max(axis=1)
    tol = 2.0 * e
    if (drop > tol).any():
        i = int(np.argmax(drop - tol))
        return False, "row %d output not monotone in x (drop %.3g > %.3g)" % (i, drop[i], tol[i])
    return True, "ok"


def shift_invariance(x0, y0, x1, y1):
    msg = _invalid(x0, y0) or _invalid(x1, y1)
    if msg:
        return False, msg
    _, _, e0, _ = _scales(x0)
    _, _, e1, _ = _scales(x1)
    d = np.abs(y1.astype(np.float64) - y0.astype(np.float64)).max(axis=1)
    tol = e0 + e1
    if (d > tol).any():
        i = int(np.argmax(d - tol))
        return False, "row %d output changed by %.3g under shift (tol %.3g)" % (i, d[i], tol[i])
    return True, "ok"


CASE_LAWS = {
    "shape_dtype_finite": shape_dtype_finite,
    "row_mean_zero": row_mean_zero,
    "row_mean_square": row_mean_square,
    "row_affine_in_x": row_affine_in_x,
    "row_monotone": row_monotone,
}
SHIFT_LAWS = {"shift_invariance": shift_invariance}
