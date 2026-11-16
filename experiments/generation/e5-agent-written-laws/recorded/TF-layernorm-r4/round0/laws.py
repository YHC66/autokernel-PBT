import numpy as np

EPS32 = 2.0 ** -23
LN_EPS = 1e-5
BASE = 128.0 * EPS32


def _struct(x, y):
    if not isinstance(y, np.ndarray):
        return "output is not an ndarray"
    if y.shape != x.shape:
        return "shape %s != %s" % (y.shape, x.shape)
    if y.dtype != x.dtype:
        return "dtype %s != %s" % (y.dtype, x.dtype)
    if not np.isfinite(y).all():
        return "non-finite output"
    return None


def _row_stats(x):
    x64 = x.astype(np.float64)
    n = x64.shape[1]
    d = x64 - x64.mean(axis=1, keepdims=True)
    v = (d * d).mean(axis=1)
    s = np.abs(x64).max(axis=1)
    k = 8.0 + 2.0 * np.sqrt(n)
    em = k * EPS32 * s
    dv = 2.0 * k * EPS32 * s * s
    sigma = np.sqrt(v + LN_EPS)
    return d, v, s, em, dv, sigma


def shape_dtype_finite(x, y):
    bad = _struct(x, y)
    if bad:
        return False, bad
    return True, "ok"


def mean_zero(x, y):
    bad = _struct(x, y)
    if bad:
        return False, bad
    d, v, s, em, dv, sigma = _row_stats(x)
    y64 = y.astype(np.float64)
    tol = em / sigma + BASE * np.abs(y64).mean(axis=1)
    err = np.abs(y64.mean(axis=1))
    viol = err > tol
    if viol.any():
        i = int(np.argmax(err - tol))
        return False, "row %d: |mean(y)|=%.3e > tol %.3e" % (i, err[i], tol[i])
    return True, "ok"


def proportional_nonneg(x, y):
    # y minus its mean must be k*(x - mean(x)) with k >= 0, up to rounding.
    bad = _struct(x, y)
    if bad:
        return False, bad
    d, v, s, em, dv, sigma = _row_stats(x)
    y64 = y.astype(np.float64)
    yc = y64 - y64.mean(axis=1, keepdims=True)
    ymax = np.abs(y64).max(axis=1)
    tol = em / sigma + BASE * ymax
    dd = (d * d).sum(axis=1)
    dmax = np.abs(d).max(axis=1)
    use = (dd > 0) & (dmax > 2.0 * EPS32 * s)
    kk = np.where(use, (yc * d).sum(axis=1) / np.where(use, dd, 1.0), 0.0)
    r = np.abs(yc - kk[:, None] * d).max(axis=1)
    viol = r > tol
    if viol.any():
        i = int(np.argmax(r - tol))
        return False, "row %d: not proportional to x-mean, residual %.3e > tol %.3e" % (i, r[i], tol[i])
    neg = -np.minimum(kk, 0.0) * dmax
    viol = neg > tol
    if viol.any():
        i = int(np.argmax(neg - tol))
        return False, "row %d: negative scale, k=%.3e" % (i, kk[i])
    return True, "ok"


def mean_square(x, y):
    # mean(y^2) = v/(v+eps): the row has (almost) unit second moment; the 1e-5 only matters for small v.
    bad = _struct(x, y)
    if bad:
        return False, bad
    d, v, s, em, dv, sigma = _row_stats(x)
    y64 = y.astype(np.float64)
    ms = (y64 * y64).mean(axis=1)
    target = v / (v + LN_EPS)
    tol = 2.0 * (dv + em * em) / (sigma * sigma) + 2.0 * BASE
    err = np.abs(ms - target)
    viol = err > tol
    if viol.any():
        i = int(np.argmax(err - tol))
        return False, "row %d: mean(y^2)=%.6e, expected %.6e (tol %.2e)" % (i, ms[i], target[i], tol[i])
    return True, "ok"


def shift_invariant(x0, y0, x1, y1):
    for x, y in ((x0, y0), (x1, y1)):
        bad = _struct(x, y)
        if bad:
            return False, bad
    if x0.shape != x1.shape:
        return False, "input shapes differ"
    d0, v0, s0, em0, dv0, sg0 = _row_stats(x0)
    d1, v1, s1, em1, dv1, sg1 = _row_stats(x1)
    sigma = np.minimum(sg0, sg1)
    pd = np.abs(d1 - d0).max(axis=1)
    y0_64 = y0.astype(np.float64)
    y1_64 = y1.astype(np.float64)
    ymax = np.maximum(np.abs(y0_64).max(axis=1), np.abs(y1_64).max(axis=1))
    rel = np.abs(sg0 - sg1) / sigma
    tol = (em0 + em1 + 2.0 * pd) / sigma + ymax * (2.0 * BASE + rel + (dv0 + dv1) / (sigma * sigma))
    err = np.abs(y1_64 - y0_64).max(axis=1)
    viol = err > tol
    if viol.any():
        i = int(np.argmax(err - tol))
        return False, "row %d: |y1-y0|=%.3e > tol %.3e" % (i, err[i], tol[i])
    return True, "ok"


CASE_LAWS = {
    "shape_dtype_finite": shape_dtype_finite,
    "mean_zero": mean_zero,
    "proportional_nonneg": proportional_nonneg,
    "mean_square": mean_square,
}

SHIFT_LAWS = {"shift_invariant": shift_invariant}
