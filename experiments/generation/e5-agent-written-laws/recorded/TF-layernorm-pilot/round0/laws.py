import numpy as np

EPS = 2.0 ** -23
VAR_EPS = 1e-5


def _valid(x, y):
    if not isinstance(y, np.ndarray):
        return False, "output is not an ndarray"
    if y.shape != x.shape:
        return False, "shape %s != %s" % (y.shape, x.shape)
    if y.dtype != np.float32:
        return False, "dtype %s != float32" % y.dtype
    if not np.all(np.isfinite(y)):
        return False, "non-finite output"
    return True, ""


def _stats(x):
    # Per-row statistics of x in float64 and a bound B on max_j |y_j - y*_j|, where y* is the
    # exact (eps = 1e-5) layernorm row. B models float32 rounding in the mean, the variance
    # (including one-pass formulas), the sqrt and the division; it grows with n and with the
    # row's value range. It is used only as a tolerance, y* is never formed.
    x64 = x.astype(np.float64)
    n = x64.shape[1]
    root = np.sqrt(float(n))
    m = x64.mean(axis=1, keepdims=True)
    d = x64 - m
    v = (d * d).mean(axis=1)
    a = np.abs(x64).max(axis=1)
    dmax = np.abs(d).max(axis=1)
    s = np.sqrt(v + VAR_EPS)
    em = EPS * a * (8.0 + 4.0 * root)
    dv = EPS * a * a * (8.0 + 4.0 * root) + em * em + EPS * v * (16.0 + 4.0 * root)
    r = dv / (s * s)
    rel = r / 2.0 + r * r
    B = ((em + 4.0 * EPS * dmax) / s) * (1.0 + rel) + (dmax / s) * (rel + 16.0 * EPS)
    return dict(x64=x64, n=n, d=d, v=v, s=s, B=B)


def shape_dtype_finite(x, y):
    return _valid(x, y)


def mean_zero(x, y):
    ok, why = _valid(x, y)
    if not ok:
        return False, why
    st = _stats(x)
    mu = np.abs(y.astype(np.float64).mean(axis=1))
    bad = mu > st["B"]
    if bad.any():
        i = int(np.argmax(mu - st["B"]))
        return False, "row %d: |mean(y)|=%.3g > tol %.3g" % (i, mu[i], st["B"][i])
    return True, "ok"


def mean_square(x, y):
    # mean(y^2) = v / (v + 1e-5): equals 1 when the variance is not small
    ok, why = _valid(x, y)
    if not ok:
        return False, why
    st = _stats(x)
    y64 = y.astype(np.float64)
    ms = (y64 * y64).mean(axis=1)
    target = st["v"] / (st["v"] + VAR_EPS)
    B = st["B"]
    tol = 2.0 * B + B * B
    err = np.abs(ms - target)
    if (err > tol).any():
        i = int(np.argmax(err - tol))
        return False, "row %d: mean(y^2)=%.6g, expected %.6g, tol %.3g" % (i, ms[i], target[i], tol[i])
    return True, "ok"


def nonneg_multiple(x, y):
    # y is a non-negative multiple of (x - mean): small distance to the line through d,
    # and non-negative correlation with d
    ok, why = _valid(x, y)
    if not ok:
        return False, why
    st = _stats(x)
    d = st["d"]
    n = st["n"]
    B = st["B"]
    y64 = y.astype(np.float64)
    dd = (d * d).sum(axis=1)
    dy = (d * y64).sum(axis=1)
    coef = np.where(dd > 0, dy / np.where(dd > 0, dd, 1.0), 0.0)
    res = y64 - coef[:, None] * d
    rms = np.sqrt((res * res).mean(axis=1))
    if (rms > B).any():
        i = int(np.argmax(rms - B))
        return False, "row %d: y not proportional to x-mean (rms residual %.3g > tol %.3g)" % (i, rms[i], B[i])
    slack = np.sqrt(dd) * np.sqrt(float(n)) * B
    if (dy < -slack).any():
        i = int(np.argmax(-dy - slack))
        return False, "row %d: y anti-correlated with x-mean (%.3g < -%.3g)" % (i, dy[i], slack[i])
    return True, "ok"


def shift_invariance(x0, y0, x1, y1):
    ok, why = _valid(x0, y0)
    if not ok:
        return False, "case 0: " + why
    ok, why = _valid(x1, y1)
    if not ok:
        return False, "case 1: " + why
    if x0.shape != x1.shape:
        return False, "shift pair has different shapes"
    s0 = _stats(x0)
    s1 = _stats(x1)
    delta = x1.astype(np.float64) - x0.astype(np.float64)
    p = np.abs(delta - delta.mean(axis=1, keepdims=True)).max(axis=1)
    root = np.sqrt(float(s0["n"]))
    extra = 2.0 * (2.0 + root) * p / np.minimum(s0["s"], s1["s"])
    tol = s0["B"] + s1["B"] + extra
    diff = np.abs(y1.astype(np.float64) - y0.astype(np.float64)).max(axis=1)
    if (diff > tol).any():
        i = int(np.argmax(diff - tol))
        return False, "row %d: |y(x+c)-y(x)|=%.3g > tol %.3g" % (i, diff[i], tol[i])
    return True, "ok"


CASE_LAWS = {
    "shape_dtype_finite": shape_dtype_finite,
    "mean_zero": mean_zero,
    "mean_square": mean_square,
    "nonneg_multiple": nonneg_multiple,
}
SHIFT_LAWS = {"shift_invariance": shift_invariance}
