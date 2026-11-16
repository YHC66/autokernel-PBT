import numpy as np

EPS = 2.0 ** -23
EXP_TOL = 1e-4  # allowance for differing exp implementations


def _norm_tol(n):
    # float32 summation error bound (n-1)*u with u = 2**-24, any order, plus slack
    return 1e-5 + 6e-8 * n


def _check_form(x, y):
    if not isinstance(y, np.ndarray):
        return "output is not an ndarray"
    if y.shape != x.shape:
        return "shape %s != %s" % (y.shape, x.shape)
    if y.dtype != np.float32:
        return "dtype %s != float32" % y.dtype
    if not np.all(np.isfinite(y)):
        return "non-finite values in output"
    return None


def well_formed(x, y):
    err = _check_form(x, y)
    return (err is None), (err or "ok")


def value_range(x, y):
    err = _check_form(x, y)
    if err:
        return False, err
    y64 = y.astype(np.float64)
    if y64.min() < 0.0:
        return False, "negative probability %g" % y64.min()
    tol = _norm_tol(x.shape[1]) + EXP_TOL
    if y64.max() > 1.0 + tol:
        return False, "probability above 1: %g" % y64.max()
    return True, "ok"


def row_sums_one(x, y):
    err = _check_form(x, y)
    if err:
        return False, err
    s = y.astype(np.float64).sum(axis=1)
    tol = _norm_tol(x.shape[1]) + EXP_TOL
    dev = np.abs(s - 1.0)
    i = int(np.argmax(dev))
    if dev[i] > tol:
        return False, "row %d sums to %.8g (tol %.2g)" % (i, s[i], tol)
    return True, "ok"


def uniform_rows(x, y):
    err = _check_form(x, y)
    if err:
        return False, err
    x64 = x.astype(np.float64)
    n = x.shape[1]
    const = np.all(x64 == x64[:, :1], axis=1)
    if not const.any():
        return True, "no constant rows"
    rtol = _norm_tol(n) + EXP_TOL
    yy = y[const].astype(np.float64)
    dev = np.abs(yy * n - 1.0)
    if dev.max() > rtol:
        return False, "constant row not uniform, rel dev %.3g (tol %.2g)" % (dev.max(), rtol)
    return True, "ok"


def order_preserving(x, y):
    err = _check_form(x, y)
    if err:
        return False, err
    x64 = x.astype(np.float64)
    y64 = y.astype(np.float64)
    idx = np.argsort(x64, axis=1, kind="stable")
    ys = np.take_along_axis(y64, idx, axis=1)
    if ys.shape[1] < 2:
        return True, "ok"
    lo = ys[:, :-1]
    hi = ys[:, 1:]
    rtol = 2e-4 + EXP_TOL
    bad = hi < lo * (1.0 - rtol) - 1e-30
    if bad.any():
        r, k = np.argwhere(bad)[0]
        return False, "row %d: larger x gets smaller y (%.6g vs %.6g)" % (r, hi[r, k], lo[r, k])
    return True, "ok"


def exp_envelope(x, y):
    # with d = x - rowmax: exp(d)/n <= y <= exp(d), since 1 <= sum_k exp(d_k) <= n
    err = _check_form(x, y)
    if err:
        return False, err
    x64 = x.astype(np.float64)
    y64 = y.astype(np.float64)
    n = x.shape[1]
    d = x64 - x64.max(axis=1, keepdims=True)
    tol = _norm_tol(n) + EXP_TOL
    e = np.exp(np.maximum(d, -700.0))
    up = e * (1.0 + tol) + 1e-30
    if np.any(y64 > up):
        r, k = np.argwhere(y64 > up)[0]
        return False, "y[%d,%d]=%.6g exceeds exp(x-max)=%.6g" % (r, k, y64[r, k], e[r, k])
    sel = d >= -60.0
    low = e / n * (1.0 - tol)
    bad = sel & (y64 < low)
    if bad.any():
        r, k = np.argwhere(bad)[0]
        return False, "y[%d,%d]=%.6g below exp(x-max)/n=%.6g" % (r, k, y64[r, k], e[r, k] / n)
    return True, "ok"


def log_ratio_constant(x, y):
    # log(y_i) - x_i is the same for every i in a row (= -log of the row's normalizer)
    err = _check_form(x, y)
    if err:
        return False, err
    x64 = x.astype(np.float64)
    y64 = y.astype(np.float64)
    d = x64 - x64.max(axis=1, keepdims=True)
    sel = d >= -25.0
    if np.any(sel & (y64 <= 0.0)):
        return False, "zero probability where x is within 25 of row max"
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(sel, np.log(np.where(sel, y64, 1.0)) - d, np.nan)
    hi = np.nanmax(np.where(sel, r, -np.inf), axis=1)
    lo = np.nanmin(np.where(sel, r, np.inf), axis=1)
    spread = hi - lo
    spread = np.where(np.isfinite(spread), spread, 0.0)
    tol = 2e-4
    i = int(np.argmax(spread))
    if spread[i] > tol:
        return False, "row %d: log(y)-x not constant, spread %.3g (tol %.2g)" % (i, spread[i], tol)
    return True, "ok"


def shift_invariance(x0, y0, x1, y1):
    e0 = _check_form(x0, y0)
    if e0:
        return False, e0
    e1 = _check_form(x1, y1)
    if e1:
        return False, e1
    if x0.shape != x1.shape:
        return False, "shape mismatch between shifted cases"
    n = x0.shape[1]
    m = np.maximum(np.abs(x0.astype(np.float64)).max(axis=1), np.abs(x1.astype(np.float64)).max(axis=1))
    rtol = (1e-4 + 2.0 * _norm_tol(n) + 4.0 * EPS * m)[:, None]
    a = y0.astype(np.float64)
    b = y1.astype(np.float64)
    bad = np.abs(a - b) > rtol * np.maximum(a, b) + 1e-12
    if bad.any():
        r, k = np.argwhere(bad)[0]
        return False, "row %d col %d: y changed under shift (%.6g vs %.6g)" % (r, k, a[r, k], b[r, k])
    return True, "ok"


CASE_LAWS = {
    "well_formed": well_formed,
    "value_range": value_range,
    "row_sums_one": row_sums_one,
    "uniform_rows": uniform_rows,
    "order_preserving": order_preserving,
    "exp_envelope": exp_envelope,
    "log_ratio_constant": log_ratio_constant,
}

SHIFT_LAWS = {
    "shift_invariance": shift_invariance,
}
