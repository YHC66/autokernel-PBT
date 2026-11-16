import numpy as np

EPS = 2.0 ** -23


def _prep(x, y):
    x = np.asarray(x)
    try:
        y = np.asarray(y)
    except Exception as e:
        return None, None, "y not array-like: %s" % e
    if x.ndim != 2:
        return None, None, "x is not 2-D"
    if y.shape != x.shape:
        return None, None, "shape %s != %s" % (y.shape, x.shape)
    if y.dtype != np.float32:
        return None, None, "dtype %s != float32" % y.dtype
    y64 = y.astype(np.float64)
    if not np.all(np.isfinite(y64)):
        return None, None, "non-finite output"
    return x.astype(np.float64), y64, None


def shape_dtype_finite(x, y):
    _, _, err = _prep(x, y)
    if err:
        return False, err
    return True, "ok"


def in_unit_interval(x, y):
    _, y64, err = _prep(x, y)
    if err:
        return False, err
    lo, hi = float(y64.min()), float(y64.max())
    if lo < 0.0:
        return False, "negative output %g" % lo
    if hi > 1.0 + 1e-5:
        return False, "output above 1: %g" % hi
    return True, "ok"


def rows_sum_to_one(x, y):
    _, y64, err = _prep(x, y)
    if err:
        return False, err
    cols = y64.shape[1]
    tol = 2e-5 + 5e-7 * np.sqrt(cols)
    dev = np.abs(y64.sum(axis=1) - 1.0)
    i = int(np.argmax(dev))
    if dev[i] > tol:
        return False, "row %d sums to %.8g (dev %.3g > %.3g)" % (i, 1.0 + (y64[i].sum() - 1.0), dev[i], tol)
    return True, "ok"


def order_preserved(x, y):
    x64, y64, err = _prep(x, y)
    if err:
        return False, err
    rows, cols = x64.shape
    if cols < 2:
        return True, "ok"
    idx = np.argsort(x64, axis=1, kind="stable")
    ys = np.take_along_axis(y64, idx, axis=1)
    xs = np.take_along_axis(x64, idx, axis=1)
    r = np.minimum(xs[:, -1] - xs[:, 0], 100.0)
    tol = 3e-5 + 4e-7 * r + 1e-6 * np.sqrt(cols)
    lower = ys[:, :-1]
    upper = ys[:, 1:]
    bad = lower > upper * (1.0 + tol[:, None]) + 1e-30
    if bad.any():
        i, k = np.argwhere(bad)[0]
        return False, "row %d: larger input has smaller output (%.8g > %.8g)" % (i, lower[i, k], upper[i, k])
    return True, "ok"


def log_ratio_matches_input(x, y):
    # log y - x must be the same constant across a row (y_j / y_k = exp(x_j - x_k))
    x64, y64, err = _prep(x, y)
    if err:
        return False, err
    rows, cols = x64.shape
    valid = y64 > 1e-30
    if not valid.any(axis=1).all():
        return False, "row with no non-negligible output"
    d = np.log(np.where(valid, y64, 1.0)) - x64
    dmax = np.where(valid, d, -np.inf).max(axis=1)
    dmin = np.where(valid, d, np.inf).min(axis=1)
    xmax = np.where(valid, x64, -np.inf).max(axis=1)
    xmin = np.where(valid, x64, np.inf).min(axis=1)
    r = np.minimum(xmax - xmin, 100.0)
    tol = 3e-5 + 4e-7 * r + 1e-6 * np.sqrt(cols)
    spread = dmax - dmin
    i = int(np.argmax(spread - tol))
    if spread[i] > tol[i]:
        return False, "row %d: log-ratio spread %.3g > %.3g" % (i, spread[i], tol[i])
    return True, "ok"


def shift_invariance(x0, y0, x1, y1):
    xa, ya, err = _prep(x0, y0)
    if err:
        return False, "case0: " + err
    xb, yb, err = _prep(x1, y1)
    if err:
        return False, "case1: " + err
    if xa.shape != xb.shape:
        return False, "x shapes differ"
    rows, cols = xa.shape
    e = xb - xa
    s = e.max(axis=1) - e.min(axis=1)
    r = np.minimum(xa.max(axis=1) - xa.min(axis=1), 100.0)
    tol = 5e-5 + 4e-7 * r + 5e-7 * np.sqrt(cols) + 2.0 * s
    mask = np.maximum(ya, yb) > 1e-20
    diff = np.abs(ya - yb)
    allowed = tol[:, None] * np.maximum(ya, yb)
    bad = mask & (diff > allowed)
    if bad.any():
        i, j = np.argwhere(bad)[0]
        return False, "row %d col %d: %.8g vs %.8g after shift" % (i, j, ya[i, j], yb[i, j])
    return True, "ok"


CASE_LAWS = {
    "shape_dtype_finite": shape_dtype_finite,
    "in_unit_interval": in_unit_interval,
    "rows_sum_to_one": rows_sum_to_one,
    "order_preserved": order_preserved,
    "log_ratio_matches_input": log_ratio_matches_input,
}

SHIFT_LAWS = {"shift_invariance": shift_invariance}
