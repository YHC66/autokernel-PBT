import numpy as np

EPS = 2.0 ** -23
TINY = 2e-38


def _structure_problem(x, y):
    if not isinstance(y, np.ndarray):
        return "output is not an ndarray"
    if y.dtype != np.float32:
        return "output dtype is %s, expected float32" % y.dtype
    if y.shape != x.shape:
        return "output shape %s, expected %s" % (y.shape, x.shape)
    if not np.all(np.isfinite(y)):
        return "output contains NaN or inf"
    return None


def shape_dtype_finite(x, y):
    p = _structure_problem(x, y)
    if p is not None:
        return False, p
    return True, "ok"


def sums_to_one(x, y):
    p = _structure_problem(x, y)
    if p is not None:
        return False, p
    y64 = y.astype(np.float64)
    cols = x.shape[1]
    tol = 2e-5 + cols * EPS
    s = y64.sum(axis=1)
    err = np.abs(s - 1.0)
    worst = int(np.argmax(err))
    if err[worst] > tol:
        return False, "row %d sums to %.8g (tol %.3g)" % (worst, s[worst], tol)
    return True, "ok"


def bounded_in_unit_interval(x, y):
    p = _structure_problem(x, y)
    if p is not None:
        return False, p
    y64 = y.astype(np.float64)
    tol = 2e-5 + x.shape[1] * EPS
    if y64.min() < 0.0:
        return False, "negative output %.6g" % y64.min()
    if y64.max() > 1.0 + tol:
        return False, "output above 1: %.8g" % y64.max()
    return True, "ok"


def ratio_to_row_max(x, y):
    # y_i / y_max == exp(x_i - x_max), the same relation as y_i*exp(x_j) == y_j*exp(x_i)
    p = _structure_problem(x, y)
    if p is not None:
        return False, p
    x64 = x.astype(np.float64)
    y64 = y.astype(np.float64)
    rows = x.shape[0]
    idx = np.argmax(x64, axis=1)
    ar = np.arange(rows)
    d = x64 - x64[ar, idx][:, None]
    ymax = y64[ar, idx]
    r = ymax[:, None] * np.exp(d)
    tol = 2e-5 + 5e-7 * np.abs(d)
    bad = np.abs(y64 - r) > tol * r + TINY
    if np.any(bad):
        i, j = np.argwhere(bad)[0]
        return False, "row %d col %d: y=%.6g but y_max*exp(x-x_max)=%.6g" % (i, j, y64[i, j], r[i, j])
    return True, "ok"


def shift_invariance(x0, y0, x1, y1):
    p = _structure_problem(x0, y0) or _structure_problem(x1, y1)
    if p is not None:
        return False, p
    a = y0.astype(np.float64)
    b = y1.astype(np.float64)
    x0d = x0.astype(np.float64)
    x1d = x1.astype(np.float64)
    cols = x0.shape[1]
    delta = EPS * np.maximum(np.abs(x0d).max(axis=1), np.abs(x1d).max(axis=1))
    d0 = x0d - x0d.max(axis=1, keepdims=True)
    rel = 3e-5 + 2.0 * cols * EPS + 3.0 * delta[:, None] + 5e-7 * np.abs(d0)
    bad = np.abs(a - b) > rel * np.maximum(a, b) + 4 * TINY
    if np.any(bad):
        i, j = np.argwhere(bad)[0]
        return False, "row %d col %d: shifted output %.6g vs %.6g" % (i, j, b[i, j], a[i, j])
    return True, "ok"


CASE_LAWS = {
    "shape_dtype_finite": shape_dtype_finite,
    "sums_to_one": sums_to_one,
    "bounded_in_unit_interval": bounded_in_unit_interval,
    "ratio_to_row_max": ratio_to_row_max,
}
SHIFT_LAWS = {"shift_invariance": shift_invariance}
