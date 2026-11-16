import numpy as np

EPS = 2.0 ** -23
TINY = 1e-37  # absolute slack for denormal / flush-to-zero outputs


def _valid(x, y):
    if not isinstance(y, np.ndarray):
        return False, "output is not an ndarray"
    if y.shape != np.shape(x):
        return False, f"shape {y.shape} != {np.shape(x)}"
    if y.dtype != np.float32:
        return False, f"dtype {y.dtype} != float32"
    if not np.all(np.isfinite(y)):
        return False, "non-finite output"
    return None


def shape_dtype_finite(x, y):
    r = _valid(x, y)
    return r if r else (True, "ok")


def nonneg_and_bounded(x, y):
    r = _valid(x, y)
    if r:
        return r
    cols = np.shape(x)[1]
    hi = 1.0 + 3e-5 + cols * EPS
    if y.min() < 0:
        return False, f"negative entry {float(y.min())}"
    if y.max() > hi:
        return False, f"entry {float(y.max())} > 1"
    return True, "ok"


def rows_sum_to_one(x, y):
    r = _valid(x, y)
    if r:
        return r
    cols = np.shape(x)[1]
    tol = 3e-5 + cols * EPS
    s = y.astype(np.float64).sum(axis=1)
    err = np.abs(s - 1.0)
    k = int(np.argmax(err))
    if err[k] > tol:
        return False, f"row {k} sums to {s[k]!r}, tol {tol:.3g}"
    return True, "ok"


def ratio_to_max(x, y):
    # y_j / y_m == exp(x_j - x_m) for m = argmax of the row (no normaliser involved).
    r = _valid(x, y)
    if r:
        return r
    x64 = np.asarray(x, dtype=np.float64)
    y64 = y.astype(np.float64)
    rows = np.arange(x64.shape[0])
    m = np.argmax(x64, axis=1)
    t = x64 - x64[rows, m][:, None]  # <= 0
    ym = y64[rows, m][:, None]
    pred = ym * np.exp(t)
    rtol = 1e-5 + 8.0 * EPS * np.minimum(np.abs(t), 100.0)
    err = np.abs(y64 - pred) - (rtol * pred + TINY)
    k = np.unravel_index(int(np.argmax(err)), err.shape)
    if err[k] > 0:
        return False, f"at {k}: y={y64[k]!r} expected {pred[k]!r} from pivot"
    return True, "ok"


def shift_invariance(x0, y0, x1, y1):
    r = _valid(x0, y0) or _valid(x1, y1)
    if r:
        return r
    a = np.asarray(x0, dtype=np.float64)
    b = np.asarray(x1, dtype=np.float64)
    ya = y0.astype(np.float64)
    yb = y1.astype(np.float64)
    rows = np.arange(a.shape[0])
    m = np.argmax(a, axis=1)
    ma = ya[rows, m][:, None]
    mb = yb[rows, m][:, None]
    if np.any(ma <= 0) or np.any(mb <= 0):
        return False, "pivot probability is not positive"
    t = np.minimum(a[rows, m][:, None] - a, 100.0)
    # float32 rounding of x0 + c perturbs the effective differences
    rnd = 3e-7 * (np.abs(a).max(axis=1) + np.abs(b).max(axis=1))[:, None]
    rtol = 2e-5 + 16.0 * EPS * t + rnd
    u = ya * mb
    v = yb * ma
    err = np.abs(u - v) - (rtol * np.maximum(u, v) + TINY * (ma + mb))
    k = np.unravel_index(int(np.argmax(err)), err.shape)
    if err[k] > 0:
        return False, f"at {k}: y0={ya[k]!r} y1={yb[k]!r} differ beyond shift tolerance"
    return True, "ok"


CASE_LAWS = {
    "shape_dtype_finite": shape_dtype_finite,
    "nonneg_and_bounded": nonneg_and_bounded,
    "rows_sum_to_one": rows_sum_to_one,
    "ratio_to_max": ratio_to_max,
}
SHIFT_LAWS = {"shift_invariance": shift_invariance}
