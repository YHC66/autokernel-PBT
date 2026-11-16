import numpy as np

EPS = 2.0 ** -23
LN_EPS = 1e-5


def _valid(x, y):
    if not isinstance(y, np.ndarray):
        return "output is not an ndarray"
    if y.shape != x.shape:
        return "shape %s != %s" % (y.shape, x.shape)
    if y.dtype != np.float32:
        return "dtype %s != float32" % y.dtype
    if not np.all(np.isfinite(y)):
        return "non-finite output"
    return None


def _row_stats(x):
    # q: scale of the row's magnitude relative to its normalizer; used only to size tolerances.
    x64 = x.astype(np.float64)
    n = x.shape[1]
    v = x64.var(axis=1)
    s = np.sqrt(v + LN_EPS)
    q = np.abs(x64).max(axis=1) / s
    return x64, n, v, s, q


def _mean_tol(n, q, ymax):
    return EPS * q * (8.0 + 3.0 * np.sqrt(n)) + 8.0 * EPS * (1.0 + ymax)


def _var_tol(n, q):
    return EPS * (8.0 + 3.0 * np.sqrt(n)) * (1.0 + q * q / 8.0)


def _elem_tol(q, ymax, k):
    return k * EPS * (1.0 + ymax + q)


def shape_dtype_finite(x, y):
    err = _valid(x, y)
    if err:
        return False, err
    return True, "ok"


def zero_mean(x, y):
    err = _valid(x, y)
    if err:
        return False, err
    x64, n, v, s, q = _row_stats(x)
    y64 = y.astype(np.float64)
    ymax = np.abs(y64).max(axis=1)
    ym = np.abs(y64.mean(axis=1))
    tol = _mean_tol(n, q, ymax)
    bad = ym > tol
    if bad.any():
        i = int(np.argmax(ym - tol))
        return False, "row %d mean(y)=%.3g > tol %.3g" % (i, ym[i], tol[i])
    return True, "ok"


def second_moment(x, y):
    err = _valid(x, y)
    if err:
        return False, err
    x64, n, v, s, q = _row_stats(x)
    y64 = y.astype(np.float64)
    ymax = np.abs(y64).max(axis=1)
    ms = (y64 * y64).mean(axis=1)
    target = v / (v + LN_EPS)
    d = _mean_tol(n, q, ymax)
    tol = 1e-5 + _var_tol(n, q) + d * d
    dev = np.abs(ms - target)
    bad = dev > tol
    if bad.any():
        i = int(np.argmax(dev - tol))
        return False, "row %d mean(y^2)=%.6g expected %.6g (tol %.3g)" % (i, ms[i], target[i], tol[i])
    return True, "ok"


def order_preserving(x, y):
    err = _valid(x, y)
    if err:
        return False, err
    if x.shape[1] < 2:
        return True, "ok"
    x64, n, v, s, q = _row_stats(x)
    y64 = y.astype(np.float64)
    ymax = np.abs(y64).max(axis=1)
    idx = np.argsort(x64, axis=1, kind="stable")
    ys = np.take_along_axis(y64, idx, axis=1)
    drop = (ys[:, :-1] - ys[:, 1:]).max(axis=1)
    tol = _elem_tol(q, ymax, 16.0)
    bad = drop > tol
    if bad.any():
        i = int(np.argmax(drop - tol))
        return False, "row %d y decreases by %.3g while x increases (tol %.3g)" % (i, drop[i], tol[i])
    return True, "ok"


def row_affine(x, y):
    err = _valid(x, y)
    if err:
        return False, err
    if x.shape[1] < 3:
        return True, "ok"
    x64, n, v, s, q = _row_stats(x)
    y64 = y.astype(np.float64)
    ymax = np.abs(y64).max(axis=1)
    r = np.arange(x.shape[0])
    imin = np.argmin(x64, axis=1)
    imax = np.argmax(x64, axis=1)
    xmin, xmax = x64[r, imin], x64[r, imax]
    ymin_, ymax_ = y64[r, imin], y64[r, imax]
    dx = xmax - xmin
    ok_rows = dx > 0
    safe = np.where(ok_rows, dx, 1.0)
    a = (ymax_ - ymin_) / safe
    pred = ymin_[:, None] + a[:, None] * (x64 - xmin[:, None])
    dev = np.abs(y64 - pred).max(axis=1)
    tol = _elem_tol(q, ymax, 32.0)
    bad = ok_rows & (dev > tol)
    if bad.any():
        i = int(np.argmax(np.where(ok_rows, dev - tol, -np.inf)))
        return False, "row %d y not affine in x: dev %.3g (tol %.3g)" % (i, dev[i], tol[i])
    return True, "ok"


def shift_invariance(x0, y0, x1, y1):
    for x, y in ((x0, y0), (x1, y1)):
        err = _valid(x, y)
        if err:
            return False, err
    if x0.shape != x1.shape:
        return False, "shift pair shapes differ"
    _, n, v0, s0, q0 = _row_stats(x0)
    _, _, v1, s1, q1 = _row_stats(x1)
    a0 = y0.astype(np.float64)
    a1 = y1.astype(np.float64)
    ymax = np.maximum(np.abs(a0).max(axis=1), np.abs(a1).max(axis=1))
    qm = np.maximum(q0, q1)
    tol = (
        _mean_tol(n, q0, ymax)
        + _mean_tol(n, q1, ymax)
        + (_var_tol(n, q0) + _var_tol(n, q1)) * ymax
        + _elem_tol(qm, ymax, 16.0)
    )
    dev = np.abs(a1 - a0).max(axis=1)
    bad = dev > tol
    if bad.any():
        i = int(np.argmax(dev - tol))
        return False, "row %d output changed by %.3g under shift (tol %.3g)" % (i, dev[i], tol[i])
    return True, "ok"


CASE_LAWS = {
    "shape_dtype_finite": shape_dtype_finite,
    "zero_mean": zero_mean,
    "second_moment": second_moment,
    "order_preserving": order_preserving,
    "row_affine": row_affine,
}

SHIFT_LAWS = {"shift_invariance": shift_invariance}
