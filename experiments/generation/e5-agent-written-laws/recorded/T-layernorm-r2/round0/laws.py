import numpy as np

EPS32 = 2.0 ** -23
EPS = 1e-5


def _prep(x, y):
    """Return (err, x64, y64, tol). tol bounds |y - exact| per element for a correct float32 kernel."""
    if not isinstance(y, np.ndarray):
        return "output is not an ndarray", None, None, None
    if y.shape != x.shape:
        return "shape %s != %s" % (y.shape, x.shape), None, None, None
    if y.dtype != np.float32:
        return "dtype %s != float32" % y.dtype, None, None, None
    if not np.all(np.isfinite(y)):
        return "non-finite output", None, None, None
    x64 = x.astype(np.float64)
    y64 = y.astype(np.float64)
    n = x64.shape[1]
    m = x64.mean(axis=1)
    v = x64.var(axis=1)
    s = np.sqrt(v + EPS)
    ax = np.abs(x64)
    mabs = ax.mean(axis=1)
    rmax = ax.max(axis=1)
    w = (v + m * m) / (v + EPS)
    a = EPS32 * (8.0 + np.sqrt(n))
    ay = np.abs(y64)
    col = lambda t: t[:, None]
    tol = 4.0 * (
        col(a * mabs / s)
        + 2.0 * EPS32 * ax / col(s)
        + 0.5 * a * col(w) * ay
        + EPS32 * col(rmax / s) * ay
        + 8.0 * EPS32 * (1.0 + ay)
    )
    return None, x64, y64, tol


def shape_dtype_finite(x, y):
    if not isinstance(y, np.ndarray):
        return False, "output is not an ndarray"
    if y.shape != x.shape:
        return False, "shape %s != %s" % (y.shape, x.shape)
    if y.dtype != np.float32:
        return False, "dtype %s != float32" % y.dtype
    if not np.all(np.isfinite(y)):
        return False, "non-finite output"
    return True, "ok"


def zero_mean(x, y):
    err, x64, y64, tol = _prep(x, y)
    if err:
        return False, err
    dev = np.abs(y64.mean(axis=1))
    lim = tol.mean(axis=1)
    bad = dev > lim
    if bad.any():
        i = int(np.argmax(dev - lim))
        return False, "row %d mean(y)=%g exceeds %g" % (i, dev[i], lim[i])
    return True, "ok"


def unit_second_moment(x, y):
    # mean(y^2) must equal v/(v+1e-5), v = population variance of the row of x
    err, x64, y64, tol = _prep(x, y)
    if err:
        return False, err
    v = x64.var(axis=1)
    target = v / (v + EPS)
    got = (y64 * y64).mean(axis=1)
    lim = (2.0 * np.abs(y64) * tol + tol * tol).mean(axis=1)
    dev = np.abs(got - target)
    bad = dev > lim
    if bad.any():
        i = int(np.argmax(dev - lim))
        return False, "row %d mean(y^2)=%g, expected %g (tol %g)" % (i, got[i], target[i], lim[i])
    return True, "ok"


def affine_in_x(x, y):
    # each output row is an affine function of the input row
    err, x64, y64, tol = _prep(x, y)
    if err:
        return False, err
    r = np.arange(x64.shape[0])
    imin = np.argmin(x64, axis=1)
    imax = np.argmax(x64, axis=1)
    xlo, xhi = x64[r, imin], x64[r, imax]
    ylo, yhi = y64[r, imin], y64[r, imax]
    tlo, thi = tol[r, imin], tol[r, imax]
    rng = xhi - xlo
    safe = np.where(rng > 0, rng, 1.0)
    lam = np.where(rng[:, None] > 0, (x64 - xlo[:, None]) / safe[:, None], 0.0)
    pred = ylo[:, None] * (1.0 - lam) + yhi[:, None] * lam
    lim = tol + (1.0 - lam) * tlo[:, None] + lam * thi[:, None]
    dev = np.abs(y64 - pred)
    bad = dev > lim
    if bad.any():
        k = int(np.argmax((dev - lim).max(axis=1)))
        j = int(np.argmax(dev[k] - lim[k]))
        return False, "row %d col %d deviates from affine by %g (tol %g)" % (k, j, dev[k, j], lim[k, j])
    return True, "ok"


def monotone(x, y):
    # larger input gives larger-or-equal output within each row
    err, x64, y64, tol = _prep(x, y)
    if err:
        return False, err
    if x64.shape[1] < 2:
        return True, "ok"
    order = np.argsort(x64, axis=1, kind="stable")
    ys = np.take_along_axis(y64, order, axis=1)
    ts = np.take_along_axis(tol, order, axis=1)
    d = ys[:, 1:] - ys[:, :-1]
    lim = ts[:, 1:] + ts[:, :-1]
    bad = d < -lim
    if bad.any():
        k = int(np.argmax((-d - lim).max(axis=1)))
        return False, "row %d output decreases with input" % k
    return True, "ok"


def shift_invariance(x0, y0, x1, y1):
    # adding a per-row constant to x leaves the output unchanged
    e0, _, a, t0 = _prep(x0, y0)
    if e0:
        return False, e0
    e1, _, b, t1 = _prep(x1, y1)
    if e1:
        return False, e1
    dev = np.abs(a - b)
    lim = t0 + t1
    bad = dev > lim
    if bad.any():
        k = int(np.argmax((dev - lim).max(axis=1)))
        j = int(np.argmax(dev[k] - lim[k]))
        return False, "row %d col %d changed by %g under shift (tol %g)" % (k, j, dev[k, j], lim[k, j])
    return True, "ok"


CASE_LAWS = {
    "shape_dtype_finite": shape_dtype_finite,
    "zero_mean": zero_mean,
    "unit_second_moment": unit_second_moment,
    "affine_in_x": affine_in_x,
    "monotone": monotone,
}
SHIFT_LAWS = {"shift_invariance": shift_invariance}
