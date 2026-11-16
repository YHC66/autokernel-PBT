import numpy as np

EPS = 2.0 ** -23
NEED_GAP = 50.0      # entries within this gap of the row max must be clearly nonzero
TINY = 1e-30         # below this, y may legitimately be flushed or denormal
BASE_TOL = 4e-6


def _structure(x, y):
    if not isinstance(y, np.ndarray):
        return "output is not an ndarray"
    if y.shape != np.shape(x):
        return "shape %s != %s" % (y.shape, np.shape(x))
    if y.dtype != np.float32:
        return "dtype %s != float32" % y.dtype
    if not np.isfinite(y).all():
        return "non-finite output"
    return None


def structure_law(x, y):
    bad = _structure(x, y)
    if bad:
        return False, bad
    return True, "shape, dtype, finiteness ok"


def range_law(x, y):
    bad = _structure(x, y)
    if bad:
        return False, bad
    y64 = y.astype(np.float64)
    if (y64 < 0).any():
        return False, "negative entry %g" % y64.min()
    if (y64 > 1.0 + 3e-5).any():
        return False, "entry above 1: %g" % y64.max()
    return True, "entries in [0, 1]"


def sum_law(x, y):
    bad = _structure(x, y)
    if bad:
        return False, bad
    cols = np.shape(x)[1]
    s = y.astype(np.float64).sum(axis=1)
    tol = 3e-5 + 2.0 * EPS * cols
    err = np.abs(s - 1.0).max()
    if err > tol:
        return False, "row sum off by %g (tol %g)" % (err, tol)
    return True, "rows sum to 1"


def _active_logs(x64, y64):
    gap = x64.max(axis=1, keepdims=True) - x64
    need = gap <= NEED_GAP
    active = y64 > TINY
    return gap, need, active


def ratio_law(x, y):
    # y_i / y_k = exp(x_i - x_k) for the row's argmax k, checked in the log domain.
    bad = _structure(x, y)
    if bad:
        return False, bad
    x64 = np.asarray(x, dtype=np.float64)
    y64 = y.astype(np.float64)
    gap, need, active = _active_logs(x64, y64)
    if (need & ~active).any():
        return False, "entry within %g of the row max is (near) zero" % NEED_GAP
    d = np.log(np.where(active, y64, 1.0)) - x64
    k = np.argmax(x64, axis=1)
    dk = d[np.arange(x64.shape[0]), k]
    tol = BASE_TOL + 8.0 * EPS * gap + BASE_TOL
    dev = np.abs(d - dk[:, None])
    badmask = active & (dev > tol)
    if badmask.any():
        return False, "log-ratio deviates by %g" % dev[badmask].max()
    return True, "ratios match exp(x_i - x_k)"


def shift_law(x0, y0, x1, y1):
    # log y1_i - log y0_i - (x1_i - x0_i) must not depend on i.
    for x, y in ((x0, y0), (x1, y1)):
        bad = _structure(x, y)
        if bad:
            return False, bad
    a0 = np.asarray(x0, dtype=np.float64)
    a1 = np.asarray(x1, dtype=np.float64)
    b0 = y0.astype(np.float64)
    b1 = y1.astype(np.float64)
    g0, _, act0 = _active_logs(a0, b0)
    g1, _, act1 = _active_logs(a1, b1)
    active = act0 & act1
    if not active.any(axis=1).all():
        return False, "no usable entries in a row"
    l0 = np.log(np.where(active, b0, 1.0))
    l1 = np.log(np.where(active, b1, 1.0))
    d = l1 - l0 - (a1 - a0)
    tol = BASE_TOL + 8.0 * EPS * (g0 + g1)
    score = np.where(active, l0 + l1, -np.inf)
    k = np.argmax(score, axis=1)
    rows = np.arange(a0.shape[0])
    dk = d[rows, k][:, None]
    tolk = tol[rows, k][:, None]
    dev = np.abs(d - dk)
    badmask = active & (dev > tol + tolk)
    if badmask.any():
        return False, "shift changes log-ratios by %g" % dev[badmask].max()
    return True, "softmax shifts consistently"


CASE_LAWS = {
    "structure": structure_law,
    "range": range_law,
    "sum_to_one": sum_law,
    "ratio": ratio_law,
}
SHIFT_LAWS = {"shift": shift_law}
