import numpy as np

EPS = 2.0 ** -23
D_CUT = 40.0       # elements further than this below the row max are "negligible" (exp(-40) ~ 4e-18)
Y_FLOOR = 1e-25    # any element within D_CUT of the max must be at least this (true value >= ~5e-22)
Y_NEG = 1e-16      # any element beyond D_CUT must be at most this (true value <= ~4.2e-18)


def _sum_tol(n):
    return 1e-5 + 2.0 * EPS * n ** 0.75


def _bad(x, y):
    if not isinstance(y, np.ndarray):
        return "output is not an ndarray"
    if y.dtype != np.float32:
        return "dtype %s, expected float32" % y.dtype
    if y.shape != x.shape:
        return "shape %s, expected %s" % (y.shape, x.shape)
    if not np.all(np.isfinite(y)):
        return "non-finite values in output"
    return None


def structure_law(x, y):
    msg = _bad(x, y)
    if msg:
        return False, msg
    if y.min() < 0:
        return False, "negative probability %g" % float(y.min())
    n = x.shape[1]
    if y.max() > 1.0 + _sum_tol(n):
        return False, "value above 1: %g" % float(y.max())
    return True, "ok"


def row_sum_law(x, y):
    msg = _bad(x, y)
    if msg:
        return False, msg
    n = x.shape[1]
    s = y.astype(np.float64).sum(axis=1)
    err = float(np.max(np.abs(s - 1.0)))
    tol = _sum_tol(n)
    if err > tol:
        return False, "row sum off by %g (tol %g)" % (err, tol)
    return True, "ok"


def log_slope_law(x, y):
    # log y - x must be constant along each row (slope exactly 1 in x)
    msg = _bad(x, y)
    if msg:
        return False, msg
    x64 = x.astype(np.float64)
    y64 = y.astype(np.float64)
    xmax = x64.max(axis=1, keepdims=True)
    d = xmax - x64
    near = d <= D_CUT
    if np.any(near & (y64 < Y_FLOOR)):
        return False, "element within %g of row max is (near) zero" % D_CUT
    if np.any(~near & (y64 > Y_NEG)):
        return False, "element far below row max is not negligible"
    ref = np.argmax(x64, axis=1)
    rows = np.arange(x.shape[0])
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.log(np.where(near, y64, 1.0)) - x64
    rref = r[rows, ref][:, None]
    dev = np.abs(r - rref)
    bound = 2e-5 + 8.0 * EPS * d
    viol = near & (dev > bound)
    if viol.any():
        i, j = np.argwhere(viol)[0]
        return False, "log y - x not constant in row %d: dev %g > %g at col %d" % (
            i, dev[i, j], bound[i, j], j)
    return True, "ok"


def order_law(x, y):
    # larger input never gets a (noticeably) smaller output within a row
    msg = _bad(x, y)
    if msg:
        return False, msg
    order = np.argsort(x, axis=1, kind="stable")
    xs = np.take_along_axis(x, order, axis=1).astype(np.float64)
    ys = np.take_along_axis(y, order, axis=1).astype(np.float64)
    if xs.shape[1] < 2:
        return True, "ok"
    d = xs[:, -1:] - xs
    lo = ys[:, :-1]
    hi = ys[:, 1:]
    tol = 1e-30 + (2e-5 + 8.0 * EPS * d[:, :-1]) * hi
    viol = lo > hi + tol
    if viol.any():
        i, j = np.argwhere(viol)[0]
        return False, "not monotone in row %d at sorted position %d" % (i, j)
    return True, "ok"


def shift_invariance_law(x0, y0, x1, y1):
    msg = _bad(x0, y0) or _bad(x1, y1)
    if msg:
        return False, msg
    if x0.shape != x1.shape:
        return False, "case shapes differ"
    n = x0.shape[1]
    a = x0.astype(np.float64)
    b = x1.astype(np.float64)
    diff = b - a
    c = 0.5 * (diff.max(axis=1, keepdims=True) + diff.min(axis=1, keepdims=True))
    m = np.abs(diff - c).max(axis=1, keepdims=True)  # float32 rounding of the shifted input
    d0 = a.max(axis=1, keepdims=True) - a
    d1 = b.max(axis=1, keepdims=True) - b
    p = y0.astype(np.float64)
    q = y1.astype(np.float64)
    near = (d0 <= D_CUT) & (d1 <= D_CUT)
    if np.any(near & ((p < Y_FLOOR) | (q < Y_FLOOR))):
        return False, "element near row max is (near) zero in one of the cases"
    far = (d0 > D_CUT + 1.0) & (d1 > D_CUT + 1.0)
    if np.any(far & ((p > Y_NEG) | (q > Y_NEG))):
        return False, "far element not negligible"
    with np.errstate(divide="ignore", invalid="ignore"):
        lr = np.abs(np.log(np.where(near, q, 1.0)) - np.log(np.where(near, p, 1.0)))
    bound = (2.0 * m + 4e-5 + 8.0 * EPS * (d0 + d1) + 4.0 * EPS * n ** 0.75)
    viol = near & (lr > bound)
    if viol.any():
        i, j = np.argwhere(viol)[0]
        return False, "output changed under row shift: row %d col %d, log ratio %g > %g" % (
            i, j, lr[i, j], bound[i, j])
    return True, "ok"


CASE_LAWS = {
    "structure": structure_law,
    "row_sum": row_sum_law,
    "log_slope": log_slope_law,
    "order": order_law,
}
SHIFT_LAWS = {"shift_invariance": shift_invariance_law}
