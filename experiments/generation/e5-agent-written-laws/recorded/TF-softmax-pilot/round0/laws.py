import numpy as np

EPS = 2.0 ** -23
RATIO_FLOOR = -60.0  # entries with x_i - max(x) below this are only upper-bounded


def _prep(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    if y.shape != x.shape:
        return None, "shape %s != %s" % (y.shape, x.shape)
    if y.dtype != np.float32:
        return None, "dtype %s != float32" % y.dtype
    if not np.all(np.isfinite(y)):
        return None, "non-finite output"
    return (x.astype(np.float64), y.astype(np.float64)), ""


def _sum_tol(cols):
    return 3e-5 + EPS * cols


def well_formed(x, y):
    p, msg = _prep(x, y)
    if p is None:
        return False, msg
    return True, "ok"


def nonneg_bounded(x, y):
    p, msg = _prep(x, y)
    if p is None:
        return False, msg
    _, y64 = p
    if np.any(y64 < 0):
        return False, "negative entry %g" % y64.min()
    hi = 1.0 + _sum_tol(y64.shape[1])
    if np.any(y64 > hi):
        return False, "entry above 1: %g" % y64.max()
    return True, "ok"


def sums_to_one(x, y):
    p, msg = _prep(x, y)
    if p is None:
        return False, msg
    _, y64 = p
    s = y64.sum(axis=1)
    err = np.abs(s - 1.0)
    tol = _sum_tol(y64.shape[1])
    if np.any(err > tol):
        r = int(np.argmax(err))
        return False, "row %d sums to %.9g" % (r, s[r])
    return True, "ok"


def ratio_law(x, y):
    # y_i / y_j == exp(x_i - x_j), checked in log space against the row's argmax entry
    p, msg = _prep(x, y)
    if p is None:
        return False, msg
    x64, y64 = p
    rows = np.arange(x64.shape[0])
    jm = np.argmax(x64, axis=1)
    dx = x64 - x64[rows, jm][:, None]
    ym = y64[rows, jm]
    if np.any(ym <= 0):
        return False, "largest-input entry is not positive"
    tol = 3e-5 + 5e-7 * np.abs(dx)
    with np.errstate(all="ignore"):
        mid = dx >= RATIO_FLOOR
        pos = y64 > 0
        safe = np.where(mid & pos, y64, 1.0)
        err = np.abs(np.log(safe / ym[:, None]) - dx)
        bad = mid & ((~pos) | (err > tol))
        ub = ym[:, None] * np.exp(np.minimum(dx, 0.0) + tol) + 1e-35
        bad |= (~mid) & (y64 > ub)
    if np.any(bad):
        r, c = np.unravel_index(int(np.argmax(bad)), bad.shape)
        return False, "row %d col %d: ratio to max entry off (x diff %.6g, y=%.6g, ymax=%.6g)" % (
            r, c, dx[r, c], y64[r, c], ym[r])
    return True, "ok"


CASE_LAWS = {
    "well_formed": well_formed,
    "nonneg_bounded": nonneg_bounded,
    "sums_to_one": sums_to_one,
    "ratio_law": ratio_law,
}

SHIFT_LAWS = {}
