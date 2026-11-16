import numpy as np

TINY = 1e-30  # entries below this may legitimately underflow / flush to zero


def _prep(x, y):
    if not isinstance(y, np.ndarray):
        return None, "output is not an ndarray"
    if y.shape != x.shape:
        return None, f"shape {y.shape} != {x.shape}"
    if y.dtype != np.float32:
        return None, f"dtype {y.dtype} != float32"
    if not np.all(np.isfinite(y)):
        return None, "non-finite values in output"
    return y.astype(np.float64), None


def _sum_tol(cols):
    return 2e-6 + 3e-8 * cols


def _exp_tol(d):
    # tolerance on log-ratios for entries at log-distance d from a reference entry
    return 2e-5 + 1e-6 * np.abs(d)


def valid_output(x, y):
    y64, err = _prep(x, y)
    if err:
        return False, err
    return True, "ok"


def value_range(x, y):
    y64, err = _prep(x, y)
    if err:
        return False, err
    tol = _sum_tol(x.shape[1])
    if y64.min() < 0:
        return False, f"negative output {y64.min():.3g}"
    if y64.max() > 1 + tol:
        return False, f"output above 1: {y64.max():.7g}"
    return True, "ok"


def row_sums_one(x, y):
    y64, err = _prep(x, y)
    if err:
        return False, err
    s = y64.sum(axis=1)
    dev = np.abs(s - 1.0)
    tol = _sum_tol(x.shape[1])
    i = int(np.argmax(dev))
    if dev[i] > tol:
        return False, f"row {i} sums to {s[i]:.8g} (tol {tol:.2g})"
    return True, "ok"


def row_max_lower_bound(x, y):
    # the largest entry of a probability row over n columns is at least 1/n
    y64, err = _prep(x, y)
    if err:
        return False, err
    n = x.shape[1]
    m = y64.max(axis=1)
    lo = (1.0 - _sum_tol(n)) / n
    bad = m < lo
    if bad.any():
        i = int(np.argmax(bad))
        return False, f"row {i} max {m[i]:.3g} < 1/n"
    return True, "ok"


def monotone_in_x(x, y):
    y64, err = _prep(x, y)
    if err:
        return False, err
    if x.shape[1] < 2:
        return True, "ok"
    r = 2e-5
    order = np.argsort(x, axis=1, kind="stable")
    xs = np.take_along_axis(x, order, axis=1)
    ys = np.take_along_axis(y64, order, axis=1)
    a, b = ys[:, :-1], ys[:, 1:]
    tie = xs[:, :-1] == xs[:, 1:]
    dec = (a > b * (1 + r) + TINY) & ~tie
    mism = tie & (np.abs(a - b) > r * np.maximum(a, b) + TINY)
    bad = dec | mism
    if bad.any():
        i, j = np.argwhere(bad)[0]
        kind = "equal inputs, unequal outputs" if tie[i, j] else "not monotone"
        return False, f"row {i}: {kind} ({a[i, j]:.6g} vs {b[i, j]:.6g})"
    return True, "ok"


def log_ratio_matches_gap(x, y):
    # log(y_j / y_ref) must equal x_j - x_ref, with ref the row's largest input
    y64, err = _prep(x, y)
    if err:
        return False, err
    x64 = x.astype(np.float64)
    ref = np.argmax(x64, axis=1)
    rows = np.arange(x.shape[0])
    yref = y64[rows, ref]
    if np.any(yref <= TINY):
        i = int(np.argmax(yref <= TINY))
        return False, f"row {i}: output at row max is {yref[i]:.3g}"
    d = x64 - x64[rows, ref][:, None]
    valid = y64 > TINY
    with np.errstate(divide="ignore", invalid="ignore"):
        lr = np.log(np.where(valid, y64, 1.0)) - np.log(yref)[:, None]
    dev = np.abs(lr - d)
    bad = valid & (dev > _exp_tol(d))
    if bad.any():
        i, j = np.argwhere(bad)[0]
        return False, (f"row {i} col {j}: log-ratio {lr[i, j]:.6g} vs input gap "
                       f"{d[i, j]:.6g}")
    return True, "ok"


def adjacent_log_ratio(x, y):
    # same law between neighbouring columns, so it does not lean only on the max entry
    y64, err = _prep(x, y)
    if err:
        return False, err
    if x.shape[1] < 2:
        return True, "ok"
    x64 = x.astype(np.float64)
    xm = x64.max(axis=1, keepdims=True)
    valid = y64 > TINY
    vv = valid[:, :-1] & valid[:, 1:]
    with np.errstate(divide="ignore", invalid="ignore"):
        ly = np.log(np.where(valid, y64, 1.0))
    lr = ly[:, 1:] - ly[:, :-1]
    d = x64[:, 1:] - x64[:, :-1]
    dd = np.maximum(np.abs(x64[:, 1:] - xm), np.abs(x64[:, :-1] - xm))
    bad = vv & (np.abs(lr - d) > 2 * _exp_tol(dd))
    if bad.any():
        i, j = np.argwhere(bad)[0]
        return False, f"row {i} cols {j},{j+1}: log-ratio {lr[i, j]:.6g} vs gap {d[i, j]:.6g}"
    return True, "ok"


def shift_invariance(x0, y0, x1, y1):
    a, err = _prep(x0, y0)
    if err:
        return False, "case 0: " + err
    b, err = _prep(x1, y1)
    if err:
        return False, "case 1: " + err
    if x0.shape != x1.shape:
        return False, "shape mismatch between shifted cases"
    x0d = x0.astype(np.float64)
    x1d = x1.astype(np.float64)
    delta = x1d - x0d  # the float32 shift actually applied, per element
    valid = (a > TINY) & (b > TINY)
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.log(np.where(valid, b, 1.0)) - np.log(np.where(valid, a, 1.0)) - delta
    # q must be one constant per row (log of the ratio of the two normalisers)
    qmax = np.where(valid, q, -np.inf).max(axis=1)
    qmin = np.where(valid, q, np.inf).min(axis=1)
    if not np.all(valid.any(axis=1)):
        i = int(np.argmin(valid.any(axis=1)))
        return False, f"row {i}: no entries comparable across shift"
    d0 = x0d - x0d.max(axis=1, keepdims=True)
    d1 = x1d - x1d.max(axis=1, keepdims=True)
    tol = np.where(valid, _exp_tol(d0) + _exp_tol(d1), 0.0).max(axis=1)
    spread = qmax - qmin
    bad = spread > tol
    if bad.any():
        i = int(np.argmax(spread - tol))
        return False, f"row {i}: output not shift-invariant (spread {spread[i]:.3g}, tol {tol[i]:.3g})"
    return True, "ok"


CASE_LAWS = {
    "valid_output": valid_output,
    "value_range": value_range,
    "row_sums_one": row_sums_one,
    "row_max_lower_bound": row_max_lower_bound,
    "monotone_in_x": monotone_in_x,
    "log_ratio_matches_gap": log_ratio_matches_gap,
    "adjacent_log_ratio": adjacent_log_ratio,
}

SHIFT_LAWS = {"shift_invariance": shift_invariance}
