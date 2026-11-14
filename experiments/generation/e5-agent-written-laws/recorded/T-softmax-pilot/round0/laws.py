import numpy as np

EPS = 2.0 ** -23
TINY = 1e-30  # below this, y may be denormal/flushed and carries no relative accuracy


def _valid(x, y):
    y = np.asarray(y)
    if y.shape != x.shape:
        return False, f"shape {y.shape} != {x.shape}"
    if y.dtype != np.float32:
        return False, f"dtype {y.dtype} != float32"
    if not np.all(np.isfinite(y)):
        return False, "non-finite output"
    return True, ""


def _rel_tol(x):
    # per-row relative tolerance on y: base + error from float32 rounding of x-scale values + summation length
    x = np.asarray(x, dtype=np.float64)
    maxabs = np.max(np.abs(x), axis=1)
    return 1e-4 + 8 * EPS * maxabs + 0.5 * EPS * x.shape[1]


def shape_dtype_finite(x, y):
    return _valid(x, y)


def range_01(x, y):
    ok, msg = _valid(x, y)
    if not ok:
        return ok, msg
    y = y.astype(np.float64)
    if y.min() < 0:
        return False, f"negative probability {y.min():.3g}"
    tol = _rel_tol(x)[:, None]
    if np.any(y > 1 + tol):
        return False, f"probability above 1: {y.max():.6g}"
    return True, "ok"


def row_sums_one(x, y):
    ok, msg = _valid(x, y)
    if not ok:
        return ok, msg
    s = y.astype(np.float64).sum(axis=1)
    tol = 1e-5 + 2e-7 * x.shape[1]
    err = np.abs(s - 1.0)
    i = int(np.argmax(err))
    if err[i] > tol:
        return False, f"row {i} sums to {s[i]:.7g} (tol {tol:.2g})"
    return True, "ok"


def log_ratio_matches_x(x, y):
    # log y - x must be one constant per row (the row's -logsumexp)
    ok, msg = _valid(x, y)
    if not ok:
        return ok, msg
    xd = x.astype(np.float64)
    yd = y.astype(np.float64)
    tol = _rel_tol(x)
    valid = yd > TINY
    if not np.all(valid.any(axis=1)):
        return False, "row with no representable probability"
    r = np.where(valid, np.log(np.where(valid, yd, 1.0)) - xd, np.nan)
    spread = np.nanmax(r, axis=1) - np.nanmin(r, axis=1)
    i = int(np.argmax(spread - tol))
    if spread[i] > tol[i]:
        return False, f"row {i}: log(y)-x varies by {spread[i]:.3g} (tol {tol[i]:.2g})"
    return True, "ok"


def order_preserved(x, y):
    ok, msg = _valid(x, y)
    if not ok:
        return ok, msg
    if x.shape[1] < 2:
        return True, "ok"
    yd = y.astype(np.float64)
    order = np.argsort(x, axis=1, kind="stable")
    xs = np.take_along_axis(x, order, axis=1)
    ys = np.take_along_axis(yd, order, axis=1)
    tol = 2 * _rel_tol(x)[:, None]
    strict = xs[:, :-1] < xs[:, 1:]
    bad = strict & (ys[:, :-1] > ys[:, 1:] * (1 + tol) + TINY)
    if bad.any():
        i, j = np.argwhere(bad)[0]
        return False, f"row {i}: larger x got smaller y (sorted pos {j})"
    return True, "ok"


def no_early_underflow(x, y):
    # entries within 60 of the row max have y >= ~1e-30 (normal float32), so they must be nonzero
    ok, msg = _valid(x, y)
    if not ok:
        return ok, msg
    xd = x.astype(np.float64)
    d = xd.max(axis=1, keepdims=True) - xd
    bad = (d <= 60.0) & (y <= 0)
    if bad.any():
        i, j = np.argwhere(bad)[0]
        return False, f"y[{i},{j}] is zero but x is within 60 of the row max"
    return True, "ok"


def shift_invariance(x0, y0, x1, y1):
    ok, msg = _valid(x0, y0)
    if not ok:
        return ok, msg
    ok, msg = _valid(x1, y1)
    if not ok:
        return ok, msg
    a = y0.astype(np.float64)
    b = y1.astype(np.float64)
    # x1 = fl(x0 + c): per-element rounding of the shifted input perturbs y too
    tol = (_rel_tol(x0) + _rel_tol(x1))[:, None]
    bad = np.abs(a - b) > tol * np.maximum(a, b) + TINY
    if bad.any():
        i, j = np.argwhere(bad)[0]
        return False, f"y changed under row shift at [{i},{j}]: {a[i, j]:.6g} vs {b[i, j]:.6g}"
    return True, "ok"


CASE_LAWS = {
    "shape_dtype_finite": shape_dtype_finite,
    "range_01": range_01,
    "row_sums_one": row_sums_one,
    "log_ratio_matches_x": log_ratio_matches_x,
    "order_preserved": order_preserved,
    "no_early_underflow": no_early_underflow,
}
SHIFT_LAWS = {"shift_invariance": shift_invariance}
