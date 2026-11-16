import numpy as np

EPS = 2.0 ** -23
TINY = 1e-30


def _clean(x, y):
    """Return y as float64 if it has the right shape/dtype and is finite, else None."""
    if not isinstance(y, np.ndarray):
        return None
    if y.shape != x.shape or y.dtype != np.float32:
        return None
    if not np.all(np.isfinite(y)):
        return None
    return y.astype(np.float64)


def _tol(x64):
    cols = x64.shape[1]
    rng = x64.max(axis=1) - x64.min(axis=1)
    return 4.0 * EPS * (cols + 16.0 + 2.0 * rng)


def structure(x, y):
    y64 = _clean(x, y)
    if y64 is None:
        return False, "wrong shape, dtype, or non-finite output"
    x64 = x.astype(np.float64)
    tol = _tol(x64)
    if np.any(y64 < 0.0):
        return False, "negative entry"
    if np.any(y64.max(axis=1) > 1.0 + tol):
        return False, "entry above 1"
    return True, "ok"


def row_sum(x, y):
    y64 = _clean(x, y)
    if y64 is None:
        return False, "wrong shape, dtype, or non-finite output"
    x64 = x.astype(np.float64)
    tol = _tol(x64)
    err = np.abs(y64.sum(axis=1) - 1.0)
    bad = err > tol
    if np.any(bad):
        r = int(np.argmax(err - tol))
        return False, f"row {r} sums to {1.0 + (y64[r].sum() - 1.0):.8g}, tol {tol[r]:.3g}"
    return True, "ok"


def log_ratio(x, y):
    # y_j / y_k = exp(x_j - x_k): log(y_j) - x_j must be the same for every j in a row.
    y64 = _clean(x, y)
    if y64 is None:
        return False, "wrong shape, dtype, or non-finite output"
    x64 = x.astype(np.float64)
    tol = _tol(x64)
    rows = x64.shape[0]
    idx = np.arange(rows)
    k = np.argmax(y64, axis=1)
    yk = y64[idx, k]
    if np.any(yk <= 0.0):
        return False, "row with no positive entry"
    xd = x64 - x64[idx, k][:, None]
    underflow = (xd >= -50.0) & (y64 < TINY)
    if np.any(underflow):
        r = int(np.argwhere(underflow)[0][0])
        return False, f"row {r} has an entry that should be >= 1e-26 but is below {TINY}"
    valid = y64 >= TINY
    safe = np.where(valid, y64, 1.0)
    d = np.log(safe) - x64
    dk = d[idx, k]
    dev = np.where(valid, np.abs(d - dk[:, None]), 0.0)
    worst = dev.max(axis=1)
    bad = worst > tol
    if np.any(bad):
        r = int(np.argmax(worst - tol))
        return False, f"row {r}: log-ratio deviation {worst[r]:.3g} > tol {tol[r]:.3g}"
    return True, "ok"


CASE_LAWS = {
    "structure": structure,
    "row_sum": row_sum,
    "log_ratio": log_ratio,
}

SHIFT_LAWS = {}
