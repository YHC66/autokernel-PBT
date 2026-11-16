import numpy as np

EPS = 1e-5
F32 = float(np.finfo(np.float32).eps)


def _valid(x, y):
    return (
        isinstance(y, np.ndarray)
        and y.dtype == np.float32
        and y.shape == x.shape
        and bool(np.all(np.isfinite(y)))
    )


def _tol(x64, v, rmax):
    n = x64.shape[1]
    s = np.sqrt(v + EPS)
    r = rmax / s
    return 1e-6 + 4.0 * F32 * (1.0 + np.sqrt(n)) * (1.0 + r * r)


def _row_stats(x):
    x64 = x.astype(np.float64)
    xc = x64 - x64.mean(axis=1, keepdims=True)
    v = (xc * xc).mean(axis=1)
    rmax = np.abs(x64).max(axis=1)
    return x64, xc, v, rmax


def shape_dtype(x, y):
    if not isinstance(y, np.ndarray):
        return False, "output is not an ndarray"
    if y.dtype != np.float32:
        return False, f"dtype {y.dtype}, expected float32"
    if y.shape != x.shape:
        return False, f"shape {y.shape}, expected {x.shape}"
    if not np.all(np.isfinite(y)):
        return False, "non-finite output"
    return True, "ok"


def zero_mean(x, y):
    if not _valid(x, y):
        return False, "invalid output"
    x64, xc, v, rmax = _row_stats(x)
    t = _tol(x64, v, rmax)
    m = y.astype(np.float64).mean(axis=1)
    bad = np.abs(m) > t
    if bad.any():
        i = int(np.argmax(np.abs(m) - t))
        return False, f"row {i}: mean of output {m[i]:.3e} exceeds {t[i]:.3e}"
    return True, "ok"


def unit_variance(x, y):
    if not _valid(x, y):
        return False, "invalid output"
    x64, xc, v, rmax = _row_stats(x)
    t = _tol(x64, v, rmax)
    y64 = y.astype(np.float64)
    q = (y64 * y64).mean(axis=1)
    target = v / (v + EPS)
    bad = np.abs(q - target) > 3.0 * t + t * t
    if bad.any():
        i = int(np.argmax(np.abs(q - target) - (3.0 * t + t * t)))
        return False, f"row {i}: mean square {q[i]:.6g}, expected about {target[i]:.6g}"
    return True, "ok"


def affine_in_x(x, y):
    if not _valid(x, y):
        return False, "invalid output"
    x64, xc, v, rmax = _row_stats(x)
    t = _tol(x64, v, rmax)
    y64 = y.astype(np.float64)
    yc = y64 - y64.mean(axis=1, keepdims=True)
    xx = (xc * xc).sum(axis=1)
    for i in range(x.shape[0]):
        if xx[i] > 0.0:
            a = float((xc[i] * yc[i]).sum() / xx[i])
            res = float(np.abs(yc[i] - a * xc[i]).max())
        else:
            res = float(np.abs(yc[i]).max())
        if res > 3.0 * t[i]:
            return False, f"row {i}: output is not affine in input (residual {res:.3e} > {3.0 * t[i]:.3e})"
    return True, "ok"


def positive_slope(x, y):
    if not _valid(x, y):
        return False, "invalid output"
    x64, xc, v, rmax = _row_stats(x)
    t = _tol(x64, v, rmax)
    y64 = y.astype(np.float64)
    yc = y64 - y64.mean(axis=1, keepdims=True)
    xx = (xc * xc).sum(axis=1)
    for i in range(x.shape[0]):
        if xx[i] > 0.0:
            a = float((xc[i] * yc[i]).sum() / xx[i])
            if a * np.sqrt(v[i]) < -3.0 * t[i]:
                return False, f"row {i}: output decreases with input (slope {a:.3e})"
    return True, "ok"


def shift_invariance(x0, y0, x1, y1):
    if not (_valid(x0, y0) and _valid(x1, y1)) or x0.shape != x1.shape:
        return False, "invalid output"
    a64, _, v0, r0 = _row_stats(x0)
    b64, _, v1, r1 = _row_stats(x1)
    v = np.minimum(v0, v1)
    rmax = np.maximum(r0, r1)
    t = _tol(a64, v, rmax)
    d = np.abs(y0.astype(np.float64) - y1.astype(np.float64)).max(axis=1)
    bad = d > 2.0 * t
    if bad.any():
        i = int(np.argmax(d - 2.0 * t))
        return False, f"row {i}: output changed by {d[i]:.3e} under a shift (limit {2.0 * t[i]:.3e})"
    return True, "ok"


CASE_LAWS = {
    "shape_dtype": shape_dtype,
    "zero_mean": zero_mean,
    "unit_variance": unit_variance,
    "affine_in_x": affine_in_x,
    "positive_slope": positive_slope,
}

SHIFT_LAWS = {
    "shift_invariance": shift_invariance,
}
