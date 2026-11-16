import numpy as np

EPS = 2.0 ** -23
TINY = 1e-25  # entries below this may be denormal/flushed in correct kernels
DCAP = 100.0


def _malformed(x, y):
    if not isinstance(y, np.ndarray):
        return "output is not an ndarray"
    if y.shape != x.shape:
        return "shape %s != %s" % (y.shape, x.shape)
    if y.dtype != np.float32:
        return "dtype %s != float32" % y.dtype
    if not bool(np.all(np.isfinite(y))):
        return "output has NaN or inf"
    return None


def wellformed(x, y):
    m = _malformed(x, y)
    if m is not None:
        return False, m
    return True, "ok"


def nonneg_bounded(x, y):
    m = _malformed(x, y)
    if m is not None:
        return False, m
    y64 = y.astype(np.float64)
    if not bool(np.all(y64 >= 0.0)):
        return False, "negative entry"
    if not bool(np.all(y64 <= 1.0 + 1e-4)):
        return False, "entry above 1"
    return True, "ok"


def sums_to_one(x, y):
    m = _malformed(x, y)
    if m is not None:
        return False, m
    cols = x.shape[1]
    tol = 2e-5 + 0.5 * EPS * cols
    s = y.astype(np.float64).sum(axis=1)
    err = np.abs(s - 1.0)
    if not bool(np.all(err <= tol)):
        return False, "row sum off by %.3g (tol %.3g)" % (float(np.max(err)), tol)
    return True, "ok"


def ratio_law(x, y):
    # y_i / y_j = exp(x_i - x_j): log y_i - x_i must be the same for every entry of a row.
    m = _malformed(x, y)
    if m is not None:
        return False, m
    x64 = x.astype(np.float64)
    y64 = y.astype(np.float64)
    rows = x.shape[0]
    mx = x64.max(axis=1, keepdims=True)
    d = np.minimum(np.abs(x64 - mx), DCAP)
    ref = np.argmax(y64, axis=1)
    idx = np.arange(rows)
    yref = y64[idx, ref]
    if not bool(np.all(yref > TINY)):
        return False, "row has no non-negligible entry"
    valid = y64 > TINY
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.log(np.where(valid, y64, 1.0)) - (x64 - mx)
    rref = r[idx, ref][:, None]
    dref = d[idx, ref][:, None]
    tol = 1e-5 + 4.0 * EPS * (d + dref)
    bad = valid & ~(np.abs(r - rref) <= tol)
    if bool(np.any(bad)):
        gap = np.abs(r - rref)
        gap = np.where(valid, gap, 0.0)
        return False, "log-ratio deviates by %.3g" % float(np.max(gap))
    return True, "ok"


def shift_invariance(x0, y0, x1, y1):
    m = _malformed(x0, y0)
    if m is not None:
        return False, m
    m = _malformed(x1, y1)
    if m is not None:
        return False, m
    if x0.shape != x1.shape:
        return False, "case shapes differ"
    a = x0.astype(np.float64)
    b = x1.astype(np.float64)
    dd = b - a
    spread = dd.max(axis=1, keepdims=True) - dd.min(axis=1, keepdims=True)
    d0 = np.minimum(np.abs(a - a.max(axis=1, keepdims=True)), DCAP)
    tol = 2e-5 + 8.0 * EPS * d0 + 2.0 * spread
    p = y0.astype(np.float64)
    q = y1.astype(np.float64)
    big = np.maximum(p, q)
    active = big > TINY
    ok = np.abs(q - p) <= tol * big
    bad = active & ~ok
    if bool(np.any(bad)):
        rel = np.abs(q - p) / np.where(active, big, 1.0)
        rel = np.where(active, rel, 0.0)
        return False, "shifted output differs by relative %.3g" % float(np.max(rel))
    return True, "ok"


CASE_LAWS = {
    "wellformed": wellformed,
    "nonneg_bounded": nonneg_bounded,
    "sums_to_one": sums_to_one,
    "ratio_law": ratio_law,
}
SHIFT_LAWS = {"shift_invariance": shift_invariance}
