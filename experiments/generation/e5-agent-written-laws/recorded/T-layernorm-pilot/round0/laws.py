import numpy as np

EPS32 = 2.0 ** -23
EPS = 1e-5


def _basic(x, y):
    if not isinstance(y, np.ndarray):
        return False, "output is not an ndarray"
    if y.shape != x.shape:
        return False, "shape %s != %s" % (y.shape, x.shape)
    if y.dtype != np.float32:
        return False, "dtype %s != float32" % y.dtype
    if not np.all(np.isfinite(y)):
        return False, "non-finite output"
    return True, ""


def _stats(x):
    xd = np.asarray(x, dtype=np.float64)
    n = xd.shape[1]
    m = xd.mean(axis=1, keepdims=True)
    xc = xd - m
    v = (xc * xc).mean(axis=1)
    s = np.sqrt(v + EPS)
    R = np.abs(xd).max(axis=1)
    std = np.sqrt(v)
    # absolute tolerance on y from rounding of the mean / fused ops, scaled by range over s
    A = EPS32 * R / s * (4.0 + np.sqrt(n)) + 1e-6
    # relative tolerance on the scale 1/s (covers one-pass variance cancellation)
    K1 = 4.0 + 0.25 * np.sqrt(n)
    with np.errstate(divide="ignore", invalid="ignore"):
        lin = np.where(std > 0, 8.0 * R / std, np.inf)
    ta = 32 * EPS32 + EPS32 * (lin + 0.5 * K1 * (R / s) ** 2)
    ta = np.minimum(ta, 0.3)
    return dict(xd=xd, xc=xc, n=n, v=v, s=s, R=R, A=A, ta=ta)


def _fit(st, y):
    yd = y.astype(np.float64)
    ybar = yd.mean(axis=1, keepdims=True)
    yc = yd - ybar
    xc = st["xc"]
    Sxx = (xc * xc).sum(axis=1)
    Sxy = (xc * yc).sum(axis=1)
    ok = Sxx > 0
    a = np.where(ok, Sxy / np.where(ok, Sxx, 1.0), 0.0)
    return yd, ybar[:, 0], yc, a, ok


def shape_dtype_finite(x, y):
    return _basic(x, y)


def row_mean_zero(x, y):
    ok, d = _basic(x, y)
    if not ok:
        return ok, d
    st = _stats(x)
    yd = y.astype(np.float64)
    mean = yd.mean(axis=1)
    tol = st["A"] + 4 * EPS32 * np.abs(yd).mean(axis=1)
    bad = np.abs(mean) > tol
    if bad.any():
        i = int(np.argmax(np.abs(mean) - tol))
        return False, "row %d mean(y)=%.3g tol=%.3g" % (i, mean[i], tol[i])
    return True, "ok"


def row_linear_in_x(x, y):
    ok, d = _basic(x, y)
    if not ok:
        return ok, d
    st = _stats(x)
    yd, ybar, yc, a, has = _fit(st, y)
    resid = np.abs(yc - a[:, None] * st["xc"]).max(axis=1)
    tol = st["A"] + 16 * EPS32 * np.abs(yd).max(axis=1)
    # constant rows: y must be ~0
    zero = np.abs(yd).max(axis=1)
    bad = np.where(has, resid > tol, zero > st["A"])
    if bad.any():
        i = int(np.argmax(bad))
        r = resid[i] if has[i] else zero[i]
        return False, "row %d not affine in x: resid=%.3g tol=%.3g" % (i, r, tol[i])
    return True, "ok"


def row_scale(x, y):
    ok, d = _basic(x, y)
    if not ok:
        return ok, d
    st = _stats(x)
    yd, ybar, yc, a, has = _fit(st, y)
    err = np.abs(a * st["s"] - 1.0)
    bad = has & (err > st["ta"])
    if bad.any():
        i = int(np.argmax(np.where(bad, err - st["ta"], -np.inf)))
        return False, "row %d slope*sqrt(var+eps)=%.6g tol=%.3g" % (i, a[i] * st["s"], st["ta"][i])
    return True, "ok"


def row_second_moment(x, y):
    ok, d = _basic(x, y)
    if not ok:
        return ok, d
    st = _stats(x)
    yd = y.astype(np.float64)
    m2 = (yd * yd).mean(axis=1)
    q = st["v"] / (st["v"] + EPS)
    A = st["A"]
    tol = 2 * st["ta"] * q + 2 * A * np.sqrt(q) + A * A + 32 * EPS32 * q + 1e-9
    err = np.abs(m2 - q)
    bad = err > tol
    if bad.any():
        i = int(np.argmax(err - tol))
        return False, "row %d mean(y^2)=%.6g expected %.6g tol=%.3g" % (i, m2[i], q[i], tol[i])
    return True, "ok"


def shift_invariance(x0, y0, x1, y1):
    ok, d = _basic(x0, y0)
    if not ok:
        return ok, d
    ok, d = _basic(x1, y1)
    if not ok:
        return ok, d
    if x0.shape != x1.shape:
        return False, "case shapes differ"
    s0 = _stats(x0)
    s1 = _stats(x1)
    a = y0.astype(np.float64)
    b = y1.astype(np.float64)
    tol = (s0["A"] + s1["A"])[:, None] + (s0["ta"] + s1["ta"])[:, None] * np.abs(a) \
        + 16 * EPS32 * (np.abs(a) + np.abs(b))
    diff = np.abs(a - b)
    bad = diff > tol
    if bad.any():
        i, j = np.unravel_index(int(np.argmax(diff - tol)), diff.shape)
        return False, "y changed under row shift at (%d,%d): %.6g vs %.6g tol=%.3g" % (
            i, j, a[i, j], b[i, j], tol[i, j])
    return True, "ok"


CASE_LAWS = {
    "shape_dtype_finite": shape_dtype_finite,
    "row_mean_zero": row_mean_zero,
    "row_linear_in_x": row_linear_in_x,
    "row_scale": row_scale,
    "row_second_moment": row_second_moment,
}
SHIFT_LAWS = {"shift_invariance": shift_invariance}
