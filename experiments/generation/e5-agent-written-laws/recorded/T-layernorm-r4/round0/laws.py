import numpy as np

EPS32 = 2.0 ** -23
LN_EPS = 1e-5


def _stats(x):
    """Float64 row statistics of x, used only to size tolerances and moment targets."""
    x64 = np.asarray(x, dtype=np.float64)
    n = x64.shape[1]
    m = x64.mean(axis=1, keepdims=True)
    xc = x64 - m
    v = (xc * xc).mean(axis=1)
    denom = np.sqrt(v + LN_EPS)
    maxabs = np.abs(x64).max(axis=1)
    q = (x64 * x64).mean(axis=1)
    ymax = np.abs(xc).max(axis=1) / denom
    sq = np.sqrt(n)
    # error of the row mean (in y units): float32 summation, grows with length and magnitude
    delta = EPS32 * maxabs * (16.0 * sq + 16.0) / denom
    # relative error of (v + 1e-5): covers E[x^2] - m^2 style variance kernels
    rel_v = 2.0 * EPS32 * q * (sq + 8.0) / (v + LN_EPS) + 32.0 * EPS32
    # pointwise rounding error of y, including fused x*rstd - m*rstd forms
    tol_e = 16.0 * EPS32 * (maxabs / denom + ymax) + 1e-7
    return dict(x=x64, xc=xc, n=n, v=v, denom=denom, maxabs=maxabs, q=q,
                ymax=ymax, delta=delta, rel_v=rel_v, tol_e=tol_e)


def _prep(x, y):
    x = np.asarray(x)
    if x.ndim != 2:
        return None, "x is not 2-D"
    y = np.asarray(y)
    if y.shape != x.shape:
        return None, "shape %s != %s" % (y.shape, x.shape)
    try:
        y64 = y.astype(np.float64)
    except Exception:
        return None, "y not numeric"
    if not np.all(np.isfinite(y64)):
        return None, "y has non-finite values"
    return (y64, _stats(x)), None


def well_formed(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    if y.shape != x.shape:
        return False, "shape %s != %s" % (y.shape, x.shape)
    if y.dtype != np.float32:
        return False, "dtype %s != float32" % y.dtype
    if not np.all(np.isfinite(y)):
        return False, "non-finite output"
    return True, "ok"


def zero_row_mean(x, y):
    p, err = _prep(x, y)
    if err:
        return False, err
    y64, s = p
    got = np.abs(y64.mean(axis=1))
    tol = s["delta"] + s["tol_e"]
    bad = got > tol
    if bad.any():
        i = int(np.argmax(got - tol))
        return False, "row %d mean %.3g > tol %.3g" % (i, got[i], tol[i])
    return True, "ok"


def second_moment(x, y):
    # mean(y^2) must equal v/(v+1e-5) for population variance v of the row
    p, err = _prep(x, y)
    if err:
        return False, err
    y64, s = p
    r = (y64 * y64).mean(axis=1)
    r0 = s["v"] / (s["v"] + LN_EPS)
    tol = r0 * s["rel_v"] + 4.0 * s["delta"] ** 2 + 4.0 * s["tol_e"] * (np.sqrt(r0) + s["tol_e"]) + 1e-9
    dev = np.abs(r - r0)
    if (dev > tol).any():
        i = int(np.argmax(dev - tol))
        return False, "row %d mean(y^2)=%.6g expected %.6g (tol %.3g)" % (i, r[i], r0[i], tol[i])
    return True, "ok"


def affine_in_x(x, y):
    # within a row y must be an affine function of x
    p, err = _prep(x, y)
    if err:
        return False, err
    y64, s = p
    xc = s["xc"]
    yc = y64 - y64.mean(axis=1, keepdims=True)
    sxx = (xc * xc).sum(axis=1)
    sxy = (xc * yc).sum(axis=1)
    a = np.zeros_like(sxx)
    np.divide(sxy, sxx, out=a, where=sxx > 0)
    resid = np.abs(yc - a[:, None] * xc).max(axis=1)
    tol = 2.0 * s["tol_e"]
    if (resid > tol).any():
        i = int(np.argmax(resid - tol))
        return False, "row %d not affine in x: residual %.3g > %.3g" % (i, resid[i], tol[i])
    return True, "ok"


def order_preserving(x, y):
    # larger x gives larger y; equal x gives equal y
    p, err = _prep(x, y)
    if err:
        return False, err
    y64, s = p
    order = np.argsort(s["x"], axis=1, kind="stable")
    xs = np.take_along_axis(s["x"], order, axis=1)
    ys = np.take_along_axis(y64, order, axis=1)
    if xs.shape[1] < 2:
        return True, "ok"
    dy = np.diff(ys, axis=1)
    dx = np.diff(xs, axis=1)
    tol = (2.0 * s["tol_e"])[:, None]
    bad = (dy < -tol) | ((dx == 0) & (np.abs(dy) > tol))
    if bad.any():
        i = int(np.argmax(bad.any(axis=1)))
        return False, "row %d violates monotonicity/ties" % i
    return True, "ok"


def shift_invariance(x0, y0, x1, y1):
    p0, err = _prep(x0, y0)
    if err:
        return False, "case0: " + err
    p1, err = _prep(x1, y1)
    if err:
        return False, "case1: " + err
    if np.shape(x0) != np.shape(x1):
        return False, "x shapes differ"
    a64, sa = p0
    b64, sb = p1
    tol = (sa["delta"] + sb["delta"] + sa["tol_e"] + sb["tol_e"]
           + sa["ymax"] * sa["rel_v"] + sb["ymax"] * sb["rel_v"])
    d = np.abs(a64 - b64).max(axis=1)
    if (d > tol).any():
        i = int(np.argmax(d - tol))
        return False, "row %d output changed by %.3g under shift (tol %.3g)" % (i, d[i], tol[i])
    return True, "ok"


CASE_LAWS = {
    "well_formed": well_formed,
    "zero_row_mean": zero_row_mean,
    "second_moment": second_moment,
    "affine_in_x": affine_in_x,
    "order_preserving": order_preserving,
}
SHIFT_LAWS = {"shift_invariance": shift_invariance}
