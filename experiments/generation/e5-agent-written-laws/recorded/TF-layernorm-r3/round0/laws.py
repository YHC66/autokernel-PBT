import numpy as np

EPS32 = 2.0 ** -23
LN_EPS = 1e-5


def _ctx(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    if x.ndim != 2:
        return None, "x is not 2-D"
    if y.shape != x.shape:
        return None, "wrong shape %s, expected %s" % (y.shape, x.shape)
    if y.dtype != np.float32:
        return None, "wrong dtype %s, expected float32" % (y.dtype,)
    if not np.all(np.isfinite(y)):
        return None, "output has NaN or inf"
    xd = x.astype(np.float64)
    yd = y.astype(np.float64)
    n = xd.shape[1]
    m = xd.mean(axis=1, keepdims=True)
    xc = xd - m
    v = (xc * xc).mean(axis=1)
    s = np.sqrt(v + LN_EPS)
    M = np.abs(xd).max(axis=1)
    Q = M / s
    Y = np.abs(xc).max(axis=1) / s
    W = (xd * xd).mean(axis=1) / (s * s)
    sn = np.sqrt(float(n))
    b_mean = 8.0 + 2.0 * sn
    b_var = 6.0 + sn
    c = dict(
        xd=xd, yd=yd, n=n, xc=xc, v=v, s=s, Q=Q, Y=Y, W=W,
        tol_mean=EPS32 * b_mean * (1.0 + Q) + 2.0 * EPS32 * Y,
        tol_col=EPS32 * (16.0 * (1.0 + Y) + 8.0 * Q),
        rel_scale=EPS32 * (1.5 * b_var * W + 12.0),
    )
    return c, None


def output_wellformed(x, y):
    c, err = _ctx(x, y)
    if c is None:
        return False, err
    return True, "output has the right shape and dtype and is finite"


def zero_mean(x, y):
    c, err = _ctx(x, y)
    if c is None:
        return False, err
    ym = c["yd"].mean(axis=1)
    bad = np.abs(ym) > c["tol_mean"]
    if np.any(bad):
        i = int(np.argmax(np.abs(ym) - c["tol_mean"]))
        return False, "row %d: mean(y)=%.3e exceeds tol %.3e" % (i, ym[i], c["tol_mean"][i])
    return True, "row means of y are zero"


def _fit(c):
    xc = c["xc"]
    yd = c["yd"]
    yc = yd - yd.mean(axis=1, keepdims=True)
    den = (xc * xc).sum(axis=1)
    num = (yc * xc).sum(axis=1)
    a = np.where(den > 0.0, num / np.where(den > 0.0, den, 1.0), 0.0)
    r = yc - a[:, None] * xc
    return yc, a, r


def collinear(x, y):
    c, err = _ctx(x, y)
    if c is None:
        return False, err
    _, a, r = _fit(c)
    rmax = np.abs(r).max(axis=1)
    bad = rmax > c["tol_col"]
    if np.any(bad):
        i = int(np.argmax(rmax - c["tol_col"]))
        return False, "row %d: y not proportional to x - mean(x): residual %.3e > tol %.3e" % (
            i, rmax[i], c["tol_col"][i])
    return True, "y is proportional to x minus its row mean"


def covariance_with_input(x, y):
    c, err = _ctx(x, y)
    if c is None:
        return False, err
    xc = c["xc"]
    yc = c["yd"] - c["yd"].mean(axis=1, keepdims=True)
    cov = (yc * xc).mean(axis=1)
    target = c["v"] / c["s"]
    tol = c["rel_scale"] * target + c["tol_col"] * np.abs(xc).mean(axis=1) + 1e-30
    dev = np.abs(cov - target)
    bad = dev > tol
    if np.any(bad):
        i = int(np.argmax(dev - tol))
        return False, "row %d: cov(y, x)=%.6e, expected %.6e (tol %.3e)" % (i, cov[i], target[i], tol[i])
    return True, "cov(y, x) equals var(x)/sqrt(var(x)+1e-5)"


CASE_LAWS = {
    "output_wellformed": output_wellformed,
    "zero_mean": zero_mean,
    "collinear": collinear,
    "covariance_with_input": covariance_with_input,
}
SHIFT_LAWS = {}
