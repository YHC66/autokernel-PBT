import numpy as np

EPS = 1e-5
U = 2.0 ** -23


def _valid(x, y):
    return (
        isinstance(y, np.ndarray)
        and y.shape == x.shape
        and y.dtype == np.float32
        and bool(np.all(np.isfinite(y)))
    )


def _tols(x):
    """Per-row error budgets (in output units) for a float32 kernel; uses only x."""
    x64 = np.asarray(x, dtype=np.float64)
    n = x64.shape[1]
    m = x64.mean(axis=1)
    xc = x64 - m[:, None]
    v = (xc * xc).mean(axis=1)
    s = np.sqrt(v + EPS)
    q = v / (v + EPS)
    A = np.abs(x64).max(axis=1)
    B = np.abs(x64).mean(axis=1)
    ymax = np.abs(xc).max(axis=1) / s

    u_m = U * (n / 2.0 + 8.0)
    u_v = U * (3.0 * np.sqrt(n) + 8.0)

    noise = 4.0 * U * (ymax + A / s)
    tol_mean = 4.0 * U + noise + u_m * B / s
    dv = 2.0 * u_v * v + 4.0 * U * A * A + (u_m * B) ** 2
    lo = v / (v + dv + EPS)
    hi = v / (np.maximum(v - dv, 0.0) + EPS)
    tol_var = (
        np.maximum(hi - q, q - lo)
        + 8.0 * U * q
        + 2.0 * noise * np.sqrt(q)
        + noise ** 2
    )
    return dict(
        xc=xc, v=v, s=s, q=q, noise=noise, tol_mean=tol_mean, tol_var=tol_var,
        tol_aff=2.0 * noise, dv=dv,
    )


def _report(bad, name, got, want, tol):
    i = int(np.argmax(bad))
    return False, "%s: row %d got %.6g, expected %.6g, tol %.3g" % (
        name, i, float(got[i]), float(want[i]), float(tol[i])
    )


def shape_dtype_finite(x, y):
    if not _valid(x, y):
        return False, "output must be a finite float32 array with the shape of x"
    return True, "ok"


def zero_mean(x, y):
    if not _valid(x, y):
        return False, "invalid output"
    t = _tols(x)
    my = y.astype(np.float64).mean(axis=1)
    bad = np.abs(my) > t["tol_mean"]
    if bad.any():
        return _report(bad, "row mean of y", my, np.zeros_like(my), t["tol_mean"])
    return True, "ok"


def variance(x, y):
    if not _valid(x, y):
        return False, "invalid output"
    t = _tols(x)
    y64 = y.astype(np.float64)
    yc = y64 - y64.mean(axis=1, keepdims=True)
    vy = (yc * yc).mean(axis=1)
    bad = np.abs(vy - t["q"]) > t["tol_var"]
    if bad.any():
        return _report(bad, "row variance of y", vy, t["q"], t["tol_var"])
    return True, "ok"


def _fit(x, y, t):
    y64 = y.astype(np.float64)
    yc = y64 - y64.mean(axis=1, keepdims=True)
    xc = t["xc"]
    sxx = (xc * xc).sum(axis=1)
    ok = t["v"] > 0.0
    a = np.zeros(x.shape[0])
    a[ok] = (xc[ok] * yc[ok]).sum(axis=1) / sxx[ok]
    resid = yc - a[:, None] * xc
    return ok, a, resid


def affine(x, y):
    if not _valid(x, y):
        return False, "invalid output"
    t = _tols(x)
    ok, a, resid = _fit(x, y, t)
    r = np.abs(resid).max(axis=1)
    bad = ok & (r > t["tol_aff"])
    if bad.any():
        return _report(bad, "max deviation of y from an affine function of x", r,
                       np.zeros_like(r), t["tol_aff"])
    return True, "ok"


def positive_slope(x, y):
    if not _valid(x, y):
        return False, "invalid output"
    t = _tols(x)
    ok, a, _ = _fit(x, y, t)
    gain = a * np.sqrt(t["v"])
    bad = ok & (gain < -2.0 * t["noise"])
    if bad.any():
        return _report(bad, "slope of y against x (scaled)", gain,
                       np.sqrt(t["q"]), 2.0 * t["noise"])
    return True, "ok"


def shift_invariance(x0, y0, x1, y1):
    if not (_valid(x0, y0) and _valid(x1, y1)) or x0.shape != x1.shape:
        return False, "invalid output"
    t0 = _tols(x0)
    t1 = _tols(x1)
    d = x1.astype(np.float64) - x0.astype(np.float64)
    sd = d.max(axis=1) - d.min(axis=1)
    smin = np.minimum(t0["s"], t1["s"])
    ymax = np.maximum(np.abs(y0).max(axis=1), np.abs(y1).max(axis=1)).astype(np.float64)
    rel = (
        np.abs(t0["s"] - t1["s"]) / smin
        + t0["dv"] / (2.0 * (t0["v"] + EPS))
        + t1["dv"] / (2.0 * (t1["v"] + EPS))
        + 8.0 * U
    )
    tol = 1.5 * (
        sd / smin
        + t0["tol_mean"] + t1["tol_mean"]
        + t0["noise"] + t1["noise"]
        + ymax * rel
    )
    diff = np.abs(y1.astype(np.float64) - y0.astype(np.float64)).max(axis=1)
    bad = diff > tol
    if bad.any():
        return _report(bad, "max |y1 - y0| after a per-row shift", diff,
                       np.zeros_like(diff), tol)
    return True, "ok"


CASE_LAWS = {
    "shape_dtype_finite": shape_dtype_finite,
    "zero_mean": zero_mean,
    "variance": variance,
    "affine": affine,
    "positive_slope": positive_slope,
}

SHIFT_LAWS = {"shift_invariance": shift_invariance}
