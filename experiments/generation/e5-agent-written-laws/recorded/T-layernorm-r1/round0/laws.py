import numpy as np

EPS32 = 2.0 ** -23
EPS_LN = 1e-5
RHO = 1e-5  # relative slack for scale errors (sqrt / rsqrt variants, output rounding)


def _form_problem(x, y):
    if not isinstance(y, np.ndarray):
        return "output is not an ndarray"
    if y.shape != x.shape:
        return f"shape {y.shape} != {x.shape}"
    if y.dtype != np.float32:
        return f"dtype {y.dtype} != float32"
    if not np.all(np.isfinite(y)):
        return "non-finite values in output"
    return None


def _stats(x):
    x64 = x.astype(np.float64)
    n = x64.shape[1]
    xc = x64 - x64.mean(axis=1, keepdims=True)
    v = (xc * xc).mean(axis=1)
    s = np.sqrt(v + EPS_LN)
    M = np.abs(x64).max(axis=1)
    return x64, xc, v, s, M, n


def _eel(s, M, n):
    # elementwise absolute error budget of a correct float32 kernel: grows with
    # row length and with max|x| / scale
    return 2e-5 + 2.0 * EPS32 * (4.0 + np.sqrt(n)) * M / s


def well_formed(x, y):
    p = _form_problem(x, y)
    return (p is None), (p or "shape/dtype/finite ok")


def zero_mean(x, y):
    p = _form_problem(x, y)
    if p:
        return False, p
    _, _, _, s, M, n = _stats(x)
    tol = _eel(s, M, n)
    err = np.abs(y.astype(np.float64).mean(axis=1))
    bad = err > tol
    if bad.any():
        i = int(np.argmax(err - tol))
        return False, f"row {i}: |mean(y)|={err[i]:.3e} > tol {tol[i]:.3e}"
    return True, "row means of y are ~0"


def unit_scale(x, y):
    # mean(y^2) must equal v/(v+1e-5), v the row's population variance
    p = _form_problem(x, y)
    if p:
        return False, p
    _, _, v, s, M, n = _stats(x)
    q = v / (v + EPS_LN)
    m2 = (y.astype(np.float64) ** 2).mean(axis=1)
    E = _eel(s, M, n)
    tol = 2e-5 + 2.0 * RHO * q + 2.0 * np.sqrt(q) * E + E * E
    err = np.abs(m2 - q)
    bad = err > tol
    if bad.any():
        i = int(np.argmax(err - tol))
        return False, f"row {i}: mean(y^2)={m2[i]:.6e}, expected {q[i]:.6e}, tol {tol[i]:.3e}"
    return True, "second moment matches v/(v+1e-5)"


def affine_in_x(x, y):
    # within a row y is an affine function of x, with non-negative slope
    p = _form_problem(x, y)
    if p:
        return False, p
    _, xc, _, s, M, n = _stats(x)
    y64 = y.astype(np.float64)
    yc = y64 - y64.mean(axis=1, keepdims=True)
    E = _eel(s, M, n)
    ymax = np.abs(y64).max(axis=1)
    sxx = (xc * xc).sum(axis=1)
    sxy = (xc * yc).sum(axis=1)
    for i in range(x.shape[0]):
        tol = 2.0 * E[i] + RHO * ymax[i]
        if sxx[i] <= 0.0:
            if ymax[i] > E[i]:
                return False, f"row {i}: constant input but max|y|={ymax[i]:.3e} > {E[i]:.3e}"
            continue
        b = sxy[i] / sxx[i]
        res = np.abs(yc[i] - b * xc[i]).max()
        if res > tol:
            return False, f"row {i}: not affine in x, residual {res:.3e} > tol {tol:.3e}"
        slack = np.sqrt(n) * tol / np.sqrt(sxx[i])
        if b < -slack:
            return False, f"row {i}: negative slope {b:.3e} (slack {slack:.3e})"
    return True, "y is affine, non-decreasing in x"


def shift_invariance(x0, y0, x1, y1):
    p = _form_problem(x0, y0) or _form_problem(x1, y1)
    if p:
        return False, p
    _, _, _, s0, M0, n = _stats(x0)
    _, _, _, s1, M1, _ = _stats(x1)
    s = np.minimum(s0, s1)
    M = np.maximum(M0, M1)
    E = _eel(s, M, n)[:, None]
    a = y0.astype(np.float64)
    b = y1.astype(np.float64)
    tol = E + 2.0 * RHO * np.abs(a)
    d = np.abs(a - b)
    bad = d > tol
    if bad.any():
        i, j = np.unravel_index(int(np.argmax(d - tol)), d.shape)
        return False, f"[{i},{j}]: |y1-y0|={d[i, j]:.3e} > tol {tol[i, j]:.3e}"
    return True, "output invariant to per-row shift"


CASE_LAWS = {
    "well_formed": well_formed,
    "zero_mean": zero_mean,
    "unit_scale": unit_scale,
    "affine_in_x": affine_in_x,
}
SHIFT_LAWS = {"shift_invariance": shift_invariance}
