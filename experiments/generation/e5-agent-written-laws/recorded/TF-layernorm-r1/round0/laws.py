import numpy as np

E32 = 2.0 ** -23
EPS = 1e-5


def _valid(x, y):
    y = np.asarray(y)
    return (
        y.shape == np.asarray(x).shape
        and y.dtype == np.float32
        and bool(np.all(np.isfinite(y)))
    )


def _prep(x, y):
    x64 = np.asarray(x).astype(np.float64)
    y64 = np.asarray(y).astype(np.float64)
    n = x64.shape[1]
    m = x64.mean(axis=1)
    d = x64 - m[:, None]
    v = (d * d).mean(axis=1)
    A = np.abs(x64).max(axis=1)
    R = np.sqrt(m * m + v)
    g = 16.0 + 2.0 * np.sqrt(n)
    h = 8.0 + np.sqrt(n)
    den = np.sqrt(v + EPS)
    delta = E32 * g * R + E32 * A
    rho = E32 * h * R * R / (v + EPS) + (delta / den) ** 2 + 16.0 * E32
    eps_r = 32.0 * E32 * A / den
    return x64, y64, m, d, v, den, delta, rho, eps_r


def shape_dtype_finite(x, y):
    if not _valid(x, y):
        return False, "output has wrong shape, dtype or non-finite values"
    return True, "ok"


def zero_mean(x, y):
    if not _valid(x, y):
        return False, "invalid output"
    x64, y64, m, d, v, den, delta, rho, eps_r = _prep(x, y)
    ym = y64.mean(axis=1)
    tol = 2.0 * (delta / den * (1.0 + rho) + eps_r)
    bad = np.abs(ym) > tol
    if np.any(bad):
        i = int(np.argmax(np.abs(ym) - tol))
        return False, f"row {i}: mean of output {ym[i]:.3e} exceeds {tol[i]:.3e}"
    return True, "ok"


def affine_positive(x, y):
    if not _valid(x, y):
        return False, "invalid output"
    x64, y64, m, d, v, den, delta, rho, eps_r = _prep(x, y)
    n = x64.shape[1]
    yc = y64 - y64.mean(axis=1, keepdims=True)
    sxx = n * v
    safe = np.where(sxx > 0, sxx, 1.0)
    shat = np.where(sxx > 0, (yc * d).sum(axis=1) / safe, 0.0)
    res = yc - shat[:, None] * d
    rms = np.sqrt((res * res).mean(axis=1))
    tol = 2.0 * eps_r
    bad = rms > tol
    if np.any(bad):
        i = int(np.argmax(rms - tol))
        return False, f"row {i}: output not affine in input, residual {rms[i]:.3e} > {tol[i]:.3e}"
    sd = np.sqrt(v)
    neg = (shat * sd) < -2.0 * eps_r
    if np.any(neg):
        i = int(np.argmax(-(shat * sd) - 2.0 * eps_r))
        return False, f"row {i}: output slope on input is negative"
    return True, "ok"


def unit_scale(x, y):
    if not _valid(x, y):
        return False, "invalid output"
    x64, y64, m, d, v, den, delta, rho, eps_r = _prep(x, y)
    q = np.sqrt((y64 * y64).mean(axis=1))
    T = np.sqrt(v / (v + EPS))
    tol = 2.0 * (T * rho + eps_r + delta / den)
    bad = np.abs(q - T) > tol
    if np.any(bad):
        i = int(np.argmax(np.abs(q - T) - tol))
        return False, f"row {i}: output rms {q[i]:.6g} but expected {T[i]:.6g} (tol {tol[i]:.3e})"
    return True, "ok"


CASE_LAWS = {
    "shape_dtype_finite": shape_dtype_finite,
    "zero_mean": zero_mean,
    "affine_positive": affine_positive,
    "unit_scale": unit_scale,
}

SHIFT_LAWS = {}
