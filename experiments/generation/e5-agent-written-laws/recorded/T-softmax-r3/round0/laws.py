import numpy as np

EPS = 2.0 ** -23


def _structure(x, y):
    if not isinstance(y, np.ndarray):
        return False, "output is not an ndarray"
    if y.shape != x.shape:
        return False, f"shape {y.shape} != {x.shape}"
    if y.dtype != np.float32:
        return False, f"dtype {y.dtype} != float32"
    if not np.all(np.isfinite(y)):
        return False, "non-finite output"
    return True, "ok"


def well_formed(x, y):
    return _structure(x, y)


def in_unit_interval(x, y):
    ok, d = _structure(x, y)
    if not ok:
        return ok, d
    y64 = y.astype(np.float64)
    tol = 1e-5 + 2 * EPS * x.shape[1]
    if y64.min() < 0.0:
        return False, f"negative entry {y64.min():.3e}"
    if y64.max() > 1.0 + tol:
        return False, f"entry above 1: {y64.max():.9g}"
    return True, "ok"


def rows_sum_to_one(x, y):
    ok, d = _structure(x, y)
    if not ok:
        return ok, d
    s = y.astype(np.float64).sum(axis=1)
    tol = 1e-5 + 2 * EPS * x.shape[1]
    err = np.abs(s - 1.0).max()
    if err > tol:
        return False, f"row sum off by {err:.3e} (tol {tol:.3e})"
    return True, "ok"


def order_preserved(x, y):
    ok, d = _structure(x, y)
    if not ok:
        return ok, d
    if x.shape[1] < 2:
        return True, "ok"
    idx = np.argsort(x, axis=1, kind="stable")
    ys = np.take_along_axis(y.astype(np.float64), idx, axis=1)
    lo, hi = ys[:, :-1], ys[:, 1:]
    bad = hi < lo * (1.0 - 5e-4) - 1e-30
    if bad.any():
        return False, "larger input got a smaller output"
    return True, "ok"


def log_output_tracks_input(x, y):
    # log y_j - x_j must be one constant per row, for every entry whose output
    # is comfortably representable.
    ok, d = _structure(x, y)
    if not ok:
        return ok, d
    x64 = x.astype(np.float64)
    y64 = y.astype(np.float64)
    xmax = x64.max(axis=1, keepdims=True)
    sel = x64 >= xmax - 60.0
    if np.any(sel & (y64 <= 1e-35)):
        return False, "output underflowed where it should be representable"
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.log(np.where(sel, y64, 1.0)) - x64
    hi = np.where(sel, r, -np.inf).max(axis=1)
    lo = np.where(sel, r, np.inf).min(axis=1)
    spread = (hi - lo).max()
    tol = 2e-4 + EPS * x.shape[1]
    if not spread <= tol:
        return False, f"log-output minus input varies by {spread:.3e} (tol {tol:.3e})"
    return True, "ok"


def shift_invariant(x0, y0, x1, y1):
    for x, y in ((x0, y0), (x1, y1)):
        ok, d = _structure(x, y)
        if not ok:
            return ok, d
    if x0.shape != x1.shape:
        return False, "case shapes differ"
    delta = x1.astype(np.float64) - x0.astype(np.float64)
    slack = (delta.max(axis=1) - delta.min(axis=1)).max()
    tol = 2e-4 + 2 * EPS * x0.shape[1] + 2.0 * slack
    diff = np.abs(y1.astype(np.float64) - y0.astype(np.float64)).max()
    if not diff <= tol:
        return False, f"outputs moved by {diff:.3e} under a row shift (tol {tol:.3e})"
    return True, "ok"


CASE_LAWS = {
    "well_formed": well_formed,
    "in_unit_interval": in_unit_interval,
    "rows_sum_to_one": rows_sum_to_one,
    "order_preserved": order_preserved,
    "log_output_tracks_input": log_output_tracks_input,
}

SHIFT_LAWS = {"shift_invariant": shift_invariant}
