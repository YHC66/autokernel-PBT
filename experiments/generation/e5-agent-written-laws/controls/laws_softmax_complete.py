"""Scripted control: the weak laws plus a log-ratio law. Expected to reject all six softmax families."""
import numpy as np

from_weak = None
EPS = 2.0 ** -23
TINY = float(np.finfo(np.float32).tiny)


def _shape(x, y):
    return isinstance(y, np.ndarray) and y.shape == x.shape and y.dtype == np.float32


def finite_unit(x, y):
    if not _shape(x, y):
        return False, "shape or dtype"
    return bool(np.all(np.isfinite(y)) and np.all(y >= 0) and np.all(y <= 1)), "range"


def rows_sum_to_one(x, y):
    if not _shape(x, y):
        return False, "shape or dtype"
    s = np.sum(y.astype(np.float64), axis=-1)
    tol = 30 * EPS * np.sqrt(x.shape[-1])
    return bool(np.all(np.abs(s - 1) <= tol)), f"max |sum-1| {np.max(np.abs(s - 1)):.3g}"


def log_ratio(x, y):
    """log y - x is constant along each row, where y is a normal float."""
    if not _shape(x, y):
        return False, "shape or dtype"
    x64, y64 = x.astype(np.float64), y.astype(np.float64)
    for i in range(x.shape[0]):
        keep = y64[i] > TINY
        if keep.sum() < 2:
            continue
        r = np.log(y64[i][keep]) - x64[i][keep]
        spread = r.max() - r.min()
        rng = x64[i].max() - x64[i].min()
        if spread > 30 * EPS * (1 + rng) + 1e-6 * 0:
            return False, f"row {i} spread {spread:.3g}"
    return True, ""


def shift_invariance(x0, y0, x1, y1):
    if not (_shape(x0, y0) and _shape(x1, y1)):
        return False, "shape or dtype"
    d = np.max(np.abs(y1.astype(np.float64) - y0))
    return bool(d <= 30 * EPS * (1 + np.max(np.abs(x1 - x0)))), f"max diff {d:.3g}"


CASE_LAWS = {"finite_unit": finite_unit, "rows_sum_to_one": rows_sum_to_one, "log_ratio": log_ratio}
SHIFT_LAWS = {"shift_invariance": shift_invariance}
