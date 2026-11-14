"""Scripted control: the current bundle's kind of laws. Expected to admit the beta families."""
import numpy as np

EPS = np.float32(2.0 ** -23)


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


def shift_invariance(x0, y0, x1, y1):
    if not (_shape(x0, y0) and _shape(x1, y1)):
        return False, "shape or dtype"
    d = np.max(np.abs(y1.astype(np.float64) - y0))
    return bool(d <= 30 * EPS * (1 + np.max(np.abs(x1 - x0)))), f"max diff {d:.3g}"


CASE_LAWS = {"finite_unit": finite_unit, "rows_sum_to_one": rows_sum_to_one}
SHIFT_LAWS = {"shift_invariance": shift_invariance}
