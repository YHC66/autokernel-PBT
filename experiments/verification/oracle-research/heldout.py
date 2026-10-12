"""The held-out set: wrong kernels no agent saw, and correct rewrites every bundle must pass.

Evaluation data for ``docs/measurements/2026-10-02-prover-in-the-toolkit-prereg.md``. Every
kernel takes the reference's signature. "Wrong" rows are wrong by far more than the rounding
budget on at least one ladder group (the evaluator asserts this against the reference arm, so a
row the harness cannot tell from the reference is never scored). "Correct" rows are
algebraically the reference and numerically different: the soundness side.

Keep this file out of any agent's sandbox.
"""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- softmax: wrong
def softmax_beta(beta: float):
    b = np.float32(beta)

    def softmax(x: np.ndarray) -> np.ndarray:
        z = np.multiply(x, b, dtype=np.float32)
        z = z - np.max(z, axis=-1, keepdims=True)
        e = np.exp(z)
        return e / np.sum(e, axis=-1, keepdims=True)

    return softmax


def softmax_fill(x: np.ndarray) -> np.ndarray:
    return np.full(x.shape, 1.0 / x.shape[-1], dtype=x.dtype)


def _softmax(x: np.ndarray) -> np.ndarray:
    wide = np.asarray(x, dtype=np.float64)
    e = np.exp(wide - np.max(wide, axis=-1, keepdims=True))
    return (e / np.sum(e, axis=-1, keepdims=True)).astype(x.dtype)


def softmax_cols_reversed(x: np.ndarray) -> np.ndarray:
    return _softmax(x)[..., ::-1].copy()


def softmax_rows_rolled(x: np.ndarray) -> np.ndarray:
    return np.roll(_softmax(x), 1, axis=0)


def softmax_wrong_row(x: np.ndarray) -> np.ndarray:
    """Output row i is the softmax of input row i+1 (mod m): outputs paired with the wrong input."""
    return _softmax(np.roll(x, -1, axis=0))


def softmax_cauchy(x: np.ndarray) -> np.ndarray:
    """y ∝ 1 / (1 + max x − x): simplex rows, shift invariant, not an exponential."""
    wide = np.asarray(x, dtype=np.float64)
    w = 1.0 / (1.0 + np.max(wide, axis=-1, keepdims=True) - wide)
    return (w / np.sum(w, axis=-1, keepdims=True)).astype(x.dtype)


def softmax_gaussian(x: np.ndarray) -> np.ndarray:
    """y ∝ exp(−(x − max x)²): simplex rows, shift invariant, an exponential of the wrong thing."""
    wide = np.asarray(x, dtype=np.float64)
    d = wide - np.max(wide, axis=-1, keepdims=True)
    w = np.exp(-(d * d))
    return (w / np.sum(w, axis=-1, keepdims=True)).astype(x.dtype)


def softmax_sorted(x: np.ndarray) -> np.ndarray:
    """softmax of the sorted row: right multiset of values, wrong positions."""
    return _softmax(np.sort(x, axis=-1))


# --------------------------------------------------------------------------- softmax: correct
def softmax_exp2(x: np.ndarray) -> np.ndarray:
    m = np.max(x, axis=-1, keepdims=True)
    e = np.exp2((x - m) * np.float32(1.4426950408889634))
    return e / np.sum(e, axis=-1, keepdims=True)


def softmax_reciprocal(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    inv = np.float32(1.0) / np.sum(e, axis=-1, keepdims=True)
    return e * inv


def softmax_float32(x: np.ndarray) -> np.ndarray:
    """Everything in float32, no widening: the loop's own softmax."""
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)


def softmax_two_pass_sum(x: np.ndarray) -> np.ndarray:
    """Sum accumulated sequentially in float32 (worst rounding order), then divided."""
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    s = np.zeros(e.shape[:-1] + (1,), dtype=e.dtype)
    for j in range(e.shape[-1]):
        s = s + e[..., j : j + 1]
    return e / s


# --------------------------------------------------------------------------- layernorm: wrong
EPS = 1e-5


def _layernorm(x: np.ndarray) -> np.ndarray:
    wide = np.asarray(x, dtype=np.float64)
    c = wide - np.mean(wide, axis=-1, keepdims=True)
    v = np.mean(c * c, axis=-1, keepdims=True)
    return (c / np.sqrt(v + EPS)).astype(x.dtype)


def layernorm_negated(x: np.ndarray) -> np.ndarray:
    return -_layernorm(x)


def layernorm_cols_reversed(x: np.ndarray) -> np.ndarray:
    return _layernorm(x)[..., ::-1].copy()


def layernorm_rows_rolled(x: np.ndarray) -> np.ndarray:
    return np.roll(_layernorm(x), 1, axis=0)


def layernorm_sorted(x: np.ndarray) -> np.ndarray:
    return np.sort(_layernorm(x), axis=-1)


def layernorm_wrong_row(x: np.ndarray) -> np.ndarray:
    return _layernorm(np.roll(x, -1, axis=0))


def layernorm_householder(x: np.ndarray) -> np.ndarray:
    """H · LN(x) per row, H = I − 2vvᵀ with v ⊥ 1, ‖v‖ = 1. Preserves zero mean (H1 = 1) and unit
    variance (orthogonal). Not a permutation. Undefined (identity) on rows of width < 2."""
    y = _layernorm(x).astype(np.float64)
    n = y.shape[-1]
    if n < 2:
        return y.astype(x.dtype)
    v = np.zeros(n)
    v[0], v[1] = 1.0, -1.0
    if n >= 3:
        v[2] = 0.5
        v -= v.mean()  # keep v ⊥ 1 after the tweak
    v /= np.linalg.norm(v)
    H = np.eye(n) - 2.0 * np.outer(v, v)
    return (y @ H.T).astype(x.dtype)


# --------------------------------------------------------------------------- layernorm: correct
def layernorm_rsqrt(x: np.ndarray) -> np.ndarray:
    c = x - np.mean(x, axis=-1, keepdims=True)
    v = np.mean(c * c, axis=-1, keepdims=True)
    return c * (np.float32(1.0) / np.sqrt(v + np.float32(EPS)))


def layernorm_float32(x: np.ndarray) -> np.ndarray:
    c = x - np.mean(x, axis=-1, keepdims=True)
    v = np.mean(c * c, axis=-1, keepdims=True)
    return c / np.sqrt(v + np.float32(EPS))


def layernorm_one_pass(x: np.ndarray) -> np.ndarray:
    """Variance as E[x²] − E[x]², in float64 (where it does not cancel on this domain)."""
    wide = np.asarray(x, dtype=np.float64)
    m = np.mean(wide, axis=-1, keepdims=True)
    v = np.mean(wide * wide, axis=-1, keepdims=True) - m * m
    return ((wide - m) / np.sqrt(v + EPS)).astype(x.dtype)


def layernorm_sequential(x: np.ndarray) -> np.ndarray:
    """Mean and variance accumulated sequentially in float32."""
    n = x.shape[-1]
    s = np.zeros(x.shape[:-1] + (1,), dtype=x.dtype)
    for j in range(n):
        s = s + x[..., j : j + 1]
    m = s / np.float32(n)
    c = x - m
    q = np.zeros_like(s)
    for j in range(n):
        q = q + c[..., j : j + 1] * c[..., j : j + 1]
    return c / np.sqrt(q / np.float32(n) + np.float32(EPS))


WRONG = {
    "softmax": {
        "beta_2": softmax_beta(2.0),
        "beta_half": softmax_beta(0.5),
        "beta_minus_1": softmax_beta(-1.0),
        "beta_0": softmax_beta(0.0),
        "fill": softmax_fill,
        "cols_reversed": softmax_cols_reversed,
        "rows_rolled": softmax_rows_rolled,
        "wrong_row": softmax_wrong_row,
        "cauchy": softmax_cauchy,
        "gaussian": softmax_gaussian,
        "sorted": softmax_sorted,
    },
    "layernorm": {
        "negated": layernorm_negated,
        "cols_reversed": layernorm_cols_reversed,
        "rows_rolled": layernorm_rows_rolled,
        "sorted": layernorm_sorted,
        "wrong_row": layernorm_wrong_row,
        "householder": layernorm_householder,
    },
}

CORRECT = {
    "softmax": {
        "exp2": softmax_exp2,
        "reciprocal": softmax_reciprocal,
        "float32": softmax_float32,
        "sequential_sum": softmax_two_pass_sum,
    },
    "layernorm": {
        "rsqrt": layernorm_rsqrt,
        "float32": layernorm_float32,
        "one_pass": layernorm_one_pass,
        "sequential": layernorm_sequential,
    },
}
