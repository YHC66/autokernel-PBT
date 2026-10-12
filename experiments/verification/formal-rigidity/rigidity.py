"""The rigidity audit: which reinterpretations of its primitives does an axiom set survive?

A formal proof that a kernel adheres to a spec is a proof in EVERY model of the axioms the spec's
primitives (exp, log, sqrt, ...) are given. If the axioms are preserved by some transformation of
the primitives — rescaling the argument of exp, say — then the proof is equally a proof about the
spec under that transformation, and the certificate names a *family* of functions rather than
one. Check C6 of the certificate design asks whether each axiom is TRUE on samples; this audit
asks whether the axiom set is RIGID: whether anything other than the intended interpretation also
passes. It is the sampled, language-neutral counterpart of the theorems in
``proofs/why3-real-theories/Why3RealTheories.lean``.

Every axiom is a predicate on a dictionary of primitive readings and one sample; a transformation
rewrites that dictionary. An axiom set *admits* a transformation when every axiom holds on every
sample under the transformed primitives, compared through the harness's own float64 budget as
``formal.checks._axioms`` does. The audit reports, per transformation, the first axiom that
rejects it (attribution) or ADMITTED. Sampling can confirm a symmetry only to the sampled points,
as every PBT result here; the Lean file proves the exact statements.

    python rigidity.py          prints the audit table for the axiom sets below
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

from autokernel_pbt.formal.checks import AXIOM_SAMPLES
from autokernel_pbt.formal.comparisons import Comparisons

Prims = Mapping[str, Callable[..., float]]
#: An axiom reading: (primitives, rng, comparisons) -> holds on one sample?
AxiomReading = Callable[[Prims, np.random.Generator, Comparisons], bool]
#: A reinterpretation of the primitives.
Transformation = Callable[[Prims], Prims]

#: Axioms are statements about the reals, read in float64 at float64's one-operation budget.
HARNESS_CMP = Comparisons(dtype=np.dtype(np.float64), n=1)


@dataclass(frozen=True)
class FlooredComparisons(Comparisons):
    """The harness's comparison with a unit floor on the scale: absolute in units of eps below
    |expected| = 1, relative above.

    INSTRUMENT FINDING, measured by ``spurious_rejections`` below: ``Comparisons.close`` divides
    the residual by max|expected| and falls back to 1 only when that is exactly zero. An axiom
    stated as an identity whose two sides pass *near* zero — Why3's own ``Log_mul`` at x·y ≈ 1,
    ``Cos_plus_pi`` at x ≈ π/2 — is then rejected on the TRUE primitives: the rounding of a value
    of size 1e-12 is 1e-28, but the comparison asks for 1e-12 · ε. Check C6 as built inherits
    this, so an agent whose axiom reading uses ``cmp.close`` on such an identity fails C6 for a
    true axiom. The floor is the standard mixed absolute/relative tolerance and is what this
    audit uses; the design should adopt it, or say that axioms must be stated with a side bounded
    away from zero.
    """

    def close(self, actual, expected) -> bool:
        actual = np.asarray(actual, dtype=np.float64)
        expected = np.asarray(expected, dtype=np.float64)
        if expected.ndim == 0:
            expected = np.broadcast_to(expected, actual.shape)
        if actual.shape != expected.shape or not (np.all(np.isfinite(actual)) and np.all(np.isfinite(expected))):
            return False
        scale = max(float(np.max(np.abs(expected))), 1.0)
        eps = float(np.finfo(np.float64).eps)
        return bool(float(np.max(np.abs(actual - expected))) / (scale * eps) < 30.0)


CMP = FlooredComparisons(dtype=np.dtype(np.float64), n=1)


@dataclass(frozen=True)
class AxiomSet:
    name: str
    source: str
    primitives: dict[str, Callable[..., float]]
    axioms: dict[str, AxiomReading]
    #: Transformation names this set is *expected* (by the Lean proofs) to admit; "identity" always.
    expected_admits: frozenset[str] = field(default_factory=frozenset)


@dataclass(frozen=True)
class Finding:
    transformation: str
    admitted: bool
    rejecting_axiom: str  # "" if admitted
    samples: int


def audit(axioms: AxiomSet, transformations: Mapping[str, Transformation], samples: int = AXIOM_SAMPLES, seed: int = 0, cmp: Comparisons = CMP) -> list[Finding]:
    out = []
    for tname, transform in transformations.items():
        prims = transform(axioms.primitives)
        rng = np.random.default_rng(seed)
        rejecting = ""
        for aname, reading in axioms.axioms.items():
            for _ in range(samples):
                with np.errstate(all="ignore"):
                    try:
                        ok = reading(prims, rng, cmp)
                    except (ValueError, ZeroDivisionError, OverflowError, FloatingPointError):
                        ok = False
                if not ok:
                    rejecting = aname
                    break
            if rejecting:
                break
        out.append(Finding(tname, rejecting == "", rejecting, samples))
    return out


# --------------------------------------------------------------------------- sampling helpers
def _x(rng, lo=-5.0, hi=5.0) -> float:
    return float(rng.uniform(lo, hi))


def _pos(rng, lo=1e-3, hi=50.0) -> float:
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def _on(sampler, arity: int):
    """Lift a predicate on (primitives, samples..., comparisons) to an AxiomReading that draws
    ``arity`` samples with ``sampler``. Keeps each axiom a one-line statement about its primitives."""

    def lift(predicate) -> AxiomReading:
        def reading(p: Prims, rng: np.random.Generator, c: Comparisons) -> bool:
            return bool(predicate(p, *[sampler(rng) for _ in range(arity)], c))

        return reading

    return lift


#: one or two real samples on [-5, 5]; one or two positive samples; one or two samples on [-1, 1]
real1, real2 = _on(_x, 1), _on(_x, 2)
pos1, pos2 = _on(_pos, 1), _on(_pos, 2)


# --------------------------------------------------------------------------- transformations
def identity(p: Prims) -> Prims:
    return dict(p)


def scale_exp_arg(beta: float) -> Transformation:
    """exp ↦ x ↦ exp(βx); log ↦ y ↦ log(y)/β (undefined at β = 0, where log/β is 0 as in Lean)."""

    def t(p: Prims) -> Prims:
        q = dict(p)
        e = p["exp"]
        q["exp"] = lambda x: e(beta * x)
        if "log" in p:
            lg = p["log"]
            q["log"] = (lambda y: lg(y) / beta) if beta != 0 else (lambda y: 0.0)
        return q

    return t


def negate_value(name: str) -> Transformation:
    def t(p: Prims) -> Prims:
        q = dict(p)
        f = p[name]
        q[name] = lambda x: -f(x)
        return q

    return t


def scale_value(name: str, c: float) -> Transformation:
    def t(p: Prims) -> Prims:
        q = dict(p)
        f = p[name]
        q[name] = lambda x: c * f(x)
        return q

    return t


def scale_trig_arg(beta: float) -> Transformation:
    """cos, sin ↦ cos(βx), sin(βx); atan ↦ atan(x)/β; pi unchanged."""

    def t(p: Prims) -> Prims:
        q = dict(p)
        c, s, a = p["cos"], p["sin"], p["atan"]
        q["cos"] = lambda x: c(beta * x)
        q["sin"] = lambda x: s(beta * x)
        q["atan"] = lambda x: a(x) / beta
        return q

    return t


EXP_TRANSFORMATIONS: dict[str, Transformation] = {
    "identity": identity,
    "exp(2x)": scale_exp_arg(2.0),
    "exp(x/2)": scale_exp_arg(0.5),
    "exp(-x)": scale_exp_arg(-1.0),
    "exp(0x)": scale_exp_arg(0.0),
    "2·exp(x)": scale_value("exp", 2.0),
    "-exp(x)": negate_value("exp"),
}
SQRT_TRANSFORMATIONS: dict[str, Transformation] = {
    "identity": identity,
    "-sqrt(x)": negate_value("sqrt"),
    "2·sqrt(x)": scale_value("sqrt", 2.0),
}
TRIG_TRANSFORMATIONS: dict[str, Transformation] = {
    "identity": identity,
    "cos,sin(2x)": scale_trig_arg(2.0),
    "cos,sin(3x)": scale_trig_arg(3.0),
    "cos,sin(5x)": scale_trig_arg(5.0),
    "cos,sin(-3x)": scale_trig_arg(-3.0),
    # β = 9 is a true symmetry too (9 ≡ 1 mod 4) but is NOT sampled: at |9(x + π)| ≈ 37 the
    # argument's float64 ulp is 7e-15, and cos carries that straight into the value, which the
    # 30·ε budget then rejects. The audit's reach is bounded by the conditioning of the
    # transformed arguments; the Lean file states the exact symmetry.
}

# --------------------------------------------------------------------------- axiom sets
# The paper's minimal theory (proofs/softmax-bundle/SoftmaxBundle.lean, `ExpLaws`).
EXP_LAWS = AxiomSet(
    name="ExpLaws (paper, 3 axioms)",
    source="proofs/softmax-bundle/SoftmaxBundle.lean",
    primitives={"exp": np.exp},
    axioms={
        "exp_pos": real1(lambda p, x, c: p["exp"](x) > 0),
        "exp_add": real2(lambda p, x, y, c: c.close(p["exp"](x + y), p["exp"](x) * p["exp"](y))),
        "exp_inj": real2(lambda p, x, y, c: x == y or not c.close(p["exp"](x), p["exp"](y))),
    },
    expected_admits=frozenset({"identity", "exp(2x)", "exp(x/2)", "exp(-x)"}),
)

# Why3 1.8.2 real.ExpLog, transcribed from https://www.why3.org/stdlib/real.html.
WHY3_EXPLOG_AXIOMS: dict[str, AxiomReading] = {
    "Exp_zero": lambda p, r, c: c.close(p["exp"](0.0), 1.0),
    "Exp_sum": real2(lambda p, x, y, c: c.close(p["exp"](x + y), p["exp"](x) * p["exp"](y))),
    "exp_increasing": real2(lambda p, x, y, c: c.at_most(p["exp"](min(x, y)), p["exp"](max(x, y)))),
    "exp_positive": real1(lambda p, x, c: p["exp"](x) > 0),
    "exp_inv": real1(lambda p, x, c: c.close(p["exp"](-x), 1.0 / p["exp"](x))),
    "Log_one": lambda p, r, c: c.close(p["log"](1.0), 0.0),
    "Log_mul": pos2(lambda p, x, y, c: c.close(p["log"](x * y), p["log"](x) + p["log"](y))),
    "log_increasing": pos2(lambda p, x, y, c: c.at_most(p["log"](min(x, y)), p["log"](max(x, y)))),
    "Log_exp": real1(lambda p, x, c: c.close(p["log"](p["exp"](x)), x)),
    "Exp_log": pos1(lambda p, x, c: c.close(p["exp"](p["log"](x)), x)),
}
WHY3_EXPLOG = AxiomSet(
    name="Why3 1.8.2 real.ExpLog (10 axioms)",
    source="https://www.why3.org/stdlib/real.html",
    primitives={"exp": np.exp, "log": np.log},
    axioms=WHY3_EXPLOG_AXIOMS,
    expected_admits=frozenset({"identity", "exp(2x)", "exp(x/2)"}),
)

WHY3_EXPLOG_TANGENT = AxiomSet(
    name="Why3 ExpLog + tangent line 1 + x ≤ exp x",
    source="Why3RealTheories.lean, tangent_rigid",
    primitives={"exp": np.exp, "log": np.log},
    axioms={**WHY3_EXPLOG_AXIOMS, "Tangent": real1(lambda p, x, c: c.at_most(1.0 + x, p["exp"](x)))},
    expected_admits=frozenset({"identity"}),
)

# Bounds on e in the style of Why3's own Pi_double_precision_bounds. These are the two adjacent
# doubles around e; with the rescaled exp, exp(β·1) must lie between them, so β is pinned to
# within ~1e-16 — categorical for every purpose a float kernel has.
E_LO, E_HI = np.nextafter(np.e, 0.0), np.nextafter(np.e, 10.0)
WHY3_EXPLOG_EBOUNDS = AxiomSet(
    name="Why3 ExpLog + E_double_precision_bounds",
    source="proposed, after Why3's Pi_double_precision_bounds",
    primitives={"exp": np.exp, "log": np.log},
    axioms={**WHY3_EXPLOG_AXIOMS, "E_bounds": lambda p, r, c: E_LO <= p["exp"](1.0) <= E_HI},
    expected_admits=frozenset({"identity"}),
)

# Why3 1.8.2 real.Square.
WHY3_SQUARE = AxiomSet(
    name="Why3 1.8.2 real.Square (5 axioms)",
    source="https://www.why3.org/stdlib/real.html",
    primitives={"sqrt": np.sqrt},
    axioms={
        "Sqrt_positive": pos1(lambda p, x, c: p["sqrt"](x) >= 0),
        "Sqrt_square": pos1(lambda p, x, c: c.close(p["sqrt"](x) ** 2, x)),
        "Square_sqrt": pos1(lambda p, x, c: c.close(p["sqrt"](x * x), x)),
        "Sqrt_mul": pos2(lambda p, x, y, c: c.close(p["sqrt"](x * y), p["sqrt"](x) * p["sqrt"](y))),
        "Sqrt_le": pos2(lambda p, x, y, c: c.at_most(p["sqrt"](min(x, y)), p["sqrt"](max(x, y)))),
    },
    expected_admits=frozenset({"identity"}),
)

# Why3 1.8.2 real.Trigonometry (pi is a constant with double-precision bounds; unchanged).
# Sampled on [-1, 1]: the identities hold for all reals (the Lean file proves them so), but
# under a rescaling by β the float evaluation of cos(β(x+y)) carries an argument-rounding error
# of about |β(x+y)|·ε that the value comparison cannot distinguish from a false axiom. On
# [-5, 5] the true symmetry at β = 5 is rejected for that reason; on [-1, 1] it is not.
PI = np.pi


def _t(rng) -> float:
    return _x(rng, -1.0, 1.0)


trig1, trig2 = _on(_t, 1), _on(_t, 2)

WHY3_TRIG = AxiomSet(
    name="Why3 1.8.2 real.Trigonometry (16 axioms)",
    source="https://www.why3.org/stdlib/real.html",
    primitives={"cos": np.cos, "sin": np.sin, "atan": np.arctan},
    axioms={
        "Pythagorean_identity": trig1(lambda p, x, c: c.close(p["cos"](x) ** 2 + p["sin"](x) ** 2, 1.0)),
        "Cos_0": lambda p, r, c: c.close(p["cos"](0.0), 1.0),
        "Sin_0": lambda p, r, c: c.close(p["sin"](0.0), 0.0),
        "Cos_pi": lambda p, r, c: c.close(p["cos"](PI), -1.0),
        "Sin_pi": lambda p, r, c: c.close(p["sin"](PI), 0.0),
        "Cos_pi2": lambda p, r, c: c.close(p["cos"](0.5 * PI), 0.0),
        "Sin_pi2": lambda p, r, c: c.close(p["sin"](0.5 * PI), 1.0),
        "Cos_plus_pi": trig1(lambda p, x, c: c.close(p["cos"](x + PI), -p["cos"](x))),
        "Sin_plus_pi": trig1(lambda p, x, c: c.close(p["sin"](x + PI), -p["sin"](x))),
        "Cos_plus_pi2": trig1(lambda p, x, c: c.close(p["cos"](x + 0.5 * PI), -p["sin"](x))),
        "Sin_plus_pi2": trig1(lambda p, x, c: c.close(p["sin"](x + 0.5 * PI), p["cos"](x))),
        "Cos_neg": trig1(lambda p, x, c: c.close(p["cos"](-x), p["cos"](x))),
        "Sin_neg": trig1(lambda p, x, c: c.close(p["sin"](-x), -p["sin"](x))),
        "Cos_sum": trig2(lambda p, x, y, c: c.close(p["cos"](x + y), p["cos"](x) * p["cos"](y) - p["sin"](x) * p["sin"](y))),
        "Sin_sum": trig2(lambda p, x, y, c: c.close(p["sin"](x + y), p["sin"](x) * p["cos"](y) + p["cos"](x) * p["sin"](y))),
        "Tan_atan": trig1(lambda p, x, c: c.close(p["sin"](p["atan"](x)) / p["cos"](p["atan"](x)), x)),
    },
    expected_admits=frozenset({"identity", "cos,sin(5x)", "cos,sin(-3x)"}),
)

SUITES: list[tuple[AxiomSet, dict[str, Transformation]]] = [
    (EXP_LAWS, EXP_TRANSFORMATIONS),
    (WHY3_EXPLOG, EXP_TRANSFORMATIONS),
    (WHY3_EXPLOG_TANGENT, EXP_TRANSFORMATIONS),
    (WHY3_EXPLOG_EBOUNDS, EXP_TRANSFORMATIONS),
    (WHY3_SQUARE, SQRT_TRANSFORMATIONS),
    (WHY3_TRIG, TRIG_TRANSFORMATIONS),
]


def spurious_rejections(axioms: AxiomSet, samples: int = AXIOM_SAMPLES, seed: int = 0) -> dict[str, float]:
    """Per axiom, the fraction of samples on which the harness's RAW comparison rejects the axiom
    on the TRUE primitives (identity). Anything above zero is a false rejection of a true axiom."""
    out = {}
    for aname, reading in axioms.axioms.items():
        rng = np.random.default_rng(seed)
        bad = 0
        for _ in range(samples):
            with np.errstate(all="ignore"):
                try:
                    ok = reading(axioms.primitives, rng, HARNESS_CMP)
                except (ValueError, ZeroDivisionError, OverflowError, FloatingPointError):
                    ok = False
            bad += not ok
        out[aname] = bad / samples
    return out


def main() -> None:
    print(f"numpy {np.__version__}; {AXIOM_SAMPLES} samples per axiom; float64 budget, n=1")
    for axioms, transformations in SUITES:
        t0 = time.perf_counter()
        findings = audit(axioms, transformations)
        ms = (time.perf_counter() - t0) * 1e3
        print(f"\n## {axioms.name}   [{axioms.source}]   {ms:.0f} ms")
        print("| reinterpretation | verdict | rejected by | matches Lean |")
        print("|---|---|---|---|")
        for f in findings:
            expected = f.transformation in axioms.expected_admits
            ok = "yes" if f.admitted == expected else "**NO**"
            print(f"| {f.transformation} | {'ADMITTED' if f.admitted else 'rejected'} | {f.rejecting_axiom or '—'} | {ok} |")
    print("\n## Instrument check: the harness's raw relative comparison on the TRUE primitives")
    print("(fraction of samples a true axiom is rejected on; any non-zero entry is a false rejection)")
    for axioms, _ in SUITES:
        rates = {k: v for k, v in spurious_rejections(axioms).items() if v > 0}
        shown = ", ".join(f"{k}={v:.3f}" for k, v in sorted(rates.items())) or "none"
        print(f"- {axioms.name}: {shown}")


if __name__ == "__main__":
    main()
