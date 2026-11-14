/- E5 preamble for layernorm: fixed by the loop, prepended to the agent's `Laws.lean`. Do not edit.
Over the reals, layernorm has no epsilon: rows with positive variance are normalized exactly. -/
open Std Lean.Grind

/-- A square root, as an opaque primitive. -/
class HasSqrt (F : Type) where
  sqrt : F → F

/-- The facts about `sqrt` a proof may use. -/
class SqrtLaws (F : Type) [Field F] [LE F] [LT F] [HasSqrt F] where
  sqrt_pos : ∀ x : F, 0 < x → 0 < HasSqrt.sqrt x
  sqrt_mul_self : ∀ x : F, 0 ≤ x → HasSqrt.sqrt x * HasSqrt.sqrt x = x

namespace E5

open HasSqrt

/-- The row length n, as an element of the field (a sum of ones). -/
def count {F : Type} [Field F] (xs : List F) : F := (xs.map (fun _ => (1 : F))).sum

/-- The row mean, Σ x_j / n. -/
def mean {F : Type} [Field F] (xs : List F) : F := xs.sum / count xs

/-- The population variance, Σ (x_j − mean)² / n. -/
def var {F : Type} [Field F] (xs : List F) : F :=
  mean (xs.map (fun x => (x - mean xs) * (x - mean xs)))

/-- Row-wise layer normalization without affine parameters: the task's function on one row. -/
def lnRow {F : Type} [Field F] [HasSqrt F] (xs : List F) : List F :=
  xs.map (fun x => (x - mean xs) / sqrt (var xs))
