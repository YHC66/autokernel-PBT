/- E5 preamble for softmax: fixed by the loop, prepended to the agent's `Laws.lean`. Do not edit. -/
open Std Lean.Grind

/-- An exponential, as an opaque primitive. -/
class HasExp (F : Type) where
  exp : F → F

/-- The facts about `exp` a proof may use. -/
class ExpLaws (F : Type) [Field F] [LT F] [HasExp F] where
  exp_pos : ∀ x : F, 0 < HasExp.exp x
  exp_add : ∀ x y : F, HasExp.exp (x + y) = HasExp.exp x * HasExp.exp y
  exp_inj : ∀ x y : F, HasExp.exp x = HasExp.exp y → x = y

namespace E5

open HasExp

/-- Σ_j exp x_j. -/
def partition {F : Type} [Field F] [HasExp F] (xs : List F) : F := (xs.map exp).sum

/-- Row-wise softmax: the task's function on one row. -/
def softmaxRow {F : Type} [Field F] [HasExp F] (xs : List F) : List F :=
  xs.map (fun x => exp x / partition xs)
