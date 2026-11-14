
## Also: prove your laws determine the function, in `Laws.lean`

Write `Laws.lean`: the same laws stated over the reals, and a proof that any function satisfying them *is* the
function above. If such a proof exists, no wrong kernel can satisfy all of your laws exactly.

The loop builds one file from `PREAMBLE.lean` (read it first; you cannot change it), then your `Laws.lean`, then
the obligation below, and compiles it with Lean 4 (version 4.33.0, core only: no Mathlib; `simp`, `grind`,
`omega` and the rest of core are available). Your file is placed inside `namespace E5`, which the preamble opens and
the loop closes. It must define exactly these two names:

```lean
def Laws (f : List F → List F) : Prop
theorem uniqueness (f : List F → List F) (h : Laws f) (xs : List F) {HYPOTHESIS} :
    f xs = {TARGET} xs
```

`F` is any ordered field: declare it yourself, with exactly the instance arguments the obligation uses:
`variable {F : Type} [Field F] [LE F] [LT F] [LawfulOrderLT F] [IsLinearOrder F] [OrderedRing F] {CLASSES}`.
`f` stands for the kernel on one row. `Laws f` must say what `laws.py` says, row by row, with no tolerance: a
case law becomes a statement about `f xs`, and a shift law becomes one about `f xs` and
`f (xs.map (· + c))`. Then the loop checks:

```lean
theorem e5_obligation ... (f : List F → List F) (h : E5.Laws f) :
    ∀ xs : List F, {HYPOTHESIS_FORALL} → f xs = E5.{TARGET} xs := fun xs hx => E5.uniqueness f h xs hx
#print axioms e5_obligation
```

The proof counts only if the file compiles, with no `sorry`, and `#print axioms` lists nothing beyond `propext`,
`Classical.choice` and `Quot.sound`. You get up to 4 rounds, each a fresh session. After each round, the loop puts
the compiler's output in `HISTORY.md` and keeps your files for the next one. If the proof cannot be done for your
laws as they stand, that is information: change the laws in both files.
