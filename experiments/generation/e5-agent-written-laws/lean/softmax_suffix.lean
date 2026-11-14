
end E5

/-- Added by the loop: the obligation, stated once, here. -/
theorem e5_obligation {F : Type} [Field F] [LE F] [LT F] [LawfulOrderLT F] [IsLinearOrder F]
    [OrderedRing F] [HasExp F] [ExpLaws F] (f : List F → List F) (h : E5.Laws f) :
    ∀ xs : List F, xs ≠ [] → f xs = E5.softmaxRow xs :=
  fun xs hne => E5.uniqueness f h xs hne

#print axioms e5_obligation
