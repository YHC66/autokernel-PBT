
end E5

/-- Added by the loop: the obligation, stated once, here. -/
theorem e5_obligation {F : Type} [Field F] [LE F] [LT F] [LawfulOrderLT F] [IsLinearOrder F]
    [OrderedRing F] [HasSqrt F] [SqrtLaws F] (f : List F → List F) (h : E5.Laws f) :
    ∀ xs : List F, 0 < E5.var xs → f xs = E5.lnRow xs :=
  fun xs hv => E5.uniqueness f h xs hv

#print axioms e5_obligation
