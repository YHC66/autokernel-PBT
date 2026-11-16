variable {F : Type} [Field F] [LE F] [LT F] [LawfulOrderLT F] [IsLinearOrder F] [OrderedRing F] [HasExp F] [ExpLaws F]

def Laws (f : List F → List F) : Prop :=
  (∀ xs : List F, (f xs).length = xs.length) ∧
  (∀ xs : List F, ∀ v ∈ f xs, 0 ≤ v) ∧
  (∀ xs : List F, xs ≠ [] → (f xs).sum = 1) ∧
  (∀ (xs : List F) (i j : Nat) (hi : i < xs.length) (hj : j < xs.length)
      (hi' : i < (f xs).length) (hj' : j < (f xs).length),
      (f xs)[i] * HasExp.exp xs[j] = (f xs)[j] * HasExp.exp xs[i]) ∧
  (∀ (xs : List F) (c : F), f (xs.map (fun x => x + c)) = f xs)

theorem sum_map_mul_left (l : List F) (a : F) :
    (l.map (fun e => a * e)).sum = a * l.sum := by
  induction l with
  | nil => simp only [List.map_nil, List.sum_nil]; grind
  | cons h t ih => simp only [List.map_cons, List.sum_cons, ih]; grind

theorem sum_map_mul_right (l : List F) (a : F) :
    (l.map (fun e => e * a)).sum = l.sum * a := by
  induction l with
  | nil => simp only [List.map_nil, List.sum_nil]; grind
  | cons h t ih => simp only [List.map_cons, List.sum_cons, ih]; grind

theorem div_of_mul_eq {a b c : F} (hb : b ≠ 0) (h : a * b = c) : a = c / b := by
  first
  | grind
  | (subst h
     simp [Lean.Grind.Field.div_eq_mul_inv, Lean.Grind.Semiring.mul_assoc,
       Lean.Grind.Field.mul_inv_cancel hb])

theorem uniqueness (f : List F → List F) (h : Laws f) (xs : List F) (hne : xs ≠ []) :
    f xs = softmaxRow xs := by
  unfold Laws at h
  obtain ⟨hlen, _hnn, hsum, hratio, _hshift⟩ := h
  have hy : (f xs).length = xs.length := hlen xs
  have hs : (f xs).sum = 1 := hsum xs hne
  have key : ∀ (i : Nat) (hi : i < xs.length),
      (f xs)[i]'(by omega) * partition xs = HasExp.exp xs[i] := by
    intro i hi
    have hi' : i < (f xs).length := by omega
    have hL : (xs.map HasExp.exp).map (fun e => (f xs)[i] * e)
        = (f xs).map (fun v => v * HasExp.exp xs[i]) := by
      apply List.ext_getElem
      · simp [hy]
      · intro j h1 h2
        have hj : j < xs.length := by simpa using h1
        have hj' : j < (f xs).length := by omega
        have := hratio xs i j hi hj hi' hj'
        simpa [List.getElem_map] using this
    have hsumL := congrArg List.sum hL
    rw [sum_map_mul_left, sum_map_mul_right] at hsumL
    rw [hs] at hsumL
    have hp : partition xs = (xs.map HasExp.exp).sum := rfl
    rw [hp]
    grind
  have hx : 0 < xs.length := by
    first
    | exact List.length_pos_of_ne_nil hne
    | exact List.length_pos_iff.mpr hne
    | (cases xs <;> simp_all)
  have hne0 : partition xs ≠ 0 := by
    intro h0
    have k := key 0 hx
    rw [h0] at k
    have p := ExpLaws.exp_pos (xs[0]'hx)
    have h2 : HasExp.exp (xs[0]'hx) = 0 := by rw [← k]; grind
    rw [h2] at p
    first
    | grind
    | exact absurd p (Std.lt_irrefl _)
    | simp at p
  apply List.ext_getElem
  · simp [softmaxRow, hy]
  · intro i h1 h2
    have hi : i < xs.length := by omega
    simp only [softmaxRow, List.getElem_map]
    exact div_of_mul_eq hne0 (key i hi)
