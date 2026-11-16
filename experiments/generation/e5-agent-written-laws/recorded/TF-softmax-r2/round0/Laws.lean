variable {F : Type} [Field F] [LE F] [LT F] [LawfulOrderLT F] [IsLinearOrder F] [OrderedRing F] [HasExp F] [ExpLaws F]

def Laws (f : List F → List F) : Prop :=
  ∀ xs : List F, xs ≠ [] →
    (f xs).length = xs.length ∧
    (∀ y ∈ f xs, 0 ≤ y) ∧
    (f xs).sum = 1 ∧
    (∀ p ∈ (f xs).zip xs, ∀ q ∈ (f xs).zip xs,
      p.1 * HasExp.exp q.2 = q.1 * HasExp.exp p.2)

theorem aux_list (c : F) : ∀ (xs ys : List F), ys.length = xs.length →
    (∀ p ∈ ys.zip xs, p.1 = c * HasExp.exp p.2) →
    ys = xs.map (fun x => c * HasExp.exp x) := by
  intro xs
  induction xs with
  | nil =>
    intro ys hl _
    cases ys with
    | nil => rfl
    | cons y ys =>
      exact absurd hl (by simp only [List.length_nil, List.length_cons]; omega)
  | cons x xs ih =>
    intro ys hl h
    cases ys with
    | nil =>
      exact absurd hl (by simp only [List.length_nil, List.length_cons]; omega)
    | cons y ys =>
      have hl' : ys.length = xs.length := by
        simp only [List.length_cons] at hl
        omega
      have h1 : y = c * HasExp.exp x := h (y, x) (by simp [List.zip_cons_cons])
      have h2 : ys = xs.map (fun x => c * HasExp.exp x) :=
        ih ys hl' (fun p hp => h p (by simp [List.zip_cons_cons, hp]))
      subst h1
      subst h2
      rfl

theorem sum_map_mul (c : F) : ∀ xs : List F,
    (xs.map (fun x => c * HasExp.exp x)).sum = c * (xs.map HasExp.exp).sum := by
  intro xs
  induction xs with
  | nil =>
    simp only [List.map_nil, List.sum_nil]
    grind
  | cons x xs ih =>
    simp only [List.map_cons, List.sum_cons]
    rw [ih]
    grind

theorem div_aux (c Z e : F) (h : c * Z = 1) : c * e = e / Z := by
  have hZ : Z ≠ 0 := by
    intro h0
    rw [h0] at h
    grind
  have hinv : Z * Z⁻¹ = 1 := by
    first
      | exact Field.mul_inv_cancel hZ
      | grind
  first
    | (rw [Field.div_eq_mul_inv]; grind)
    | grind

theorem core (xs ys : List F) (hne : xs ≠ []) (hlen : ys.length = xs.length)
    (hsum : ys.sum = 1)
    (hrat : ∀ p ∈ ys.zip xs, ∀ q ∈ ys.zip xs,
      p.1 * HasExp.exp q.2 = q.1 * HasExp.exp p.2) :
    ys = softmaxRow xs := by
  cases xs with
  | nil => exact absurd rfl hne
  | cons x0 xs' =>
    cases ys with
    | nil =>
      exact absurd hlen (by simp only [List.length_nil, List.length_cons]; omega)
    | cons y0 ys' =>
      have hp0 : (y0, x0) ∈ (y0 :: ys').zip (x0 :: xs') := by
        simp [List.zip_cons_cons]
      have hpos : 0 < HasExp.exp x0 := ExpLaws.exp_pos x0
      have hne0 : HasExp.exp x0 ≠ 0 := by
        intro h0
        rw [h0] at hpos
        first
          | exact absurd hpos (Std.lt_irrefl _)
          | grind
      have hinv : HasExp.exp x0 * (HasExp.exp x0)⁻¹ = 1 := by
        first
          | exact Field.mul_inv_cancel hne0
          | grind
      obtain ⟨c, hc⟩ : ∃ c : F, c = y0 / HasExp.exp x0 := ⟨_, rfl⟩
      have hall : ∀ p ∈ (y0 :: ys').zip (x0 :: xs'), p.1 = c * HasExp.exp p.2 := by
        intro p hp
        have h1 : p.1 * HasExp.exp x0 = y0 * HasExp.exp p.2 := hrat p hp (y0, x0) hp0
        first
          | (rw [hc, Field.div_eq_mul_inv]; grind)
          | (rw [hc]; grind)
          | grind
      have hlist := aux_list c (x0 :: xs') (y0 :: ys') hlen hall
      rw [hlist] at hsum
      rw [sum_map_mul] at hsum
      rw [hlist]
      unfold softmaxRow partition
      exact List.map_congr_left (fun x _ => div_aux _ _ x hsum)

theorem uniqueness (f : List F → List F) (h : Laws f) (xs : List F) (hne : xs ≠ []) :
    f xs = softmaxRow xs := by
  obtain ⟨hlen, _, hsum, hrat⟩ := h xs hne
  exact core xs (f xs) hne hlen hsum hrat
