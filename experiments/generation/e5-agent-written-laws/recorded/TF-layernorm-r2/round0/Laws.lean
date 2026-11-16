variable {F : Type} [Field F] [LE F] [LT F] [LawfulOrderLT F] [IsLinearOrder F] [OrderedRing F] [HasSqrt F] [SqrtLaws F]

/-- The laws, over the reals, on rows of positive variance. -/
def Laws (f : List F → List F) : Prop :=
  (∀ xs : List F, 0 < var xs →
    (f xs).length = xs.length ∧
    (f xs).sum = 0 ∧
    ((f xs).map (fun y => y * y)).sum = count xs ∧
    ∃ a b : F, 0 < a ∧ f xs = xs.map (fun x => a * x + b)) ∧
  (∀ (xs : List F) (c : F), 0 < var xs → f (xs.map (fun x => x + c)) = f xs)

theorem count_nil : count ([] : List F) = 0 := by
  simp only [count, List.map_nil, List.sum_nil]

theorem count_cons (x : F) (t : List F) : count (x :: t) = 1 + count t := by
  simp only [count, List.map_cons, List.sum_cons]

theorem count_map (g : F → F) : ∀ l : List F, count (l.map g) = count l
  | [] => by simp only [List.map_nil]
  | x :: t => by rw [List.map_cons, count_cons, count_cons, count_map g t]

theorem count_nonneg : ∀ l : List F, 0 ≤ count l
  | [] => by rw [count_nil]; grind
  | x :: t => by
      have := count_nonneg t
      rw [count_cons]
      grind

theorem mean_eq (l : List F) : mean l = l.sum / count l := rfl

theorem sum_aff (a b : F) : ∀ l : List F,
    (l.map (fun x => a * x + b)).sum = a * l.sum + count l * b
  | [] => by
      simp only [List.map_nil, List.sum_nil, count_nil]
      grind
  | x :: t => by
      have ih := sum_aff a b t
      simp only [List.map_cons, List.sum_cons, count_cons]
      grind

theorem sum_scale (k : F) : ∀ l : List F, (l.map (fun z => k * z)).sum = k * l.sum
  | [] => by
      simp only [List.map_nil, List.sum_nil]
      grind
  | x :: t => by
      have ih := sum_scale k t
      simp only [List.map_cons, List.sum_cons]
      grind

theorem div_mul_cancel_of_ne (p q : F) (hq : q ≠ 0) : p / q * q = p := by
  first
  | grind
  | (rw [Field.div_eq_mul_inv]
     have h1 := Field.mul_inv_cancel hq
     grind)

theorem cancel_left (p q : F) (hq : q ≠ 0) (h : q * p = 0) : p = 0 := by
  first
  | grind
  | (have h1 : q * q⁻¹ = 1 := Field.mul_inv_cancel hq
     calc p = (q * q⁻¹) * p := by rw [h1]; grind
       _ = q⁻¹ * (q * p) := by grind
       _ = 0 := by rw [h]; grind)

theorem div_eq_of_mul (a s p : F) (hs : s ≠ 0) (h : a * s = 1) : p / s = a * p := by
  first
  | grind
  | (rw [Field.div_eq_mul_inv]
     have h1 := Field.mul_inv_cancel hs
     grind)

theorem count_ne_zero (xs : List F) (hv : 0 < var xs) : count xs ≠ 0 := by
  cases xs with
  | nil =>
    have h0 : var ([] : List F) = 0 := by
      simp only [var, mean, List.map_nil, List.sum_nil, count_nil]
      grind
    grind
  | cons x t =>
    have := count_nonneg t
    rw [count_cons]
    grind

theorem uniqueness (f : List F → List F) (h : Laws f) (xs : List F) (hv : 0 < var xs) :
    f xs = lnRow xs := by
  unfold Laws at h
  obtain ⟨h1, _⟩ := h
  obtain ⟨_, hsum, hsq, a, b, ha, hf⟩ := h1 xs hv
  have hn : count xs ≠ 0 := count_ne_zero xs hv
  have hm : mean xs * count xs = xs.sum := by
    rw [mean_eq]
    exact div_mul_cancel_of_ne _ _ hn
  have hvn : var xs * count xs = (xs.map (fun x => (x - mean xs) * (x - mean xs))).sum := by
    have e : var xs = (xs.map (fun x => (x - mean xs) * (x - mean xs))).sum / count xs := by
      show mean (xs.map (fun x => (x - mean xs) * (x - mean xs))) = _
      rw [mean_eq (xs.map (fun x => (x - mean xs) * (x - mean xs))), count_map]
    rw [e]
    exact div_mul_cancel_of_ne _ _ hn
  have hsum' := hsum
  rw [hf, sum_aff a b xs] at hsum'
  have hb : b = -(a * mean xs) := by
    have h2 : count xs * (a * mean xs + b) = 0 := by grind
    have h3 := cancel_left _ _ hn h2
    grind
  have hsq' : a * a * (xs.map (fun x => (x - mean xs) * (x - mean xs))).sum = count xs := by
    have e : (f xs).map (fun y => y * y)
        = (xs.map (fun x => (x - mean xs) * (x - mean xs))).map (fun z => (a * a) * z) := by
      rw [hf, List.map_map, List.map_map]
      apply List.map_congr_left
      intro x _
      show (a * x + b) * (a * x + b) = (a * a) * ((x - mean xs) * (x - mean xs))
      rw [hb]
      grind
    rw [e, sum_scale] at hsq
    exact hsq
  have hs : 0 < HasSqrt.sqrt (var xs) := SqrtLaws.sqrt_pos (var xs) hv
  have hss : HasSqrt.sqrt (var xs) * HasSqrt.sqrt (var xs) = var xs :=
    SqrtLaws.sqrt_mul_self (var xs) (by grind)
  have hs0 : HasSqrt.sqrt (var xs) ≠ 0 := by grind
  have has : a * HasSqrt.sqrt (var xs) = 1 := by
    have h1 : (a * a * var xs - 1) * count xs = 0 := by grind
    have h2 : a * a * var xs - 1 = 0 := cancel_left _ _ hn (by grind)
    have h3 : (a * HasSqrt.sqrt (var xs)) * (a * HasSqrt.sqrt (var xs)) = 1 := by grind
    have h4 : 0 < a * HasSqrt.sqrt (var xs) := by
      first
      | grind
      | (have h := OrderedRing.mul_lt_mul_of_pos_left hs ha
         grind)
    have h5 : a * HasSqrt.sqrt (var xs) - 1 = 0 :=
      cancel_left _ (a * HasSqrt.sqrt (var xs) + 1) (by grind) (by grind)
    grind
  rw [hf]
  unfold lnRow
  apply List.map_congr_left
  intro x _
  show a * x + b = (x - mean xs) / HasSqrt.sqrt (var xs)
  rw [div_eq_of_mul a _ (x - mean xs) hs0 has, hb]
  grind
