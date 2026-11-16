variable {F : Type} [Field F] [LE F] [LT F] [LawfulOrderLT F] [IsLinearOrder F] [OrderedRing F] [HasSqrt F] [SqrtLaws F]

/-- The laws, row by row, over the reals (no epsilon). -/
def Laws (f : List F → List F) : Prop :=
  (∀ xs, (f xs).length = xs.length) ∧
  (∀ xs, (f xs).sum = 0) ∧
  (∀ xs c, f (xs.map (fun x => x + c)) = f xs) ∧
  (∀ xs, ∃ k : F, 0 ≤ k ∧ f xs = xs.map (fun x => k * (x - mean xs))) ∧
  (∀ xs, mean ((f xs).map (fun y => y * y)) * var xs = var xs)

theorem sum_map_mul_left (c : F) : ∀ l : List F, (l.map (fun x => c * x)).sum = c * l.sum
  | [] => by
    simp only [List.map_nil, List.sum_nil]
    grind
  | a :: l => by
    simp only [List.map_cons, List.sum_cons]
    rw [sum_map_mul_left c l]
    grind

theorem count_map (g : F → F) (xs : List F) : count (xs.map g) = count xs := by
  unfold count
  rw [List.map_map]
  try rfl

theorem mean_map_scale (c : F) (l : List F) : mean (l.map (fun z => c * z)) = c * mean l := by
  unfold mean
  rw [count_map, sum_map_mul_left]
  grind

theorem mul_right_cancel0 {a b : F} (hb : b ≠ 0) (h : a * b = 0) : a = 0 := by
  have hinv : b * b⁻¹ = 1 := Field.mul_inv_cancel hb
  grind

theorem mul_nonneg_aux {a b : F} (ha : 0 ≤ a) (hb : 0 < b) : 0 ≤ a * b := by
  by_cases h : a = 0
  · subst h
    grind
  · have hpos : 0 < a := by grind
    first
    | grind
    | (have h2 := Lean.Grind.OrderedRing.mul_lt_mul_of_pos_left hb hpos; grind)

theorem eq_one_of_sq {a : F} (h0 : 0 ≤ a) (h : a * a = 1) : a = 1 := by
  have h2 : 0 < a + 1 := by grind
  have h3 : a + 1 ≠ 0 := by grind
  have h1 : (a - 1) * (a + 1) = 0 := by grind
  have h4 := mul_right_cancel0 h3 h1
  grind

theorem uniqueness (f : List F → List F) (h : Laws f) (xs : List F) (hv : 0 < var xs) :
    f xs = lnRow xs := by
  unfold Laws at h
  obtain ⟨_, _, _, hprop, hscale⟩ := h
  obtain ⟨k, hk0, hfk⟩ := hprop xs
  have hs := hscale xs
  rw [hfk] at hs
  have hsq : (xs.map (fun x => k * (x - mean xs))).map (fun y => y * y)
      = (xs.map (fun x => (x - mean xs) * (x - mean xs))).map (fun z => k * k * z) := by
    rw [List.map_map, List.map_map]
    apply List.map_congr_left
    intro x _
    show k * (x - mean xs) * (k * (x - mean xs)) = k * k * ((x - mean xs) * (x - mean xs))
    grind
  have hvar : mean (xs.map (fun x => (x - mean xs) * (x - mean xs))) = var xs := rfl
  rw [hsq, mean_map_scale, hvar] at hs
  have hvne : var xs ≠ 0 := by grind
  have h1 : (k * k * var xs - 1) * var xs = 0 := by grind
  have h2 : k * k * var xs - 1 = 0 := mul_right_cancel0 hvne h1
  have ht0 : 0 < HasSqrt.sqrt (var xs) := SqrtLaws.sqrt_pos _ hv
  have hv0 : 0 ≤ var xs := by grind
  have htt : HasSqrt.sqrt (var xs) * HasSqrt.sqrt (var xs) = var xs :=
    SqrtLaws.sqrt_mul_self _ hv0
  have h3 : (k * HasSqrt.sqrt (var xs)) * (k * HasSqrt.sqrt (var xs)) = 1 := by grind
  have h4 : 0 ≤ k * HasSqrt.sqrt (var xs) := mul_nonneg_aux hk0 ht0
  have hk : k * HasSqrt.sqrt (var xs) = 1 := eq_one_of_sq h4 h3
  have htne : HasSqrt.sqrt (var xs) ≠ 0 := by grind
  have hinv : HasSqrt.sqrt (var xs) * (HasSqrt.sqrt (var xs))⁻¹ = 1 := Field.mul_inv_cancel htne
  rw [hfk]
  unfold lnRow
  apply List.map_congr_left
  intro x _
  show k * (x - mean xs) = (x - mean xs) / HasSqrt.sqrt (var xs)
  grind
