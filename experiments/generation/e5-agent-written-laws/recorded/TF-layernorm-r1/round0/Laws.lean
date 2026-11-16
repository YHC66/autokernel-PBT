variable {F : Type} [Field F] [LE F] [LT F] [LawfulOrderLT F] [IsLinearOrder F] [OrderedRing F] [HasSqrt F] [SqrtLaws F]

def Laws (f : List F → List F) : Prop :=
  ∀ xs : List F, 0 < var xs →
    (∃ s u : F, 0 < s ∧ f xs = xs.map (fun x => s * x + u)) ∧
    (f xs).sum = 0 ∧
    mean ((f xs).map (fun y => y * y)) = 1

theorem sum_affine (s u : F) (xs : List F) :
    (xs.map (fun x => s * x + u)).sum = s * xs.sum + count xs * u := by
  induction xs with
  | nil => simp only [count, List.map_nil, List.sum_nil] <;> grind
  | cons a l ih =>
    simp only [count, List.map_cons, List.sum_cons] at ih ⊢
    grind

theorem sum_const_mul (c : F) (g : F → F) (xs : List F) :
    (xs.map (fun x => c * g x)).sum = c * (xs.map g).sum := by
  induction xs with
  | nil => simp only [List.map_nil, List.sum_nil] <;> grind
  | cons a l ih =>
    simp only [List.map_cons, List.sum_cons] at ih ⊢
    grind

theorem count_map (g : F → F) (xs : List F) : count (xs.map g) = count xs := by
  induction xs with
  | nil => rfl
  | cons a l ih =>
    simp only [count, List.map_cons, List.sum_cons] at ih ⊢
    rw [ih]

theorem mean_mul_count (l : List F) (h : count l ≠ 0) : mean l * count l = l.sum := by
  have hinv : count l * (count l)⁻¹ = 1 := Field.mul_inv_cancel h
  unfold mean
  rw [Field.div_eq_mul_inv]
  grind

theorem mean_of_count_zero (l : List F) (h : count l = 0) : mean l = 0 := by
  unfold mean
  rw [h, Field.div_eq_mul_inv, Field.inv_zero]
  grind

theorem uniqueness (f : List F → List F) (h : Laws f) (xs : List F) (hv : 0 < var xs) :
    f xs = lnRow xs := by
  unfold Laws at h
  obtain ⟨⟨s, u, hs, hf⟩, hsum, hms⟩ := h xs hv
  have hne : count xs ≠ 0 := by
    intro h0
    have h1 : count (xs.map (fun x => (x - mean xs) * (x - mean xs))) = 0 := by
      rw [count_map]; exact h0
    have h2 : var xs = 0 :=
      mean_of_count_zero (xs.map (fun x => (x - mean xs) * (x - mean xs))) h1
    rw [h2] at hv
    grind
  have hinv : count xs * (count xs)⁻¹ = 1 := Field.mul_inv_cancel hne
  have hmn : mean xs * count xs = xs.sum := mean_mul_count xs hne
  have hQ : var xs * count xs = (xs.map (fun x => (x - mean xs) * (x - mean xs))).sum := by
    have h1 : count (xs.map (fun x => (x - mean xs) * (x - mean xs))) ≠ 0 := by
      rw [count_map]; exact hne
    have h2 := mean_mul_count _ h1
    rw [count_map] at h2
    exact h2
  rw [hf] at hsum hms
  have h1 := sum_affine s u xs
  rw [hsum] at h1
  have hu : u = -(s * mean xs) := by grind
  subst hu
  have hL : (xs.map (fun x => s * x + -(s * mean xs))).map (fun y => y * y)
      = xs.map (fun x => s * s * ((x - mean xs) * (x - mean xs))) := by
    rw [List.map_map]
    apply List.map_congr_left
    intro x _
    show (s * x + -(s * mean xs)) * (s * x + -(s * mean xs))
      = s * s * ((x - mean xs) * (x - mean xs))
    grind
  rw [hL] at hms
  have hc : count (xs.map (fun x => s * s * ((x - mean xs) * (x - mean xs)))) ≠ 0 := by
    rw [count_map]; exact hne
  have h3 := mean_mul_count _ hc
  have h4 : (xs.map (fun x => s * s * ((x - mean xs) * (x - mean xs)))).sum
      = s * s * (xs.map (fun x => (x - mean xs) * (x - mean xs))).sum :=
    sum_const_mul (s * s) (fun x => (x - mean xs) * (x - mean xs)) xs
  rw [count_map, hms] at h3
  have hsv : s * s * var xs = 1 := by grind
  have hv0 : 0 ≤ var xs := by grind
  obtain ⟨T, hT⟩ : ∃ T, T = HasSqrt.sqrt (var xs) := ⟨_, rfl⟩
  have ht0 : 0 < T := by rw [hT]; exact SqrtLaws.sqrt_pos _ hv
  have htt : T * T = var xs := by rw [hT]; exact SqrtLaws.sqrt_mul_self _ hv0
  have hp2 : (s * T) * (s * T) = 1 := by grind
  have hp0 : 0 < s * T := by
    first
      | exact OrderedRing.mul_pos hs ht0
      | exact Lean.Grind.OrderedRing.mul_pos hs ht0
      | grind
  have hp1 : s * T + 1 ≠ 0 := by grind
  have hinv1 : (s * T + 1) * (s * T + 1)⁻¹ = 1 := Field.mul_inv_cancel hp1
  have hst : s * T = 1 := by grind
  have hTne : T ≠ 0 := by grind
  have hTinv : T * T⁻¹ = 1 := Field.mul_inv_cancel hTne
  rw [hf]
  unfold lnRow
  rw [← hT]
  apply List.map_congr_left
  intro x _
  rw [Field.div_eq_mul_inv]
  grind
