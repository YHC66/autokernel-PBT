theorem count_nil {F : Type} [Field F] : count ([] : List F) = 0 := by
  simp only [count, List.map_nil, List.sum_nil]

theorem count_cons {F : Type} [Field F] (x : F) (l : List F) :
    count (x :: l) = 1 + count l := by
  simp only [count, List.map_cons, List.sum_cons]

theorem count_map {F : Type} [Field F] (g : F → F) (l : List F) :
    count (l.map g) = count l := by
  induction l with
  | nil => simp only [List.map_nil]
  | cons h t ih => rw [List.map_cons, count_cons, count_cons, ih]

theorem sum_map_affine {F : Type} [Field F] (a b : F) (l : List F) :
    (l.map (fun x => a * x + b)).sum = a * l.sum + count l * b := by
  induction l with
  | nil => simp only [List.map_nil, List.sum_nil, count_nil]; grind
  | cons h t ih => simp only [List.map_cons, List.sum_cons, count_cons]; grind

theorem sum_map_mul {F : Type} [Field F] (k : F) (g : F → F) (l : List F) :
    (l.map (fun x => k * g x)).sum = k * (l.map g).sum := by
  induction l with
  | nil => simp only [List.map_nil, List.sum_nil]; grind
  | cons h t ih => simp only [List.map_cons, List.sum_cons]; grind

theorem mean_map_mul {F : Type} [Field F] (l : List F) (k : F) (g : F → F) :
    mean (l.map (fun x => k * g x)) = k * mean (l.map g) := by
  unfold mean
  rw [sum_map_mul, count_map, count_map]
  grind

theorem mean_zero_count {F : Type} [Field F] (l : List F) (h : count l = 0) :
    mean l = 0 := by
  unfold mean
  rw [h]
  first
    | grind
    | (rw [Field.div_eq_mul_inv, Field.inv_zero]; grind)

theorem mean_map_affine {F : Type} [Field F] (l : List F) (a b : F) (hn : count l ≠ 0) :
    mean (l.map (fun x => a * x + b)) = a * mean l + b := by
  unfold mean
  rw [sum_map_affine, count_map]
  have h1 : count l * (count l)⁻¹ = 1 := by
    first
      | exact Field.mul_inv_cancel hn
      | grind
  grind

theorem var_map_affine {F : Type} [Field F] (l : List F) (a b : F) (hn : count l ≠ 0) :
    var (l.map (fun x => a * x + b)) = a * a * var l := by
  have hv : var (l.map (fun x => a * x + b)) =
      mean ((l.map (fun x => a * x + b)).map (fun y =>
        (y - mean (l.map (fun x => a * x + b))) * (y - mean (l.map (fun x => a * x + b))))) := rfl
  have h1 : (l.map (fun x => a * x + b)).map (fun y =>
        (y - mean (l.map (fun x => a * x + b))) * (y - mean (l.map (fun x => a * x + b))))
      = l.map (fun x => a * a * ((x - mean l) * (x - mean l))) := by
    rw [mean_map_affine l a b hn, List.map_map]
    apply List.map_congr_left
    intro x _
    show (a * x + b - (a * mean l + b)) * (a * x + b - (a * mean l + b))
      = a * a * ((x - mean l) * (x - mean l))
    grind
  rw [hv, h1]
  exact mean_map_mul l (a * a) (fun x => (x - mean l) * (x - mean l))

theorem alg_final {F : Type} [Field F] (a b x S n s : F) (hn : n ≠ 0) (hs : s ≠ 0)
    (has : a * s = 1) (hsum : a * S + n * b = 0) :
    a * x + b = (x - S / n) / s := by
  have h1 : n * n⁻¹ = 1 := by
    first
      | exact Field.mul_inv_cancel hn
      | grind
  have h2 : s * s⁻¹ = 1 := by
    first
      | exact Field.mul_inv_cancel hs
      | grind
  have ha : a = s⁻¹ := by grind
  have hb : b = -(a * S * n⁻¹) := by grind
  grind

variable {F : Type} [Field F] [LE F] [LT F] [LawfulOrderLT F] [IsLinearOrder F] [OrderedRing F] [HasSqrt F] [SqrtLaws F]

theorem unit_of_sq (t : F) (hpos : 0 < t) (h : t * t = 1) : t = 1 := by
  refine Classical.byContradiction (fun hne => ?_)
  have hcases : t < 1 ∨ 1 < t := by
    first
      | exact Std.lt_or_gt_of_ne hne
      | grind
  rcases hcases with hlt | hgt
  · have h1 := OrderedRing.mul_lt_mul_of_pos_left hlt hpos
    grind
  · have h1 := OrderedRing.mul_lt_mul_of_pos_right hgt hpos
    grind

def Laws (f : List F → List F) : Prop :=
  ∀ xs : List F, 0 < var xs →
    (f xs).sum = 0 ∧
    var (f xs) = 1 ∧
    (∃ a b : F, 0 < a ∧ f xs = xs.map (fun x => a * x + b)) ∧
    (∀ c : F, f (xs.map (fun x => x + c)) = f xs)

theorem uniqueness (f : List F → List F) (h : Laws f) (xs : List F) (hv : 0 < var xs) :
    f xs = lnRow xs := by
  unfold Laws at h
  obtain ⟨hsum, hvar, ⟨a, b, ha, hfx⟩, _⟩ := h xs hv
  have hn : count xs ≠ 0 := by
    intro h0
    have h2 : var xs = 0 :=
      mean_zero_count (xs.map (fun x => (x - mean xs) * (x - mean xs)))
        ((count_map _ _).trans h0)
    rw [h2] at hv
    first
      | grind
      | exact absurd hv (Std.lt_irrefl _)
  rw [hfx] at hsum hvar
  rw [sum_map_affine] at hsum
  rw [var_map_affine xs a b hn] at hvar
  have hs_pos : 0 < HasSqrt.sqrt (var xs) := SqrtLaws.sqrt_pos _ hv
  have hss : HasSqrt.sqrt (var xs) * HasSqrt.sqrt (var xs) = var xs :=
    SqrtLaws.sqrt_mul_self _ (by grind)
  have hmul : (a * HasSqrt.sqrt (var xs)) * (a * HasSqrt.sqrt (var xs)) = 1 := by grind
  have hpos : 0 < a * HasSqrt.sqrt (var xs) := by
    first
      | grind
      | (have := OrderedRing.mul_lt_mul_of_pos_left hs_pos ha; grind)
  have ht1 : a * HasSqrt.sqrt (var xs) = 1 := unit_of_sq _ hpos hmul
  have hs0 : HasSqrt.sqrt (var xs) ≠ 0 := by grind
  rw [hfx]
  unfold lnRow
  apply List.map_congr_left
  intro x _
  exact alg_final a b x xs.sum (count xs) (HasSqrt.sqrt (var xs)) hn hs0 ht1 hsum
