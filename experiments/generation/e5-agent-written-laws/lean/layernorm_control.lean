/- Positive control: zero mean, unit variance, and the affine-tie law with a positive slope. -/
set_option linter.unusedSectionVars false

variable {F : Type} [Field F] [LE F] [LT F] [LawfulOrderLT F] [IsLinearOrder F] [OrderedRing F]
  [HasSqrt F] [SqrtLaws F]

def Laws (f : List F → List F) : Prop :=
  ∀ xs : List F, 0 < var xs →
    mean (f xs) = 0 ∧ var (f xs) = 1 ∧
      ∃ a b : F, 0 < a ∧ f xs = xs.map (fun x => a * x + b)

theorem count_map {G : F → F} (xs : List F) : count (xs.map G) = count xs := by
  unfold count; simp [Function.comp_def]

theorem sum_affine (a b : F) (xs : List F) :
    (xs.map (fun x => a * x + b)).sum = a * xs.sum + b * count xs := by
  unfold count
  induction xs with
  | nil => simp; grind
  | cons y l ih => simp [ih]; grind

theorem count_pos (xs : List F) (h : xs ≠ []) : 0 < count xs := by
  have h1 : (0 : F) < 1 := OrderedRing.zero_lt_one (R := F)
  unfold count
  induction xs with
  | nil => exact absurd rfl h
  | cons y l ih =>
    cases l with
    | nil => simp; grind
    | cons z m =>
      have := ih (by simp)
      simp at this ⊢
      grind

theorem mean_affine (a b : F) (xs : List F) (hn : count xs ≠ 0) :
    mean (xs.map (fun x => a * x + b)) = a * mean xs + b := by
  unfold mean
  rw [sum_affine, count_map]
  grind

theorem sum_sq_affine (a b m : F) (xs : List F) :
    ((xs.map (fun x => a * x + b)).map (fun y => (y - (a * m + b)) * (y - (a * m + b)))).sum
      = a * a * (xs.map (fun x => (x - m) * (x - m))).sum := by
  induction xs with
  | nil => simp; grind
  | cons y l ih => simp at ih ⊢; rw [ih]; grind

theorem var_affine (a b : F) (xs : List F) (hn : count xs ≠ 0) :
    var (xs.map (fun x => a * x + b)) = a * a * var xs := by
  unfold var
  rw [mean_affine a b xs hn]
  have hm := sum_sq_affine a b (mean xs) xs
  unfold mean at hm ⊢
  simp only [List.map_map, Function.comp_def, count_map] at hm ⊢
  rw [hm]
  grind

theorem uniqueness (f : List F → List F) (h : Laws f) (xs : List F) (hv : 0 < var xs) :
    f xs = lnRow xs := by
  obtain ⟨hm0, hv1, a, b, ha, hf⟩ := h xs hv
  have hne : xs ≠ [] := by
    rintro rfl
    simp [var, mean, count] at hv
    grind
  have hn : count xs ≠ 0 := Preorder.ne_of_gt (count_pos xs hne)
  rw [hf] at hm0 hv1 ⊢
  rw [mean_affine a b xs hn] at hm0
  rw [var_affine a b xs hn] at hv1
  have hs : 0 < HasSqrt.sqrt (var xs) := SqrtLaws.sqrt_pos _ hv
  have hss : HasSqrt.sqrt (var xs) * HasSqrt.sqrt (var xs) = var xs :=
    SqrtLaws.sqrt_mul_self _ (Preorder.le_of_lt hv)
  have has : 0 < a * HasSqrt.sqrt (var xs) := OrderedRing.mul_pos ha hs
  have htt : (a * HasSqrt.sqrt (var xs)) * (a * HasSqrt.sqrt (var xs)) = 1 := by grind
  have hp : 0 < a * HasSqrt.sqrt (var xs) + 1 := by grind
  have hinv := Field.mul_inv_cancel (Preorder.ne_of_gt hp)
  have h1 : a * HasSqrt.sqrt (var xs) = 1 := by grind
  unfold lnRow
  congr 1
  funext x
  grind
