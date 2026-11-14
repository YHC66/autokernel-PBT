section LawsSection

variable {F : Type} [Field F] [LE F] [LT F] [LawfulOrderLT F] [IsLinearOrder F] [OrderedRing F] [HasSqrt F] [SqrtLaws F]

def Laws (f : List F → List F) : Prop :=
  (∀ xs : List F, (f xs).length = xs.length) ∧
  (∀ xs : List F, mean (f xs) = 0) ∧
  (∀ xs : List F, ∃ k : F, 0 ≤ k ∧ f xs = xs.map (fun x => k * (x - mean xs))) ∧
  (∀ xs : List F, 0 < var xs → mean ((f xs).map (fun y => y * y)) = 1) ∧
  (∀ (xs : List F) (c : F), f (xs.map (fun x => x + c)) = f xs)

theorem sum_map_mul_left (c : F) (g : F → F) (l : List F) :
    (l.map (fun x => c * g x)).sum = c * (l.map g).sum := by
  induction l with
  | nil => simp <;> grind
  | cons a t ih => simp [List.sum_cons, ih] <;> grind

theorem count_map (g : F → F) (l : List F) : count (l.map g) = count l := by
  first
  | simp [count, List.map_map, Function.comp_def]
  | simp [count, List.map_map]

theorem mean_map_mul (c : F) (g : F → F) (l : List F) :
    mean (l.map (fun x => c * g x)) = c * mean (l.map g) := by
  unfold mean
  rw [count_map (fun x => c * g x) l, count_map g l, sum_map_mul_left c g l]
  first
  | grind
  | simp [Field.div_eq_mul_inv, Lean.Grind.Semiring.mul_assoc]

theorem sq_map (k m : F) (xs : List F) :
    (xs.map (fun x => k * (x - m))).map (fun y => y * y)
      = xs.map (fun x => (k * k) * ((x - m) * (x - m))) := by
  rw [List.map_map]
  apply List.map_congr_left
  intro a _
  show (k * (a - m)) * (k * (a - m)) = (k * k) * ((a - m) * (a - m))
  grind

theorem mul_nonneg' {a b : F} (ha : 0 ≤ a) (hb : 0 ≤ b) : 0 ≤ a * b := by
  first
  | exact Lean.Grind.OrderedRing.mul_nonneg ha hb
  | grind

theorem uniqueness (f : List F → List F) (h : Laws f) (xs : List F) (hv : 0 < var xs) :
    f xs = lnRow xs := by
  obtain ⟨_, _, hprop, hsq, _⟩ := h
  obtain ⟨k, hk0, hk⟩ := hprop xs
  have hs : 0 < HasSqrt.sqrt (var xs) := SqrtLaws.sqrt_pos _ hv
  have hss : HasSqrt.sqrt (var xs) * HasSqrt.sqrt (var xs) = var xs :=
    SqrtLaws.sqrt_mul_self _ (Std.le_of_lt hv)
  have h1 := hsq xs hv
  rw [hk, sq_map] at h1
  have h2 : (k * k) * var xs = 1 := by
    have h3 := mean_map_mul (k * k) (fun x => (x - mean xs) * (x - mean xs)) xs
    exact h3.symm.trans h1
  have hnn : 0 ≤ k * HasSqrt.sqrt (var xs) := mul_nonneg' hk0 (Std.le_of_lt hs)
  have hsq1 : (k * HasSqrt.sqrt (var xs)) * (k * HasSqrt.sqrt (var xs)) = 1 := by
    calc (k * HasSqrt.sqrt (var xs)) * (k * HasSqrt.sqrt (var xs))
        = (k * k) * (HasSqrt.sqrt (var xs) * HasSqrt.sqrt (var xs)) := by grind
      _ = (k * k) * var xs := by rw [hss]
      _ = 1 := h2
  have hks : k * HasSqrt.sqrt (var xs) = 1 := by
    have hne : k * HasSqrt.sqrt (var xs) + 1 ≠ 0 := by
      intro h0
      have hpos : 0 < k * HasSqrt.sqrt (var xs) + 1 := by grind
      grind
    have h5 : (k * HasSqrt.sqrt (var xs) - 1) * (k * HasSqrt.sqrt (var xs) + 1) = 0 := by grind
    first
    | grind
    | (have h7 := Field.mul_inv_cancel hne
       calc k * HasSqrt.sqrt (var xs) - 1
           = ((k * HasSqrt.sqrt (var xs) - 1) * (k * HasSqrt.sqrt (var xs) + 1))
               * (k * HasSqrt.sqrt (var xs) + 1)⁻¹ := by grind
         _ = 0 := by rw [h5]; grind
       grind)
  have hsne : HasSqrt.sqrt (var xs) ≠ 0 := by
    intro h0
    rw [h0] at hs
    grind
  have hkinv : k = (HasSqrt.sqrt (var xs))⁻¹ := by
    have h7 := Field.mul_inv_cancel hsne
    calc k = k * (HasSqrt.sqrt (var xs) * (HasSqrt.sqrt (var xs))⁻¹) := by grind
      _ = (k * HasSqrt.sqrt (var xs)) * (HasSqrt.sqrt (var xs))⁻¹ := by grind
      _ = (HasSqrt.sqrt (var xs))⁻¹ := by rw [hks]; grind
  rw [hk]
  unfold lnRow
  apply List.map_congr_left
  intro a _
  rw [hkinv, Field.div_eq_mul_inv]
  grind

end LawsSection
