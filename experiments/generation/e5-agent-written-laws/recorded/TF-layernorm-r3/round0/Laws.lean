variable {F : Type} [Field F] [LE F] [LT F] [LawfulOrderLT F] [IsLinearOrder F] [OrderedRing F] [HasSqrt F] [SqrtLaws F]

theorem ln_sum_map_mul (a : F) (q : F → F) (l : List F) :
    (l.map (fun x => a * q x)).sum = a * (l.map q).sum := by
  induction l with
  | nil => first | (simp; done) | (simp; grind) | grind
  | cons x l ih =>
    first
      | (simp only [List.map_cons, List.sum_cons, ih]; grind)
      | (simp [ih]; grind)
      | (simp [ih]; done)

theorem ln_count_map (g : F → F) (l : List F) : count (l.map g) = count l := by
  first
    | (unfold count; rw [List.map_map]; done)
    | (unfold count; rw [List.map_map]; rfl)
    | (simp [count, List.map_map, Function.comp_def])

theorem ln_mean_scale (a : F) (q : F → F) (l : List F) :
    mean (l.map (fun x => a * q x)) = a * mean (l.map q) := by
  unfold mean
  rw [ln_sum_map_mul a q l, ln_count_map (fun x => a * q x) l, ln_count_map q l]
  grind

theorem ln_zip_aux (a m : F) (l : List F) :
    List.zipWith (fun y x => y * (x - m)) (l.map (fun x => a * (x - m))) l
      = l.map (fun x => a * ((x - m) * (x - m))) := by
  induction l with
  | nil => first | rfl | simp
  | cons x l ih =>
    have e : a * (x - m) * (x - m) = a * ((x - m) * (x - m)) := by grind
    first
      | (simp only [List.map_cons, List.zipWith_cons_cons, ih, e]; done)
      | (simp [ih, e]; done)
      | (simp [ih]; grind)

theorem ln_cov (a : F) (l : List F) :
    mean (List.zipWith (fun y x => y * (x - mean l)) (l.map (fun x => a * (x - mean l))) l)
      = a * var l := by
  rw [ln_zip_aux a (mean l) l]
  exact ln_mean_scale a (fun x => (x - mean l) * (x - mean l)) l

def Laws (f : List F → List F) : Prop :=
  (∀ xs : List F, (f xs).length = xs.length) ∧
  (∀ xs : List F, mean (f xs) = 0) ∧
  (∀ xs : List F, ∃ a : F, f xs = xs.map (fun x => a * (x - mean xs))) ∧
  (∀ xs : List F, 0 < var xs →
    mean (List.zipWith (fun y x => y * (x - mean xs)) (f xs) xs) = HasSqrt.sqrt (var xs))

theorem uniqueness (f : List F → List F) (h : Laws f) (xs : List F) (hv : 0 < var xs) :
    f xs = lnRow xs := by
  unfold Laws at h
  obtain ⟨_, _, hcol, hcov⟩ := h
  obtain ⟨a, ha⟩ := hcol xs
  have hc := hcov xs hv
  rw [ha, ln_cov a xs] at hc
  have hs : 0 < HasSqrt.sqrt (var xs) := SqrtLaws.sqrt_pos _ hv
  have hv0 : 0 ≤ var xs := by
    first | grind | exact Std.le_of_lt hv
  have hss : HasSqrt.sqrt (var xs) * HasSqrt.sqrt (var xs) = var xs :=
    SqrtLaws.sqrt_mul_self _ hv0
  have hs0 : HasSqrt.sqrt (var xs) ≠ 0 := by
    first
      | grind
      | (intro h0; rw [h0] at hs; exact absurd hs (Std.lt_irrefl _))
  have hinv : HasSqrt.sqrt (var xs) * (HasSqrt.sqrt (var xs))⁻¹ = 1 :=
    Lean.Grind.Field.mul_inv_cancel hs0
  have h1 : a * (HasSqrt.sqrt (var xs) * HasSqrt.sqrt (var xs)) = HasSqrt.sqrt (var xs) := by
    rw [hss]; exact hc
  have ha' : a = (HasSqrt.sqrt (var xs))⁻¹ := by grind
  rw [ha]
  unfold lnRow
  apply List.map_congr_left
  intro x _
  show a * (x - mean xs) = (x - mean xs) / HasSqrt.sqrt (var xs)
  rw [ha']
  first
    | grind
    | (rw [Lean.Grind.Field.div_eq_mul_inv]; grind)
