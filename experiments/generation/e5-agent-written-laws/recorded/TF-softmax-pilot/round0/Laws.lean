variable {F : Type} [Field F] [LE F] [LT F] [LawfulOrderLT F] [IsLinearOrder F] [OrderedRing F] [HasExp F] [ExpLaws F]

/-- Per row: same length, nonnegative, sums to 1, and `y_i * exp x_j = y_j * exp x_i`. -/
def Laws (f : List F → List F) : Prop :=
  ∀ xs : List F, xs ≠ [] →
    (f xs).length = xs.length ∧
    (∀ y ∈ f xs, 0 ≤ y) ∧
    (f xs).sum = 1 ∧
    (∀ p ∈ xs.zip (f xs), ∀ q ∈ xs.zip (f xs),
      p.2 * HasExp.exp q.1 = q.2 * HasExp.exp p.1)

theorem key_aux (c e : F) : ∀ (xs ys : List F), xs.length = ys.length →
    (∀ q ∈ xs.zip ys, c * HasExp.exp q.1 = q.2 * e) →
    c * (xs.map HasExp.exp).sum = e * ys.sum
  | [], [], _, _ => by
    simp only [List.map_nil, List.sum_nil]
    grind
  | [], _ :: _, h, _ => by simp at h
  | _ :: _, [], h, _ => by simp at h
  | a :: xs, b :: ys, h, hq => by
    have h1 : c * HasExp.exp a = b * e := hq (a, b) (by simp)
    have hl : xs.length = ys.length := by
      simp only [List.length_cons] at h
      omega
    have ih := key_aux c e xs ys hl (fun q hq' => hq q (by
      rw [List.zip_cons_cons]
      exact List.mem_cons_of_mem _ hq'))
    simp only [List.map_cons, List.sum_cons]
    grind

theorem eq_map (g : F → F) : ∀ (xs ys : List F), xs.length = ys.length →
    (∀ p ∈ xs.zip ys, p.2 = g p.1) → ys = xs.map g
  | [], [], _, _ => by simp
  | [], _ :: _, h, _ => by simp at h
  | _ :: _, [], h, _ => by simp at h
  | a :: xs, b :: ys, h, hq => by
    have h1 : b = g a := hq (a, b) (by simp)
    have hl : xs.length = ys.length := by
      simp only [List.length_cons] at h
      omega
    have ih := eq_map g xs ys hl (fun q hq' => hq q (by
      rw [List.zip_cons_cons]
      exact List.mem_cons_of_mem _ hq'))
    rw [List.map_cons, ← h1, ← ih]

theorem exists_pair (xs ys : List F) (hne : xs ≠ []) (h : xs.length = ys.length) :
    ∃ p, p ∈ xs.zip ys := by
  cases xs with
  | nil => exact (hne rfl).elim
  | cons a t =>
    cases ys with
    | nil => simp at h
    | cons b u => exact ⟨(a, b), by simp⟩

theorem uniqueness (f : List F → List F) (h : Laws f) (xs : List F) (hne : xs ≠ []) :
    f xs = softmaxRow xs := by
  obtain ⟨hlen, _, hsum, hrat⟩ := h xs hne
  have hZ : ∀ p ∈ xs.zip (f xs), p.2 * partition xs = HasExp.exp p.1 := by
    intro p hp
    have h3 : p.2 * partition xs = HasExp.exp p.1 * (f xs).sum :=
      key_aux p.2 (HasExp.exp p.1) xs (f xs) hlen.symm (fun q hq => hrat p hp q hq)
    rw [hsum] at h3
    grind
  obtain ⟨p, hp⟩ := exists_pair xs (f xs) hne hlen.symm
  have hZne : partition xs ≠ 0 := by
    intro h0
    have h1 := hZ p hp
    rw [h0] at h1
    have h2 := ExpLaws.exp_pos p.1
    grind
  unfold softmaxRow
  refine eq_map _ xs (f xs) hlen.symm ?_
  intro q hq
  have h1 := hZ q hq
  show q.2 = HasExp.exp q.1 / partition xs
  first
    | grind
    | (have hinv : partition xs * (partition xs)⁻¹ = 1 := Field.mul_inv_cancel hZne
       rw [Field.div_eq_mul_inv]
       grind)
