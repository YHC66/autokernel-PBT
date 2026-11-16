variable {F : Type} [Field F] [LE F] [LT F] [LawfulOrderLT F] [IsLinearOrder F] [OrderedRing F] [HasExp F] [ExpLaws F]

def Laws (f : List F → List F) : Prop :=
  (∀ xs : List F, (f xs).length = xs.length) ∧
  (∀ xs : List F, ∀ y ∈ f xs, 0 ≤ y) ∧
  (∀ xs : List F, xs ≠ [] → (f xs).sum = 1) ∧
  (∀ xs : List F, ∀ p : F × F, p ∈ List.zip xs (f xs) → ∀ q : F × F, q ∈ List.zip xs (f xs) →
      p.2 * HasExp.exp q.1 = q.2 * HasExp.exp p.1) ∧
  (∀ (xs : List F) (c : F), f (xs.map (· + c)) = f xs)

theorem sum_exp_nonneg : ∀ xs : List F, 0 ≤ (xs.map HasExp.exp).sum
  | [] => by
    simp only [List.map_nil, List.sum_nil]
    grind
  | x :: t => by
    simp only [List.map_cons, List.sum_cons]
    have h1 : 0 < HasExp.exp x := ExpLaws.exp_pos x
    have h2 := sum_exp_nonneg t
    grind

theorem sum_exp_pos (x : F) (t : List F) : 0 < ((x :: t).map HasExp.exp).sum := by
  simp only [List.map_cons, List.sum_cons]
  have h1 : 0 < HasExp.exp x := ExpLaws.exp_pos x
  have h2 := sum_exp_nonneg t
  grind

theorem aux_sum (a b : F) : ∀ (xs ys : List F), xs.length = ys.length →
    (∀ q ∈ xs.zip ys, b * HasExp.exp q.1 = q.2 * HasExp.exp a) →
    b * (xs.map HasExp.exp).sum = ys.sum * HasExp.exp a
  | [], [], _, _ => by
    simp only [List.map_nil, List.sum_nil]
    grind
  | [], _ :: _, h, _ => by simp at h
  | _ :: _, [], h, _ => by simp at h
  | x :: xs, y :: ys, h, hp => by
    have h1 : b * HasExp.exp x = y * HasExp.exp a := hp (x, y) (by simp)
    have h2 := aux_sum a b xs ys (by simpa using h) (fun q hq => hp q (by simp [hq]))
    simp only [List.map_cons, List.sum_cons]
    grind

theorem fin_aux (Z : F) (hZ : Z ≠ 0) : ∀ (xs ys : List F), xs.length = ys.length →
    (∀ p ∈ xs.zip ys, p.2 * Z = HasExp.exp p.1) →
    ys = xs.map (fun x => HasExp.exp x / Z)
  | [], [], _, _ => rfl
  | [], _ :: _, h, _ => by simp at h
  | _ :: _, [], h, _ => by simp at h
  | x :: xs, y :: ys, h, hp => by
    have h1 : y * Z = HasExp.exp x := hp (x, y) (by simp)
    have h2 := fin_aux Z hZ xs ys (by simpa using h) (fun p hq => hp p (by simp [hq]))
    have h3 : y = HasExp.exp x / Z := by grind
    simp only [List.map_cons]
    rw [h2, h3]

theorem uniqueness (f : List F → List F) (h : Laws f) (xs : List F) (hne : xs ≠ []) :
    f xs = softmaxRow xs := by
  unfold Laws at h
  obtain ⟨hlen, _, hsum, hrat, _⟩ := h
  have hl : (f xs).length = xs.length := hlen xs
  have hs : (f xs).sum = 1 := hsum xs hne
  have hZ : partition xs ≠ 0 := by
    cases xs with
    | nil => exact absurd rfl hne
    | cons x t =>
      have h1 := sum_exp_pos x t
      unfold partition
      grind
  have hp : ∀ p : F × F, p ∈ List.zip xs (f xs) → p.2 * partition xs = HasExp.exp p.1 := by
    intro p hp
    have h1 := aux_sum p.1 p.2 xs (f xs) hl.symm (fun q hq => hrat xs p hp q hq)
    unfold partition
    grind
  exact fin_aux (partition xs) hZ xs (f xs) hl.symm hp
