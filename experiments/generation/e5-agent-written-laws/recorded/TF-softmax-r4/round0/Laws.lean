variable {F : Type} [Field F] [LE F] [LT F] [LawfulOrderLT F] [IsLinearOrder F] [OrderedRing F] [HasExp F] [ExpLaws F]

def Laws (f : List F → List F) : Prop :=
  (∀ xs : List F, xs ≠ [] → (f xs).length = xs.length) ∧
  (∀ xs : List F, xs ≠ [] → ∀ y ∈ f xs, 0 ≤ y) ∧
  (∀ xs : List F, xs ≠ [] → (f xs).sum = 1) ∧
  (∀ xs : List F, xs ≠ [] → ∀ p ∈ List.zip (f xs) xs, ∀ q ∈ List.zip (f xs) xs,
      p.1 * HasExp.exp q.2 = q.1 * HasExp.exp p.2) ∧
  (∀ (xs : List F) (c : F), xs ≠ [] → f (xs.map (fun x => x + c)) = f xs)

theorem aux_sum (a b : F) : ∀ (ys xs : List F), ys.length = xs.length →
    (∀ p ∈ List.zip ys xs, a * HasExp.exp p.2 = p.1 * b) →
    a * (xs.map HasExp.exp).sum = ys.sum * b
  | [], [], _, _ => by
    simp only [List.map_nil, List.sum_nil]
    grind
  | [], _ :: _, h, _ => by simp at h
  | _ :: _, [], h, _ => by simp at h
  | y :: ys, x :: xs, h, H => by
    have h1 : a * HasExp.exp x = y * b := H (y, x) (by simp)
    have h2 : a * (xs.map HasExp.exp).sum = ys.sum * b :=
      aux_sum a b ys xs (by simpa using h) (fun p hp => H p (by simp [hp]))
    simp only [List.map_cons, List.sum_cons]
    grind

theorem aux_ext (S : F) : ∀ (ys xs : List F), ys.length = xs.length →
    (∀ p ∈ List.zip ys xs, p.1 = HasExp.exp p.2 / S) →
    ys = xs.map (fun x => HasExp.exp x / S)
  | [], [], _, _ => by simp
  | [], _ :: _, h, _ => by simp at h
  | _ :: _, [], h, _ => by simp at h
  | y :: ys, x :: xs, h, H => by
    have h1 : y = HasExp.exp x / S := H (y, x) (by simp)
    have h2 : ys = xs.map (fun x => HasExp.exp x / S) :=
      aux_ext S ys xs (by simpa using h) (fun p hp => H p (by simp [hp]))
    simp only [List.map_cons]
    rw [← h1, ← h2]

theorem uniqueness (f : List F → List F) (h : Laws f) (xs : List F) (hne : xs ≠ []) :
    f xs = softmaxRow xs := by
  obtain ⟨hlen, _, hsum, hrat, _⟩ := h
  have hl := hlen xs hne
  have hs := hsum xs hne
  have hr := hrat xs hne
  have key : ∀ p ∈ List.zip (f xs) xs, p.1 = HasExp.exp p.2 / partition xs := by
    intro p hp
    have h0 := aux_sum p.1 (HasExp.exp p.2) (f xs) xs hl (fun q hq => hr p hp q hq)
    rw [hs] at h0
    have e : p.1 * partition xs = HasExp.exp p.2 := by
      unfold partition
      grind
    have hpos := ExpLaws.exp_pos p.2
    have hP : partition xs ≠ 0 := by
      intro hz
      rw [hz] at e
      grind
    grind
  exact aux_ext (partition xs) (f xs) xs hl key
