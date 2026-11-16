variable {F : Type} [Field F] [LE F] [LT F] [LawfulOrderLT F] [IsLinearOrder F] [OrderedRing F] [HasExp F] [ExpLaws F]

def Laws (f : List F → List F) : Prop :=
  (∀ xs : List F, (f xs).length = xs.length) ∧
  (∀ xs : List F, xs ≠ [] → (f xs).sum = 1) ∧
  (∀ (xs : List F) (i j : Nat) (a b u v : F),
      (f xs)[i]? = some a → (f xs)[j]? = some b → xs[i]? = some u → xs[j]? = some v →
      a * HasExp.exp v = b * HasExp.exp u) ∧
  (∀ (xs : List F) (c : F), f (xs.map (· + c)) = f xs)

theorem sum_map_mul (k : F) (l : List F) :
    (l.map (fun v => k * HasExp.exp v)).sum = k * (l.map HasExp.exp).sum := by
  induction l with
  | nil =>
    first
      | (simp only [List.map_nil, List.sum_nil]; grind)
      | (show (0 : F) = k * 0; grind)
  | cons a l ih =>
    simp only [List.map_cons, List.sum_cons]
    rw [ih]
    grind

theorem uniqueness (f : List F → List F) (h : Laws f) (xs : List F) (hne : xs ≠ []) :
    f xs = softmaxRow xs := by
  unfold Laws at h
  obtain ⟨hlen, hsum, hrat, -⟩ := h
  obtain ⟨u, t, rfl⟩ := List.exists_cons_of_ne_nil hne
  have hl := hlen (u :: t)
  have hs := hsum (u :: t) hne
  have hr := hrat (u :: t)
  generalize f (u :: t) = y at hl hs hr ⊢
  have hexp : HasExp.exp u ≠ 0 := by
    have hp := ExpLaws.exp_pos u
    first
      | grind
      | exact Std.ne_of_gt hp
      | exact ne_of_gt hp
  cases y with
  | nil => first | (simp at hl; done) | (exact absurd hl (by simp))
  | cons a t' =>
    have hy : a :: t' = (u :: t).map (fun v => (a / HasExp.exp u) * HasExp.exp v) := by
      apply List.ext_getElem?
      intro n
      rw [List.getElem?_map]
      cases hx : (u :: t)[n]? with
      | none =>
        have h1 : (u :: t).length ≤ n := List.getElem?_eq_none_iff.mp hx
        have h2 : (a :: t')[n]? = none := by
          apply List.getElem?_eq_none_iff.mpr
          rw [hl]
          exact h1
        rw [h2]
        rfl
      | some v =>
        obtain ⟨hn, -⟩ := List.getElem?_eq_some_iff.mp hx
        have hn' : n < (a :: t').length := by
          rw [hl]
          exact hn
        have hb : (a :: t')[n]? = some ((a :: t')[n]'hn') := List.getElem?_eq_getElem hn'
        have hkey : a * HasExp.exp v = (a :: t')[n]'hn' * HasExp.exp u :=
          hr 0 n a _ u v (by simp) hb (by simp) hx
        rw [hb]
        show some _ = some _
        congr 1
        generalize (a :: t')[n]'hn' = b at hkey ⊢
        grind
    have hsum' : (a :: t').sum = (a / HasExp.exp u) * partition (u :: t) := by
      rw [hy]
      exact sum_map_mul _ _
    have hk : (a / HasExp.exp u) * partition (u :: t) = 1 := by
      rw [← hsum']
      exact hs
    have hP : partition (u :: t) ≠ 0 := by
      intro h0
      rw [h0] at hk
      grind
    unfold softmaxRow
    rw [hy]
    first
      | (apply List.map_congr_left
         intro v _
         show a / HasExp.exp u * HasExp.exp v = HasExp.exp v / partition (u :: t)
         generalize a / HasExp.exp u = K at hk ⊢
         generalize partition (u :: t) = P at hk hP ⊢
         grind)
      | (congr 1
         funext v
         generalize a / HasExp.exp u = K at hk ⊢
         generalize partition (u :: t) = P at hk hP ⊢
         grind)
