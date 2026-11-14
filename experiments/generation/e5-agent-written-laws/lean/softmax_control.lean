/- Positive control: the completed bundle's laws (row sums and the log-ratio law), and a proof. -/

variable {F : Type} [Field F] [LE F] [LT F] [LawfulOrderLT F] [IsLinearOrder F] [OrderedRing F]
  [HasExp F] [ExpLaws F]

def Laws (f : List F → List F) : Prop :=
  ∀ xs : List F, xs ≠ [] →
    (f xs).sum = 1 ∧ ∃ c : F, f xs = xs.map (fun x => c * exp x)

theorem sum_map_mul_left' (c : F) (g : F → F) (xs : List F) :
    (xs.map (fun x => c * g x)).sum = c * (xs.map g).sum := by
  induction xs with
  | nil => simp; grind
  | cons a l ih => simp [ih]; grind

theorem uniqueness (f : List F → List F) (h : Laws f) (xs : List F) (hne : xs ≠ []) :
    f xs = softmaxRow xs := by
  obtain ⟨hsum, c, hc⟩ := h xs hne
  rw [hc] at hsum ⊢
  rw [sum_map_mul_left'] at hsum
  have hc' : c = (partition xs)⁻¹ := Field.eq_inv_of_mul_eq_one hsum
  unfold softmaxRow
  rw [hc']
  congr 1
  funext x
  grind
