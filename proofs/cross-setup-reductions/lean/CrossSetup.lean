/-
Cross-setup consistency of the ladder's reductions, in core Lean 4 — no Mathlib.

The statement is fixed by `proofs/cross-setup-reductions/README.md`. Over any commutative monoid,
the Triton kernel's single-tile tree and the CUDA port's strided-fold-then-block-tree compute
`spec xs = foldl (⊕) e xs`; hence the Triton and CUDA softmax and layernorm models compute the
spec's row, whatever `exp`, `sqrt`, `/`, `-`, `*`, the cast and `eps` are. Those operations are
fields of the structure argument `Ops` and no law about any of them is assumed. The monoid laws
are fields of the structure `CMonoid`; `max` is a field of `MaxOp` together with its
commutativity and associativity, the only laws used. The file adds nothing to Lean's logic;
the `#print axioms` lines at the end audit each theorem.

## Model ↔ kernel text

Triton, `kernels/triton/ladder.py`:
  `lane`, `tile`       `offs = tl.arange(0, BLOCK)`, `mask = offs < n_cols`,
                       `tl.load(..., mask=mask, other=e)`: lane `i` is `xs[i]` if `i < n`,
                       else the identity `e`                                  lines 86-89, 99-101
  `tree`, `tileReduce` `tl.max(x, axis=0)` / `tl.sum(.., axis=0)` over the tile, as the README's
                       balanced halving tree                          lines 90, 93, 102, 106
  `n ≤ 2^kB`           `block_for`: BLOCK is n_cols rounded up to a power of two   lines 47-71
  `softmaxTriton`      `_softmax_kernel` (84-93): the max tile loads `other=-inf`, the bottom
                       `none` of `withBot` (89-90); the sum tile's masked lanes hold `0` directly,
                       the README's reading of `exp(-inf - M) = 0` (91-93); output `e / sum` (93)
  `layernormTriton`    `_layernorm_kernel` (97-107): `x` tile `other=0.0` (101), `mean` (102),
                       `var` over `centered * centered` with `centered = tl.where(mask, x - mean,
                       0.0)` (105-106), output `centered / tl.sqrt(var + EPS)` (107). The variance
                       tile is the squared row with masked lanes holding `0` directly;
                       `triton_var_tile` proves that is the kernel's literal tile, whose masked
                       lanes hold `0.0 * 0.0`, given `0 * 0 = 0`.

CUDA, `kernels/cuda/softmax.cu` (S) and `kernels/cuda/layernorm.cu` (L):
  `loopCols`, `strided` `for (int c = tid; c < n_cols; c += width)`  S:34, 51, 64; L:43, 49, 56
  `partial0`           init with the identity, fold the thread's columns in loop order with
                       `acc = acc ⊕ x[c]`, store `partial[tid]`
                       S:33-37 (`-inf`, `fmaxf`), S:50-54 (`0`, `+=`); L:41-45, 48-52, 18
  `step`               `if (tid < s) partial[tid] = partial[tid] ⊕ partial[tid + s]`
                                                                       S:40-42, 57-59; L:21-23
  `treeLoop`           `for (int s = width >> 1; s > 0; s >>= 1)`, literally   S:39, 56; L:20
  `levels`             that loop as the composition `step 1 ∘ … ∘ step (W/2)` of the README
                       (`treeLoop_eq_levels`)
  `blockReduce`        the tree over `W = 2^k` partials, read at `partial[0]`    S:45, 62; L:26
  `W = 2^k`            launch contract: `block = (width, 1, 1)`, width a power of two
                       (S:14, L:11; `threads_for` in kernels/cuda/ladder.py)
  `softmaxCuda`        `softmax_rows`: row max S:31-45, denominator S:49-62, output S:64-66
  `layernormCuda`      `layernorm_rows`: mean L:41-46, var L:48-53, `sqrtf(var + eps)` L:54,
                       output L:56-58

`lnMean`, `lnVar` and `layernormWith` / `softmaxWith` hold the arithmetic the two setups share,
parameterized by the reduction; the spec instantiates it with `spec`, the Triton model with
`tileReduce`, the CUDA model with `blockReduce`.

## Why the functional `step` is the in-place loop

At level `s`, thread `t < s` runs `partial[t] = partial[t] ⊕ partial[t + s]` and threads `t ≥ s`
do nothing. Every write lands at an index `< s`, and index `t` is written by thread `t` alone.
Every read is at `t`, by thread `t` itself before its own write, or at `t + s ≥ s`, an index no
thread writes at this level. So no read observes a write of the same level, under any
interleaving, and the level maps the array `p` to `step s p`: `p t ⊕ p (t + s)` below `s` and
`p t` elsewhere. The `__syncthreads()` closing each level (S:43, S:60, L:24) makes the next level
read exactly that array. The model is that barrier-enforced sequential schedule; concurrency
itself is out of scope, as the README states.

## Lean-specific modelling choices

* Commutative monoid: the structure `CMonoid` (fields `op`, `e`, `op_assoc`, `op_comm`, `e_op`),
  passed explicitly, so the `+` monoid and the `max` monoid can coexist over one carrier.
* Bottom for `max`: `Option F`, with `none` as `⊥ = -inf`; `withBot L` is the monoid.
* `xs[c]`: `lane` reads `xs[c]` when `c < n`, else `e`. The CUDA loops only produce `c < n`, so
  there the default is never read; in the tile it is the masked lane.
* The C loop has no termination measure when `width = 0`, so `loopCols` carries fuel and
  `strided` gives it `n`, an upper bound on the trip count for every `W ≥ 1`;
  `strided_unfold` proves the fuel never runs out.
* The in-place array is a function `Nat → M`; each tree level is `step`.
* Softmax on the empty row: the row max is `⊥`, and the model writes no column (`[]`), as both
  kernels store nothing when `n_cols = 0`. For `n ≥ 1` the max is finite (`spec_max_cons`) and the
  spec is the textbook softmax (`softmaxSpec_cons`).
-/

namespace CrossSetup

universe u

/-! ## Commutative monoids and the spec -/

/-- A commutative monoid `(M, op, e)`. The laws are fields: hypotheses carried by the
structure. -/
structure CMonoid (M : Type u) where
  op : M → M → M
  e : M
  op_assoc : ∀ a b c, op (op a b) c = op a (op b c)
  op_comm : ∀ a b, op a b = op b a
  e_op : ∀ a, op e a = a

section Monoid

variable {M : Type u} (m : CMonoid M)

theorem CMonoid.op_e (a : M) : m.op a m.e = a := by rw [m.op_comm, m.e_op]

/-- `spec xs = foldl (⊕) e xs`. -/
def spec (xs : List M) : M := xs.foldl m.op m.e

theorem foldl_init (x : M) (l : List M) : l.foldl m.op x = m.op x (l.foldl m.op m.e) := by
  induction l generalizing x with
  | nil => exact (m.op_e x).symm
  | cons a l ih =>
    simp only [List.foldl_cons]
    rw [ih (m.op x a), ih (m.op m.e a), m.e_op, m.op_assoc]

theorem spec_nil : spec m [] = m.e := rfl

theorem spec_cons (x : M) (l : List M) : spec m (x :: l) = m.op x (spec m l) := by
  unfold spec
  rw [List.foldl_cons, foldl_init, m.e_op]

theorem spec_append (a b : List M) : spec m (a ++ b) = m.op (spec m a) (spec m b) := by
  unfold spec
  rw [List.foldl_append, foldl_init]

theorem spec_flatten (L : List (List M)) : spec m L.flatten = spec m (L.map (spec m)) := by
  induction L with
  | nil => rfl
  | cons l L ih => rw [List.flatten_cons, spec_append, ih, List.map_cons, spec_cons]

/-- Permutation invariance: the fold of a commutative monoid does not see the order. -/
theorem spec_perm {l l' : List M} (h : l.Perm l') : spec m l = spec m l' :=
  h.foldl_eq' (fun x _ y _ z => by rw [m.op_assoc, m.op_comm x y, ← m.op_assoc]) m.e

theorem spec_replicate_e (n : Nat) : spec m (List.replicate n m.e) = m.e := by
  induction n with
  | zero => rfl
  | succ n ih => rw [List.replicate_succ, spec_cons, ih, m.e_op]

theorem spec_map_op {α : Type} (f g : α → M) (l : List α) :
    spec m (l.map (fun t => m.op (f t) (g t))) = m.op (spec m (l.map f)) (spec m (l.map g)) := by
  induction l with
  | nil => exact (m.e_op m.e).symm
  | cons a l ih =>
    simp only [List.map_cons, spec_cons, ih]
    rw [m.op_assoc, m.op_assoc, ← m.op_assoc (g a), m.op_comm (g a), m.op_assoc]

end Monoid

/-! ## The tile model (Triton) -/

section Tile

variable {M : Type u} (m : CMonoid M)

/-- Lane `i` of `tl.load(x_ptr + offs, mask=offs < n_cols, other=e)`. -/
def lane (xs : List M) (i : Nat) : M := if h : i < xs.length then xs[i] else m.e

/-- The tile of width `B`: `[ if i < n then xs[i] else e | i ← 0 .. B-1 ]`. -/
def tile (B : Nat) (xs : List M) : List M := (List.range B).map (lane m xs)

/-- The balanced tree over a list of length `2^k`: `tree [a] = a`, and
`tree (l ++ r) = tree l ⊕ tree r` with `|l| = |r| = 2^(k-1)`. -/
def tree : Nat → List M → M
  | 0, l => l.headD m.e
  | k + 1, l => m.op (tree k (l.take (2 ^ k))) (tree k (l.drop (2 ^ k)))

/-- `tl.max` / `tl.sum` over the tile of width `B = 2^k`. -/
def tileReduce (k : Nat) (xs : List M) : M := tree m k (tile m (2 ^ k) xs)

theorem tree_eq_spec (k : Nat) (l : List M) (h : l.length = 2 ^ k) : tree m k l = spec m l := by
  induction k generalizing l with
  | zero =>
    match l, h with
    | [a], _ => exact (m.e_op a).symm
  | succ k ih =>
    have hp : 2 ^ (k + 1) = 2 ^ k + 2 ^ k := by rw [Nat.pow_succ]; omega
    unfold tree
    rw [ih _ (by simp; omega), ih _ (by simp; omega), ← spec_append, List.take_append_drop]

/-- The tile is the row's first `B` columns, padded with the identity. -/
theorem tile_eq (B : Nat) (xs : List M) :
    tile m B xs = xs.take B ++ List.replicate (B - xs.length) m.e := by
  apply List.ext_getElem
  · simp [tile]; omega
  · intro i h1 _
    have hB : i < B := by simpa [tile] using h1
    simp only [tile, List.getElem_map, List.getElem_range, lane, List.getElem_append]
    by_cases hi : i < xs.length
    · rw [dif_pos hi, dif_pos (by simp; omega), List.getElem_take]
    · rw [dif_neg hi, dif_neg (by simp; omega), List.getElem_replicate]

theorem length_tile (B : Nat) (xs : List M) : (tile m B xs).length = B := by simp [tile]

/-- Reading every lane of the row in order returns the row. -/
theorem range_map_lane (xs : List M) : (List.range xs.length).map (lane m xs) = xs := by
  apply List.ext_getElem
  · simp
  · intro i _ h2
    simp [lane, h2]

/-- A map that sends the masked value to the masked value commutes with loading the tile. -/
theorem tile_map {N : Type u} (m' : CMonoid N) (f : M → N) (hf : f m.e = m'.e) (B : Nat)
    (xs : List M) : (tile m B xs).map f = tile m' B (xs.map f) := by
  unfold tile
  rw [List.map_map]
  apply List.map_congr_left
  intro i _
  show f (lane m xs i) = lane m' (xs.map f) i
  unfold lane
  by_cases h : i < xs.length
  · rw [dif_pos h, dif_pos (by simpa using h), List.getElem_map]
  · rw [dif_neg h, dif_neg (by simpa using h), hf]

/-- **`tile_reduce_drops_tail`**: the tile model reduces the first `2^k` columns, for every
row; on a row wider than the tile the tail is dropped. -/
theorem tile_reduce_drops_tail (k : Nat) (xs : List M) :
    tileReduce m k xs = spec m (xs.take (2 ^ k)) := by
  unfold tileReduce
  rw [tree_eq_spec m k _ (length_tile m _ xs), tile_eq, spec_append, spec_replicate_e, m.op_e]

/-- **`tile_reduce_eq_spec`**: when the tile holds the whole row, the tile model is the spec. -/
theorem tile_reduce_eq_spec (k : Nat) (xs : List M) (h : xs.length ≤ 2 ^ k) :
    tileReduce m k xs = spec m xs := by
  rw [tile_reduce_drops_tail, List.take_of_length_le h]

end Tile

/-! ## The strided loop -/

section Strided

/-- One thread's trip through `for (int c = t; c < n; c += W)`: the columns it visits, in
order. `fuel` makes the definition total (the C loop does not terminate when `W = 0`). -/
def loopCols (W n : Nat) : Nat → Nat → List Nat
  | 0, _ => []
  | fuel + 1, c => if c < n then c :: loopCols W n fuel (c + W) else []

/-- `strided W n t`: the columns thread `t` visits, `t, t + W, t + 2W, … < n`, ascending. -/
def strided (W n t : Nat) : List Nat := loopCols W n n t

theorem loopCols_fuel (W n : Nat) (hW : 1 ≤ W) :
    ∀ (f g c : Nat), n - c ≤ f → n - c ≤ g → loopCols W n f c = loopCols W n g c := by
  intro f
  induction f with
  | zero =>
    intro g c hf _
    cases g with
    | zero => rfl
    | succ g => simp [loopCols, show ¬ c < n by omega]
  | succ f ih =>
    intro g c hf hg
    cases g with
    | zero => simp [loopCols, show ¬ c < n by omega]
    | succ g =>
      simp only [loopCols]
      split
      · rw [ih g (c + W) (by omega) (by omega)]
      · rfl

/-- The fuel never runs out: `strided` satisfies the loop's own recursion equation. -/
theorem strided_unfold (W n t : Nat) (hW : 1 ≤ W) :
    strided W n t = if t < n then t :: strided W n (t + W) else [] := by
  unfold strided
  cases n with
  | zero => simp [loopCols]
  | succ n =>
    have e : loopCols W (n + 1) (n + 1) t
        = if t < n + 1 then t :: loopCols W (n + 1) n (t + W) else [] := rfl
    rw [e]
    by_cases h : t < n + 1
    · rw [if_pos h, if_pos h, loopCols_fuel W (n + 1) hW n (n + 1) (t + W) (by omega) (by omega)]
    · rw [if_neg h, if_neg h]

theorem loopCols_mem (W n : Nat) : ∀ (f c c' : Nat), c' ∈ loopCols W n f c →
    c ≤ c' ∧ c' < n ∧ c' % W = c % W := by
  intro f
  induction f with
  | zero => intro c c' h; simp [loopCols] at h
  | succ f ih =>
    intro c c' h
    simp only [loopCols] at h
    split at h
    · rcases List.mem_cons.1 h with rfl | h
      · exact ⟨Nat.le_refl _, by assumption, rfl⟩
      · obtain ⟨h1, h2, h3⟩ := ih _ _ h
        refine ⟨by omega, h2, ?_⟩
        rw [h3, Nat.add_mod_right]
    · simp at h

theorem loopCols_complete (W n : Nat) : ∀ (f c c' : Nat),
    c ≤ c' → c' < n → c' % W = c % W → c' - c < f * W → c' ∈ loopCols W n f c := by
  intro f
  induction f with
  | zero => intro c c' _ _ _ h; simp at h
  | succ f ih =>
    intro c c' h1 h2 h3 h4
    simp only [loopCols, if_pos (show c < n by omega)]
    by_cases hc : c' = c
    · subst hc; simp
    · apply List.mem_cons_of_mem
      have hdvd : W ∣ c' - c := Nat.dvd_of_mod_eq_zero (Nat.sub_mod_eq_zero_of_mod_eq h3)
      have hle : W ≤ c' - c := Nat.le_of_dvd (by omega) hdvd
      apply ih (c + W) c' (by omega) h2 (by rw [h3, Nat.add_mod_right])
      rw [Nat.succ_mul] at h4
      omega

theorem loopCols_sorted (W n : Nat) (hW : 1 ≤ W) :
    ∀ (f c : Nat), (loopCols W n f c).Pairwise (· < ·) := by
  intro f
  induction f with
  | zero => intro c; simp [loopCols]
  | succ f ih =>
    intro c
    simp only [loopCols]
    split
    · refine List.pairwise_cons.2 ⟨fun c' h => ?_, ih _⟩
      have := (loopCols_mem W n f (c + W) c' h).1
      omega
    · exact List.Pairwise.nil

/-- Thread `t` visits exactly the columns `c < n` with `c ≥ t` and `c ≡ t (mod W)`. -/
theorem mem_strided (W n t c : Nat) (hW : 1 ≤ W) :
    c ∈ strided W n t ↔ t ≤ c ∧ c < n ∧ c % W = t % W := by
  constructor
  · exact loopCols_mem W n n t c
  · intro ⟨h1, h2, h3⟩
    apply loopCols_complete W n n t c h1 h2 h3
    have : n * 1 ≤ n * W := Nat.mul_le_mul_left n hW
    omega

/-- **`strided_cover`**: for `W ≥ 1`, the concatenation over `t = 0 .. W-1` of `strided W n t`
is a permutation of `[0, n)` — every column is visited by exactly one thread, once. -/
theorem strided_cover (W n : Nat) (hW : 1 ≤ W) :
    ((List.range W).flatMap (strided W n)).Perm (List.range n) := by
  apply (List.perm_ext_iff_of_nodup ?_ List.nodup_range).2
  · intro c
    simp only [List.mem_flatMap, List.mem_range]
    constructor
    · rintro ⟨t, _, hc⟩
      exact ((mem_strided W n t c hW).1 hc).2.1
    · intro hc
      exact ⟨c % W, Nat.mod_lt c (by omega),
        (mem_strided W n _ c hW).2 ⟨Nat.mod_le c W, hc, (Nat.mod_mod c W).symm⟩⟩
  · apply List.pairwise_flatMap.2
    refine ⟨fun t _ => (loopCols_sorted W n hW n t).imp Nat.ne_of_lt, ?_⟩
    refine List.Pairwise.imp_of_mem (fun {a b} ha hb hab => ?_) List.pairwise_lt_range
    intro x hx y hy hxy
    rw [List.mem_range] at ha hb
    have h1 := ((mem_strided W n a x hW).1 hx).2.2
    have h2 := ((mem_strided W n b y hW).1 hy).2.2
    subst hxy
    rw [Nat.mod_eq_of_lt ha] at h1
    rw [Nat.mod_eq_of_lt hb] at h2
    omega

end Strided

/-! ## The block model (CUDA) -/

section Block

variable {M : Type u} (m : CMonoid M)

/-- Phase 1: thread `t`'s sequential fold over its columns, stored into `partial[t]`. -/
def partial0 (W : Nat) (xs : List M) (t : Nat) : M :=
  ((strided W xs.length t).map (lane m xs)).foldl m.op m.e

/-- One level of the in-place tree: `if (t < s) partial[t] = partial[t] ⊕ partial[t + s]`. -/
def step (s : Nat) (p : Nat → M) (t : Nat) : M := if t < s then m.op (p t) (p (t + s)) else p t

/-- `levels k p = (step 1 ∘ step 2 ∘ … ∘ step (2^(k-1))) p`; the level `s = 2^(k-1)` runs
first, and `k = 0` runs no level. -/
def levels : Nat → (Nat → M) → Nat → M
  | 0, p => p
  | j + 1, p => levels j (step m (2 ^ j) p)

/-- `blockReduce k xs`: the tree over the `W = 2^k` partials, read at `partial[0]`. -/
def blockReduce (k : Nat) (xs : List M) : M := levels m k (partial0 m (2 ^ k) xs) 0

/-- The kernels' tree loop, literally: `for (s = W >> 1; s > 0; s >>= 1) step s`. -/
def treeLoop (s : Nat) (p : Nat → M) : Nat → M :=
  if 0 < s then treeLoop (s >>> 1) (step m s p) else p
termination_by s
decreasing_by rw [Nat.shiftRight_eq_div_pow]; omega

/-- The literal loop started at `s = W >> 1` is the README's composition of levels. -/
theorem treeLoop_eq_levels (k : Nat) : ∀ p, treeLoop m (2 ^ k >>> 1) p = levels m k p := by
  induction k with
  | zero =>
    intro p
    have h : 2 ^ 0 >>> 1 = 0 := rfl
    rw [h, treeLoop]
    simp [levels]
  | succ k ih =>
    intro p
    have h : 2 ^ (k + 1) >>> 1 = 2 ^ k := by
      rw [Nat.shiftRight_eq_div_pow, Nat.pow_succ]; simp
    rw [h, treeLoop, if_pos (Nat.two_pow_pos k), ih, levels]

theorem blockReduce_eq_loop (k : Nat) (xs : List M) :
    blockReduce m k xs = treeLoop m (2 ^ k >>> 1) (partial0 m (2 ^ k) xs) 0 := by
  rw [treeLoop_eq_levels]
  rfl

/-- One level folds the `2s` live slots into `s`: the fold over them is unchanged. -/
theorem sum_step (s : Nat) (p : Nat → M) :
    spec m ((List.range s).map (step m s p)) = spec m ((List.range (s + s)).map p) := by
  have h : (List.range s).map (step m s p)
      = (List.range s).map (fun t => m.op (p t) (p (s + t))) := by
    apply List.map_congr_left
    intro t ht
    rw [List.mem_range] at ht
    rw [step, if_pos ht, Nat.add_comm t s]
  rw [h, spec_map_op, List.range_add, List.map_append, spec_append, List.map_map]
  rfl

theorem levels_spec (j : Nat) :
    ∀ p : Nat → M, levels m j p 0 = spec m ((List.range (2 ^ j)).map p) := by
  induction j with
  | zero =>
    intro p
    show p 0 = spec m [p 0]
    rw [spec_cons, spec_nil, m.op_e]
  | succ j ih =>
    intro p
    rw [levels, ih, sum_step, Nat.pow_succ, Nat.mul_two]

/-- **`block_reduce_eq_spec`**: the block model is the spec, for every `k` and every row; no
relation between `W = 2^k` and `n` is needed. -/
theorem block_reduce_eq_spec (k : Nat) (xs : List M) : blockReduce m k xs = spec m xs := by
  have hW : 1 ≤ 2 ^ k := Nat.one_le_two_pow
  unfold blockReduce
  rw [levels_spec]
  have h1 : (List.range (2 ^ k)).map (partial0 m (2 ^ k) xs)
      = ((List.range (2 ^ k)).map
          (fun t => (strided (2 ^ k) xs.length t).map (lane m xs))).map (spec m) := by
    rw [List.map_map]
    rfl
  rw [h1, ← spec_flatten, ← List.flatMap_def, ← List.map_flatMap,
    spec_perm m ((strided_cover (2 ^ k) xs.length hW).map (lane m xs)), range_map_lane]

end Block

/-! ## The two kernels over an abstract carrier -/

section Kernels

variable {F : Type u}

/-- A `max` with the two laws the proofs use: commutativity and associativity. -/
structure MaxOp (F : Type u) where
  max : F → F → F
  max_comm : ∀ a b, max a b = max b a
  max_assoc : ∀ a b c, max (max a b) c = max a (max b c)

/-- `max` on `F ∪ {⊥}`, with `⊥ = none` playing `-inf`. -/
def botMax (L : MaxOp F) : Option F → Option F → Option F
  | none, b => b
  | some a, none => some a
  | some a, some b => some (L.max a b)

/-- `(F ∪ {⊥}, max, ⊥)` is a commutative monoid. -/
def withBot (L : MaxOp F) : CMonoid (Option F) where
  op := botMax L
  e := none
  op_assoc := by
    intro a b c
    cases a <;> cases b <;> cases c <;> simp [botMax, L.max_assoc]
  op_comm := by
    intro a b
    cases a <;> cases b <;> simp [botMax, L.max_comm]
  e_op := by intro a; cases a <;> rfl

/-- The scalar operations the kernels share, as parameters. No law is assumed about any. -/
structure Ops (F : Type u) where
  sub : F → F → F
  mul : F → F → F
  div : F → F → F
  exp : F → F
  sqrt : F → F
  cast : Nat → F
  eps : F

/-- Softmax of a row, the row max computed by `rmax` and the denominator by `rsum`:
`y[i] = exp (xs[i] - M) / S` with `M = rmax xs`, `S = rsum [exp (x - M) | x ← xs]`.
A row whose max is `⊥` (only the empty row) writes no column. -/
def softmaxWith (O : Ops F) (rmax : List (Option F) → Option F) (rsum : List F → F)
    (xs : List F) : List F :=
  match rmax (xs.map some) with
  | none => []
  | some M =>
    xs.map (fun x => O.div (O.exp (O.sub x M)) (rsum (xs.map (fun y => O.exp (O.sub y M)))))

def softmaxSpec (plus : CMonoid F) (L : MaxOp F) (O : Ops F) (xs : List F) : List F :=
  softmaxWith O (spec (withBot L)) (spec plus) xs

def softmaxTriton (plus : CMonoid F) (L : MaxOp F) (O : Ops F) (kB : Nat) (xs : List F) :
    List F :=
  softmaxWith O (tileReduce (withBot L) kB) (tileReduce plus kB) xs

def softmaxCuda (plus : CMonoid F) (L : MaxOp F) (O : Ops F) (kW : Nat) (xs : List F) :
    List F :=
  softmaxWith O (blockReduce (withBot L) kW) (blockReduce plus kW) xs

/-- `mean = rsum xs / n`. -/
def lnMean (O : Ops F) (rsum : List F → F) (xs : List F) : F :=
  O.div (rsum xs) (O.cast xs.length)

/-- `var = rsum [(x - mean) * (x - mean) | x ← xs] / n`. -/
def lnVar (O : Ops F) (rsum : List F → F) (xs : List F) (mean : F) : F :=
  O.div (rsum (xs.map (fun y => O.mul (O.sub y mean) (O.sub y mean)))) (O.cast xs.length)

/-- Layer normalization of a row, every sum computed by `rsum`:
`y[i] = (xs[i] - mean) / sqrt (var + eps)`. -/
def layernormWith (plus : CMonoid F) (O : Ops F) (rsum : List F → F) (xs : List F) : List F :=
  xs.map (fun x => O.div (O.sub x (lnMean O rsum xs))
    (O.sqrt (plus.op (lnVar O rsum xs (lnMean O rsum xs)) O.eps)))

def layernormSpec (plus : CMonoid F) (O : Ops F) (xs : List F) : List F :=
  layernormWith plus O (spec plus) xs

def layernormTriton (plus : CMonoid F) (O : Ops F) (kB : Nat) (xs : List F) : List F :=
  layernormWith plus O (tileReduce plus kB) xs

def layernormCuda (plus : CMonoid F) (O : Ops F) (kW : Nat) (xs : List F) : List F :=
  layernormWith plus O (blockReduce plus kW) xs

/-- On a non-empty row the `⊥`-extended max is finite: the max of the row. -/
theorem spec_max_cons (L : MaxOp F) (x : F) (rest : List F) :
    spec (withBot L) ((x :: rest).map some) = some (rest.foldl L.max x) := by
  show (rest.map some).foldl (botMax L) (some x) = some (rest.foldl L.max x)
  induction rest generalizing x with
  | nil => rfl
  | cons b l ih => exact ih (L.max x b)

/-- For `n ≥ 1` the spec is the textbook softmax, with `M` the (finite) row max. -/
theorem softmaxSpec_cons (plus : CMonoid F) (L : MaxOp F) (O : Ops F) (x : F) (rest : List F) :
    softmaxSpec plus L O (x :: rest) =
      (x :: rest).map (fun y => O.div (O.exp (O.sub y (rest.foldl L.max x)))
        (spec plus ((x :: rest).map (fun z => O.exp (O.sub z (rest.foldl L.max x)))))) := by
  unfold softmaxSpec softmaxWith
  rw [spec_max_cons]

/-- `_layernorm_kernel` lines 105-106, literally: centre under the mask (masked lanes `0.0`),
then square every lane of the tile, masked ones included. Its reduction is the model's
`tileReduce` of the squared row as soon as the masked lane's `0 * 0` is `0`. -/
theorem triton_var_tile (plus : CMonoid F) (O : Ops F) (kB : Nat) (xs : List F) (mean : F)
    (h00 : O.mul plus.e plus.e = plus.e) :
    tree plus kB ((tile plus (2 ^ kB) (xs.map (fun x => O.sub x mean))).map (fun c => O.mul c c))
      = tileReduce plus kB (xs.map (fun y => O.mul (O.sub y mean) (O.sub y mean))) := by
  rw [tile_map plus plus (fun c => O.mul c c) h00, List.map_map]
  rfl

theorem softmaxWith_congr (O : Ops F) (r r' : List (Option F) → Option F) (s s' : List F → F)
    (xs : List F) (hr : r (xs.map some) = r' (xs.map some))
    (hs : ∀ ys : List F, ys.length = xs.length → s ys = s' ys) :
    softmaxWith O r s xs = softmaxWith O r' s' xs := by
  unfold softmaxWith
  rw [hr]
  cases r' (xs.map some) with
  | none => rfl
  | some M =>
    dsimp only
    rw [hs _ (List.length_map _)]

theorem layernormWith_congr (plus : CMonoid F) (O : Ops F) (s s' : List F → F) (xs : List F)
    (hs : ∀ ys : List F, ys.length = xs.length → s ys = s' ys) :
    layernormWith plus O s xs = layernormWith plus O s' xs := by
  have hmean : lnMean O s xs = lnMean O s' xs := by
    unfold lnMean
    rw [hs xs rfl]
  have hvar : ∀ mean, lnVar O s xs mean = lnVar O s' xs mean := by
    intro mean
    unfold lnVar
    rw [hs _ (List.length_map _)]
  unfold layernormWith
  rw [hmean, hvar]

/-- **`softmax_consistent`**: for every `kB` with `n ≤ 2^kB` and every `kW`, the Triton and the
CUDA softmax models return the spec's row, column for column. -/
theorem softmax_consistent (plus : CMonoid F) (L : MaxOp F) (O : Ops F) (kB kW : Nat)
    (xs : List F) (hB : xs.length ≤ 2 ^ kB) :
    softmaxTriton plus L O kB xs = softmaxSpec plus L O xs ∧
    softmaxCuda plus L O kW xs = softmaxSpec plus L O xs := by
  constructor
  · apply softmaxWith_congr
    · exact tile_reduce_eq_spec _ _ _ (by simpa using hB)
    · intro ys hys
      exact tile_reduce_eq_spec _ _ _ (by omega)
  · apply softmaxWith_congr
    · exact block_reduce_eq_spec _ _ _
    · intro ys _
      exact block_reduce_eq_spec _ _ _

/-- **`layernorm_consistent`**: for every `kB` with `n ≤ 2^kB` and every `kW`, the Triton and
the CUDA layernorm models return the spec's row, column for column. -/
theorem layernorm_consistent (plus : CMonoid F) (O : Ops F) (kB kW : Nat) (xs : List F)
    (hB : xs.length ≤ 2 ^ kB) :
    layernormTriton plus O kB xs = layernormSpec plus O xs ∧
    layernormCuda plus O kW xs = layernormSpec plus O xs := by
  constructor
  · apply layernormWith_congr
    intro ys hys
    exact tile_reduce_eq_spec _ _ _ (by omega)
  · apply layernormWith_congr
    intro ys _
    exact block_reduce_eq_spec _ _ _

end Kernels

end CrossSetup

#print axioms CrossSetup.strided_cover
#print axioms CrossSetup.tile_reduce_eq_spec
#print axioms CrossSetup.tile_reduce_drops_tail
#print axioms CrossSetup.block_reduce_eq_spec
#print axioms CrossSetup.softmax_consistent
#print axioms CrossSetup.layernorm_consistent

-- The faithfulness bridges between the README's model and the kernel text.
#print axioms CrossSetup.strided_unfold
#print axioms CrossSetup.blockReduce_eq_loop
#print axioms CrossSetup.softmaxSpec_cons
#print axioms CrossSetup.triton_var_tile
