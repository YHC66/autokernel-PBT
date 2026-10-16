(* ========================================================================== *)
(*  Cross-setup consistency of the ladder's reductions -- Rocq 9.3, Stdlib only *)
(*                                                                            *)
(*  Statement: proofs/cross-setup-reductions/README.md (six theorems, names   *)
(*  fixed there). Compile: rocq compile CrossSetup.v                          *)
(* ========================================================================== *)

(* HOW EACH DEFINITION MODELS THE KERNEL TEXT

   Commutative monoid (README "(M, ⊕, e)").
     [is_comm_monoid op e] is a Prop-valued record of the three laws
     (associativity, commutativity, left identity; the right identity is
     derived). Every reduction lemma takes it as an argument: it is a
     section variable of [Reductions], so after the section closes each
     theorem is universally quantified over (M, op, e) and the laws.

   spec            README [spec xs = foldl (⊕) e xs]                 -> [spec]

   Triton tile model, kernels/triton/ladder.py
     [tile e B xs]   lines 86-89 and 99-101:
                       offs = tl.arange(0, BLOCK); mask = offs < n_cols;
                       x = tl.load(..., mask=mask, other=<identity>)
                     lane i holds xs[i] if i < n_cols, else the identity
                     ([other=-float("inf")] for max, [other=0.0] for sum).
     [tree k l]      tl.max (line 90), tl.sum (lines 93, 102, 106), modelled
                     as the README's balanced halving tree:
                     tree [a] = a, tree (l ++ r) = tree l ⊕ tree r.
     [tileReduce]    tree over the tile of width B = 2^k = BLOCK
                     (block_for, lines 47-71, picks a power of two).

   CUDA block model, kernels/cuda/softmax.cu and kernels/cuda/layernorm.cu
     [strided_loop], [strided]
                     the loop  for (int c = tid; c < n_cols; c += width)
                     softmax.cu lines 34, 51, 64; layernorm.cu lines 43, 49, 56,
                     transcribed as written: test c < n, emit c, add W.
                     Rocq needs a structurally decreasing argument, so the
                     loop carries fuel n. Fuel n never truncates it: for W >= 1
                     it runs at most n - t times, and [strided_loop_fuel]
                     proves that one more unit of fuel never changes the
                     result once fuel >= n - c (the loop always exits through
                     its own test c < n, never through the fuel).
     [partial0 W xs t]
                     phase 1, thread t: start from the identity, fold the
                     columns in loop order with acc = acc ⊕ x[c]:
                     softmax.cu 33-37 (m = -inf; m = fmaxf(m, x[c]);
                     partial[tid] = m) and 50-54 (acc = 0; acc += expf(..));
                     layernorm.cu 42-45 and 48-52 (acc/sq), stored by
                     block_sum line 18 (partial[tid] = value).
     [step s p t]    one level of the shared-memory tree:
                       if (tid < s) partial[tid] = partial[tid] ⊕ partial[tid+s]
                     softmax.cu 40-42 and 57-59; layernorm.cu 21-23.
     [levels k p]    the loop  for (s = width >> 1; s > 0; s >>= 1)
                     softmax.cu 39, 56; layernorm.cu 20. With width = 2^k it
                     visits s = 2^(k-1), ..., 2, 1 (k levels; none for k = 0),
                     the widest level first, which is [levels (S k') p =
                     levels k' (step (2^k') p)].
     [blockReduce k xs]
                     levels k applied to partial0 (2^k) xs, read at t = 0:
                     softmax.cu 45 (row_max = partial[0]), 62 (denom);
                     layernorm.cu 26 (total = partial[0]).

   WHY THE FUNCTIONAL [step] IS THE IN-PLACE LOOP BODY
     At the level with stride s (s = 2^j, j < k, so 2s <= W) thread t < s
     writes partial[t] only, an index < s, and each index is written by at
     most one thread (its own). It reads partial[t] -- its own cell, before it
     writes it -- and partial[t + s], an index in [s, 2s). No thread writes an
     index >= s at this level, so no read observes a write of the same level:
     every value read is the value before the level, and the new array is the
     pure function [step s p] of the old one. Threads t >= s leave partial[t]
     unchanged, which is the else branch. The __syncthreads() at the end of
     each iteration (softmax.cu 43, 60; layernorm.cu 24) orders the levels,
     and the one after the phase-1 store (softmax.cu 38, 55; layernorm.cu 19)
     orders phase 1 before the tree; this is the sequentialized schedule the
     barriers enforce (concurrency itself is out of scope, see the README).
     Reads stay in bounds: t + s < 2s <= W. [p] is total on nat, but by the
     same bound no level ever reads an index >= W, so the values there are
     irrelevant ([levels_spec] only folds indices < 2^k).

   Softmax and layernorm (Section [Kernels])
     F with abstract add, zero, sub, mul, div, exp, sqrt, cast (nat -> F),
     eps and fmax, all section variables; no law about exp, sqrt, sub, mul,
     div or cast is assumed anywhere.
     -inf: max is taken over [option F], with [None] the bottom element ⊥;
     [omax] extends fmax with ⊥ as identity, and [omax_cm] proves it a
     commutative monoid from fmax commutative and associative (idempotence,
     which the README grants, is not needed). [unbot] reads the row max back
     into F; its None branch is dead for n >= 1 ([softmax_row_max_finite]).
     softmax:   max  = reduction of [map Some xs] in (option F, omax, None)
                       (ladder.py 89-90; softmax.cu 31-45)
                sum  = reduction of [exp (x - M)] in (F, add, zero)
                       (ladder.py 90-93; softmax.cu 49-62); masked Triton
                       lanes hold 0 directly (exp(-inf) = 0, README's table)
                out  = exp (x - M) / S per column
                       (ladder.py 93; softmax.cu 64-66)
     layernorm: mean = reduction of xs / cast n
                       (ladder.py 101-102; layernorm.cu 41-46)
                var  = reduction of (x - mean) * (x - mean) / cast n
                       (ladder.py 105-106; layernorm.cu 48-53)
                out  = (x - mean) / sqrt (var + eps)
                       (ladder.py 107; layernorm.cu 54-57)
     The output loops (store with mask; for c = tid; c < n; c += width) are
     modelled as a map over the row: every column is written once, which is
     [strided_cover] on the CUDA side and the mask on the Triton side.
     Triton's masked lanes of the variance tile hold where(mask, x-mean, 0)^2
     = 0.0 * 0.0 in the kernel; the model puts the identity 0 there, as for
     softmax. [triton_var_tile_literal] proves the two tiles equal given
     mul zero zero = zero, stated as a premise of that lemma only.
*)

From Stdlib Require Import List Arith Lia PeanoNat Permutation.
Import ListNotations.

(* -------------------------------------------------------------------------- *)
(* 0. Commutative monoids and two generic list facts                          *)
(* -------------------------------------------------------------------------- *)

Record is_comm_monoid {M : Type} (op : M -> M -> M) (e : M) : Prop := {
  cm_assoc : forall a b c, op (op a b) c = op a (op b c);
  cm_comm  : forall a b, op a b = op b a;
  cm_id_l  : forall a, op e a = a
}.

Lemma pow2_ge_1 : forall k, 1 <= 2 ^ k.
Proof. induction k; simpl; lia. Qed.

Lemma nth_map_seq_lt : forall {A : Type} (f : nat -> A) B i d,
  i < B -> nth i (map f (seq 0 B)) d = f i.
Proof.
  intros A f B i d Hi.
  rewrite nth_indep with (d' := f 0) by (rewrite length_map, length_seq; exact Hi).
  rewrite map_nth, seq_nth by exact Hi.
  reflexivity.
Qed.

(* -------------------------------------------------------------------------- *)
(* 1. The strided loop, and strided_cover                                     *)
(* -------------------------------------------------------------------------- *)

(* for (int c = t; c < n; c += W) { ...c... }   -- the visited c, in order *)
Fixpoint strided_loop (fuel W n c : nat) : list nat :=
  match fuel with
  | O => []
  | S fuel' => if c <? n then c :: strided_loop fuel' W n (c + W) else []
  end.

(* Thread t's columns: the loop started at c = t, with fuel n. *)
Definition strided (W n t : nat) : list nat := strided_loop n W n t.

Lemma strided_loop_unfold : forall f W n c,
  strided_loop (S f) W n c = if c <? n then c :: strided_loop f W n (c + W) else [].
Proof. reflexivity. Qed.

(* The fuel is never what stops the loop: past n - c, more fuel changes nothing. *)
Lemma strided_loop_fuel : forall W n, 1 <= W ->
  forall f c, n - c <= f -> strided_loop (S f) W n c = strided_loop f W n c.
Proof.
  intros W n HW f. induction f as [|f IH]; intros c Hc.
  - rewrite strided_loop_unfold. simpl.
    destruct (Nat.ltb_spec c n); [lia | reflexivity].
  - rewrite (strided_loop_unfold (S f) W n c), (strided_loop_unfold f W n c).
    destruct (Nat.ltb_spec c n); [|reflexivity].
    rewrite IH by lia. reflexivity.
Qed.

Lemma strided_loop_ge : forall W n f c x, In x (strided_loop f W n c) -> c <= x.
Proof.
  intros W n f. induction f as [|f IH]; intros c x H; simpl in H.
  - contradiction.
  - destruct (c <? n); simpl in H; [|contradiction].
    destruct H as [<- | H]; [lia|].
    apply IH in H. lia.
Qed.

Lemma strided_loop_NoDup : forall W n, 1 <= W -> forall f c, NoDup (strided_loop f W n c).
Proof.
  intros W n HW f. induction f as [|f IH]; intros c; simpl.
  - constructor.
  - destruct (c <? n).
    + constructor; [|apply IH].
      intro H. apply strided_loop_ge in H. lia.
    + constructor.
Qed.

Lemma strided_loop_In : forall W n, 1 <= W -> forall f c, n - c <= f -> forall x,
  In x (strided_loop f W n c) <-> c <= x < n /\ exists q, x = c + q * W.
Proof.
  intros W n HW f. induction f as [|f IH]; intros c Hc x; simpl.
  - split; [intros []|]. intros [H _]. lia.
  - destruct (Nat.ltb_spec c n) as [Hlt|Hge]; simpl.
    + rewrite IH by lia. split.
      * intros [<- | [Hx [q ->]]].
        -- split; [lia|]. exists 0. lia.
        -- split; [lia|]. exists (S q). simpl. lia.
      * intros [Hx [[|q] ->]].
        -- left. lia.
        -- right. split; [simpl in *; lia|]. exists q. simpl. lia.
    + split; [intros []|]. intros [H _]. lia.
Qed.

Lemma strided_In : forall W n t x, 1 <= W ->
  In x (strided W n t) <-> t <= x < n /\ exists q, x = t + q * W.
Proof. intros W n t x HW. unfold strided. apply strided_loop_In; lia. Qed.

Lemma NoDup_flat_map_seq : forall (g : nat -> list nat) a m,
  (forall t, NoDup (g t)) ->
  (forall t t' x, a <= t < a + m -> a <= t' < a + m -> t <> t' ->
                  In x (g t) -> In x (g t') -> False) ->
  NoDup (flat_map g (seq a m)).
Proof.
  intros g a m Hnd. revert a. induction m as [|m IH]; intros a Hdis; simpl.
  - constructor.
  - apply NoDup_app.
    + apply Hnd.
    + apply IH. intros t t' x Ht Ht'. apply Hdis; lia.
    + intros x Hx Hx'. apply in_flat_map in Hx' as [t' [Ht' Hx']].
      apply in_seq in Ht'.
      apply (Hdis a t' x); [lia | lia | lia | exact Hx | exact Hx'].
Qed.

(* README: for W >= 1, the concatenation over t = 0 .. W-1 of strided W n t is
   a permutation of [0, n). *)
Theorem strided_cover : forall W n, 1 <= W ->
  Permutation (flat_map (strided W n) (seq 0 W)) (seq 0 n).
Proof.
  intros W n HW. apply NoDup_Permutation.
  - apply NoDup_flat_map_seq.
    + intro t. apply strided_loop_NoDup. exact HW.
    + intros t t' x Ht Ht' Hne Hx Hx'.
      apply (strided_In W n t x HW) in Hx as [_ [q Hq]].
      apply (strided_In W n t' x HW) in Hx' as [_ [q' Hq']].
      apply Hne.
      assert (E : t = x mod W) by (apply Nat.mod_unique with q; lia).
      assert (E' : t' = x mod W) by (apply Nat.mod_unique with q'; lia).
      congruence.
  - apply seq_NoDup.
  - intro x. split.
    + intros Hx. apply in_flat_map in Hx as [t [_ Hx]].
      apply (strided_In W n t x HW) in Hx as [Hx _].
      apply in_seq. lia.
    + intros Hx. apply in_seq in Hx. apply in_flat_map.
      assert (HW0 : W <> 0) by lia.
      pose proof (Nat.mod_upper_bound x W HW0) as Hm.
      pose proof (Nat.div_mod_eq x W) as Hd.
      exists (x mod W). split.
      * apply in_seq. lia.
      * apply (strided_In W n (x mod W) x HW). split; [lia|].
        exists (x / W). lia.
Qed.

(* -------------------------------------------------------------------------- *)
(* 2. The reductions over a commutative monoid                                *)
(* -------------------------------------------------------------------------- *)

Section Reductions.

Context {M : Type} (op : M -> M -> M) (e : M).

(* README: spec xs = foldl (⊕) e xs *)
Definition spec (xs : list M) : M := fold_left op xs e.

(* ---- Triton tile model ---- *)

(* lane i < B holds xs[i] if i < n (the mask), else the identity (other=) *)
Definition tile (B : nat) (xs : list M) : list M :=
  map (fun i => if i <? length xs then nth i xs e else e) (seq 0 B).

(* tree [a] = a;  tree (l ++ r) = tree l ⊕ tree r,  |l| = |r| = 2^(k-1) *)
Fixpoint tree (k : nat) (l : list M) : M :=
  match k with
  | O => hd e l
  | S k' => op (tree k' (firstn (2 ^ k') l)) (tree k' (skipn (2 ^ k') l))
  end.

Definition tileReduce (k : nat) (xs : list M) : M := tree k (tile (2 ^ k) xs).

(* ---- CUDA block model ---- *)

(* phase 1: thread t folds its strided columns from the identity, acc = acc ⊕ x[c] *)
Definition partial0 (W : nat) (xs : list M) (t : nat) : M :=
  fold_left (fun acc c => op acc (nth c xs e)) (strided W (length xs) t) e.

(* one tree level: if (t < s) partial[t] = partial[t] ⊕ partial[t + s] *)
Definition step (s : nat) (p : nat -> M) (t : nat) : M :=
  if t <? s then op (p t) (p (t + s)) else p t.

(* for (s = W >> 1; s > 0; s >>= 1) with W = 2^k: s = 2^(k-1) first, 1 last *)
Fixpoint levels (k : nat) (p : nat -> M) : nat -> M :=
  match k with
  | O => p
  | S k' => levels k' (step (2 ^ k') p)
  end.

Definition blockReduce (k : nat) (xs : list M) : M :=
  levels k (partial0 (2 ^ k) xs) 0.

(* ---- the laws ---- *)

Context (HM : is_comm_monoid op e).

Lemma op_assoc : forall a b c, op (op a b) c = op a (op b c).
Proof. exact (cm_assoc op e HM). Qed.

Lemma op_comm : forall a b, op a b = op b a.
Proof. exact (cm_comm op e HM). Qed.

Lemma op_id_l : forall a, op e a = a.
Proof. exact (cm_id_l op e HM). Qed.

Lemma op_id_r : forall a, op a e = a.
Proof. intro a. rewrite op_comm. apply op_id_l. Qed.

Lemma op_swap_middle : forall a b c d, op (op a b) (op c d) = op (op a c) (op b d).
Proof.
  intros a b c d. rewrite !op_assoc. f_equal.
  rewrite <- !op_assoc. f_equal. apply op_comm.
Qed.

Lemma fold_op_init : forall l a, fold_left op l a = op a (fold_left op l e).
Proof.
  induction l as [|x l IH]; intros a; simpl.
  - rewrite op_id_r. reflexivity.
  - rewrite (IH (op a x)), (IH (op e x)), op_id_l, op_assoc. reflexivity.
Qed.

Lemma spec_nil : spec [] = e.
Proof. reflexivity. Qed.

Lemma spec_cons : forall a l, spec (a :: l) = op a (spec l).
Proof. intros a l. unfold spec. simpl. rewrite op_id_l. apply fold_op_init. Qed.

Lemma spec_app : forall l1 l2, spec (l1 ++ l2) = op (spec l1) (spec l2).
Proof. intros l1 l2. unfold spec. rewrite fold_left_app. apply fold_op_init. Qed.

Lemma spec_repeat : forall m, spec (repeat e m) = e.
Proof.
  induction m as [|m IH]; cbn [repeat].
  - apply spec_nil.
  - rewrite spec_cons, IH. apply op_id_l.
Qed.

(* Permutation invariance: the heart of "any tree computes the same fold". *)
Lemma fold_perm : forall l l', Permutation l l' ->
  forall a, fold_left op l a = fold_left op l' a.
Proof.
  intros l l' H. induction H as [| x l l' _ IH | x y l | l l' l'' _ IH1 _ IH2];
    intros a; simpl.
  - reflexivity.
  - apply IH.
  - rewrite !op_assoc, (op_comm y x). reflexivity.
  - rewrite IH1. apply IH2.
Qed.

Lemma spec_perm : forall l l', Permutation l l' -> spec l = spec l'.
Proof. intros l l' H. unfold spec. apply fold_perm. exact H. Qed.

(* ---- tile model ---- *)

Lemma tile_pad : forall B xs, length xs <= B -> tile B xs = xs ++ repeat e (B - length xs).
Proof.
  intros B xs H. apply nth_ext with (d := e) (d' := e).
  - unfold tile. rewrite length_map, length_seq, length_app, repeat_length. lia.
  - intros i Hi. unfold tile in *. rewrite length_map, length_seq in Hi.
    rewrite nth_map_seq_lt by exact Hi.
    destruct (Nat.ltb_spec i (length xs)).
    + rewrite app_nth1 by lia. reflexivity.
    + rewrite app_nth2 by lia. rewrite nth_repeat. reflexivity.
Qed.

Lemma tree_spec : forall k l, length l <= 2 ^ k -> tree k l = spec l.
Proof.
  induction k as [|k IH]; intros l Hl.
  - destruct l as [|a [|b l]]; simpl in Hl.
    + reflexivity.
    + simpl. unfold spec. simpl. symmetry. apply op_id_l.
    + lia.
  - cbn [tree].
    rewrite (IH (firstn (2 ^ k) l)) by apply firstn_le_length.
    rewrite (IH (skipn (2 ^ k) l)) by (rewrite length_skipn; simpl in Hl; lia).
    rewrite <- spec_app, firstn_skipn. reflexivity.
Qed.

(* README: length xs <= 2^k -> tileReduce k xs = spec xs *)
Theorem tile_reduce_eq_spec : forall k xs, length xs <= 2 ^ k -> tileReduce k xs = spec xs.
Proof.
  intros k xs H. unfold tileReduce.
  rewrite tree_spec by (unfold tile; rewrite length_map, length_seq; lia).
  rewrite tile_pad by exact H.
  rewrite spec_app, spec_repeat, op_id_r. reflexivity.
Qed.

Lemma tile_firstn : forall B xs, tile B xs = tile B (firstn B xs).
Proof.
  intros B xs. unfold tile. apply map_ext_in. intros i Hi. apply in_seq in Hi.
  rewrite length_firstn, nth_firstn.
  destruct (Nat.ltb_spec i (length xs)), (Nat.ltb_spec i (Nat.min B (length xs))),
           (Nat.ltb_spec i B); try reflexivity; lia.
Qed.

(* README: tileReduce k xs = spec (take (2^k) xs), for every xs *)
Theorem tile_reduce_drops_tail : forall k xs, tileReduce k xs = spec (firstn (2 ^ k) xs).
Proof.
  intros k xs. unfold tileReduce. rewrite tile_firstn.
  change (tileReduce k (firstn (2 ^ k) xs) = spec (firstn (2 ^ k) xs)).
  apply tile_reduce_eq_spec. apply firstn_le_length.
Qed.

(* ---- block model ---- *)

Lemma fold_map : forall (f : nat -> M) l a,
  fold_left (fun acc c => op acc (f c)) l a = fold_left op (map f l) a.
Proof. intros f l. induction l as [|c l IH]; intros a; simpl; [reflexivity | apply IH]. Qed.

Lemma spec_map_op : forall (f g : nat -> M) l,
  spec (map (fun t => op (f t) (g t)) l) = op (spec (map f l)) (spec (map g l)).
Proof.
  intros f g l. induction l as [|t l IH]; cbn [map].
  - rewrite spec_nil. symmetry. apply op_id_l.
  - rewrite !spec_cons, IH. apply op_swap_middle.
Qed.

Lemma map_shift : forall (p : nat -> M) s m a,
  map (fun t => p (t + s)) (seq a m) = map p (seq (a + s) m).
Proof.
  intros p s m. induction m as [|m IH]; intros a; simpl; [reflexivity|].
  rewrite IH. reflexivity.
Qed.

(* (a) one level preserves the fold of the live partials: 2s of them become s *)
Lemma step_level : forall s p,
  spec (map (step s p) (seq 0 s)) = spec (map p (seq 0 (s + s))).
Proof.
  intros s p.
  rewrite (map_ext_in (step s p) (fun t => op (p t) (p (t + s)))).
  2: { intros t Ht. apply in_seq in Ht. unfold step.
       destruct (Nat.ltb_spec t s); [reflexivity | lia]. }
  rewrite spec_map_op, seq_app, map_app, spec_app, map_shift. reflexivity.
Qed.

Lemma levels_spec : forall k p, levels k p 0 = spec (map p (seq 0 (2 ^ k))).
Proof.
  induction k as [|k IH]; intros p.
  - cbn. symmetry. apply op_id_l.
  - cbn [levels]. rewrite IH, step_level.
    replace (2 ^ S k) with (2 ^ k + 2 ^ k) by (simpl; lia). reflexivity.
Qed.

Lemma spec_map_flat_map : forall (f : nat -> M) (g : nat -> list nat) l,
  spec (map (fun t => spec (map f (g t))) l) = spec (map f (flat_map g l)).
Proof.
  intros f g l. induction l as [|t l IH]; [reflexivity|].
  cbn [map flat_map]. rewrite spec_cons, IH, map_app, spec_app. reflexivity.
Qed.

Lemma map_nth_seq : forall xs, map (fun c => nth c xs e) (seq 0 (length xs)) = xs.
Proof.
  induction xs as [|x xs IH]; simpl; [reflexivity|].
  f_equal. rewrite <- seq_shift, map_map. exact IH.
Qed.

(* (b) the fold of the W partials is the fold of the row *)
Lemma spec_partials : forall W xs, 1 <= W ->
  spec (map (partial0 W xs) (seq 0 W)) = spec xs.
Proof.
  intros W xs HW.
  rewrite (map_ext (partial0 W xs)
                   (fun t => spec (map (fun c => nth c xs e) (strided W (length xs) t)))).
  2: { intro t. unfold partial0, spec. apply fold_map. }
  rewrite (spec_map_flat_map (fun c => nth c xs e) (strided W (length xs))).
  rewrite (spec_perm _ _ (Permutation_map (fun c => nth c xs e)
                                          (strided_cover W (length xs) HW))).
  rewrite map_nth_seq. reflexivity.
Qed.

(* README: blockReduce k xs = spec xs, for every k and xs *)
Theorem block_reduce_eq_spec : forall k xs, blockReduce k xs = spec xs.
Proof.
  intros k xs. unfold blockReduce.
  rewrite levels_spec. apply spec_partials. apply pow2_ge_1.
Qed.

End Reductions.

(* A pointwise map commutes with the tile when it sends identity to identity. *)
Lemma map_tile : forall {A B : Type} (f : A -> B) (ea : A) (eb : B) n ys,
  f ea = eb -> map f (tile ea n ys) = tile eb n (map f ys).
Proof.
  intros A B f ea eb n ys Hf. subst eb. unfold tile.
  rewrite map_map, length_map. apply map_ext. intros i.
  destruct (i <? length ys); [symmetry; apply map_nth | reflexivity].
Qed.

(* -------------------------------------------------------------------------- *)
(* 3. Softmax and layer normalization over an abstract carrier               *)
(* -------------------------------------------------------------------------- *)

Section Kernels.

Context {F : Type}
        (add : F -> F -> F) (zero : F)
        (sub mul div : F -> F -> F)
        (exp sqrt : F -> F)
        (cast : nat -> F)
        (eps : F)
        (fmax : F -> F -> F).

(* The only laws: (F, add, zero) is a commutative monoid; fmax is commutative
   and associative. Nothing about exp, sqrt, sub, mul, div, cast or eps. *)
Context (add_cm : is_comm_monoid add zero)
        (fmax_comm : forall a b, fmax a b = fmax b a)
        (fmax_assoc : forall a b c, fmax (fmax a b) c = fmax a (fmax b c)).

(* F extended with a bottom element: None is -inf, the identity of max. *)
Definition omax (a b : option F) : option F :=
  match a, b with
  | Some x, Some y => Some (fmax x y)
  | None, _ => b
  | _, None => a
  end.

Lemma omax_cm : is_comm_monoid omax None.
Proof.
  split.
  - intros [a|] [b|] [c|]; simpl; try reflexivity. rewrite fmax_assoc. reflexivity.
  - intros [a|] [b|]; simpl; try reflexivity. rewrite fmax_comm. reflexivity.
  - intros [a|]; reflexivity.
Qed.

(* The row max as a float; the None branch is dead for n >= 1. *)
Definition unbot (o : option F) : F :=
  match o with Some m => m | None => zero end.

Lemma softmax_row_max_finite : forall xs, 1 <= length xs ->
  exists m, spec omax None (map Some xs) = Some m.
Proof.
  intros [|x xs] H; [simpl in H; lia|]. clear H.
  unfold spec. simpl. revert x.
  induction xs as [|y xs IH]; intros x; simpl; [exists x; reflexivity | apply IH].
Qed.

(* ---- softmax ---- *)

Definition softmaxSpec (xs : list F) : list F :=
  let M := unbot (spec omax None (map Some xs)) in
  let S := spec add zero (map (fun x => exp (sub x M)) xs) in
  map (fun x => div (exp (sub x M)) S) xs.

(* ladder.py _softmax_kernel *)
Definition softmaxTriton (kB : nat) (xs : list F) : list F :=
  let M := unbot (tileReduce omax None kB (map Some xs)) in       (* 86-90 *)
  let S := tileReduce add zero kB (map (fun x => exp (sub x M)) xs) in   (* 90-93 *)
  map (fun x => div (exp (sub x M)) S) xs.                          (* 93 *)

(* softmax.cu softmax_rows *)
Definition softmaxCuda (kW : nat) (xs : list F) : list F :=
  let row_max := unbot (blockReduce omax None kW (map Some xs)) in         (* 31-45 *)
  let denom := blockReduce add zero kW (map (fun x => exp (sub x row_max)) xs) in (* 49-62 *)
  map (fun x => div (exp (sub x row_max)) denom) xs.                       (* 64-66 *)

(* ---- layer normalization ---- *)

Definition layernormSpec (xs : list F) : list F :=
  let n := cast (length xs) in
  let mean := div (spec add zero xs) n in
  let var := div (spec add zero (map (fun x => mul (sub x mean) (sub x mean)) xs)) n in
  map (fun x => div (sub x mean) (sqrt (add var eps))) xs.

(* ladder.py _layernorm_kernel *)
Definition layernormTriton (kB : nat) (xs : list F) : list F :=
  let n := cast (length xs) in
  let mean := div (tileReduce add zero kB xs) n in                       (* 99-102 *)
  let var := div (tileReduce add zero kB
                    (map (fun x => mul (sub x mean) (sub x mean)) xs)) n in  (* 105-106 *)
  map (fun x => div (sub x mean) (sqrt (add var eps))) xs.               (* 107 *)

(* layernorm.cu layernorm_rows *)
Definition layernormCuda (kW : nat) (xs : list F) : list F :=
  let n := cast (length xs) in
  let mean := div (blockReduce add zero kW xs) n in                      (* 41-46 *)
  let var := div (blockReduce add zero kW
                    (map (fun x => mul (sub x mean) (sub x mean)) xs)) n in  (* 48-53 *)
  map (fun x => div (sub x mean) (sqrt (add var eps))) xs.               (* 54-57 *)

(* The kernel's variance tile, where(mask, x - mean, 0) squared lane-wise, is the
   model's tile once 0.0 * 0.0 = 0.0 -- a premise of this lemma only. *)
Lemma triton_var_tile_literal : mul zero zero = zero -> forall B xs mean,
  map (fun c => mul c c) (tile zero B (map (fun x => sub x mean) xs))
  = tile zero B (map (fun x => mul (sub x mean) (sub x mean)) xs).
Proof.
  intros H B xs mean. rewrite map_tile with (eb := zero) by exact H.
  rewrite map_map. reflexivity.
Qed.

(* README: Triton and CUDA softmax agree with the spec on every column, for
   every kB with n <= 2^kB, every kW, and n >= 1. *)
Theorem softmax_consistent : forall kB kW xs,
  1 <= length xs -> length xs <= 2 ^ kB ->
  softmaxTriton kB xs = softmaxSpec xs /\ softmaxCuda kW xs = softmaxSpec xs.
Proof.
  intros kB kW xs _ HB. unfold softmaxTriton, softmaxCuda, softmaxSpec. cbv zeta.
  split.
  - rewrite (tile_reduce_eq_spec _ _ omax_cm) by (rewrite length_map; exact HB).
    rewrite (tile_reduce_eq_spec _ _ add_cm) by (rewrite length_map; exact HB).
    reflexivity.
  - rewrite (block_reduce_eq_spec _ _ omax_cm), (block_reduce_eq_spec _ _ add_cm).
    reflexivity.
Qed.

(* README: the same for layer normalization. *)
Theorem layernorm_consistent : forall kB kW xs,
  length xs <= 2 ^ kB ->
  layernormTriton kB xs = layernormSpec xs /\ layernormCuda kW xs = layernormSpec xs.
Proof.
  intros kB kW xs HB. unfold layernormTriton, layernormCuda, layernormSpec. cbv zeta.
  split.
  - rewrite !(tile_reduce_eq_spec _ _ add_cm) by (rewrite ?length_map; exact HB).
    reflexivity.
  - rewrite !(block_reduce_eq_spec _ _ add_cm). reflexivity.
Qed.

End Kernels.

Print Assumptions strided_cover.
Print Assumptions tile_reduce_eq_spec.
Print Assumptions tile_reduce_drops_tail.
Print Assumptions block_reduce_eq_spec.
Print Assumptions softmax_consistent.
Print Assumptions layernorm_consistent.
