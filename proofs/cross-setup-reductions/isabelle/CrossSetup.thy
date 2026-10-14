(*  Cross-setup consistency of the ladder's reductions, in Isabelle/HOL.

    Isabelle2025-2, session HOL with HOL-Library.Multiset (for mset), no AFP.
    Statement: proofs/cross-setup-reductions/README.md (six theorems, names fixed there).
    Build:     isabelle build -d . -v CrossSetup

Over any commutative monoid (f, e), the Triton kernel's single-tile tree and the CUDA port's
strided-fold-then-block-tree compute  spec f e xs = foldl f e xs.  Hence the Triton and CUDA
softmax and layernorm models compute the spec's row, whatever exp, sqrt, /, -, *, the cast
and eps are. Every abstract operation is a variable of the statement and every law is an
explicit premise of the theorem that uses it: "comm_monoid f e" is HOL's own locale
predicate (Groups.thy: associative, commutative, e a right identity), and the laws of max
are two premises. The theory adds nothing to HOL but definitions. At the end, thm_oracles
prints the oracles each theorem depends on (an unfinished proof would show skip_proof), and
an ML check fails the build if any of the six theorems depends on one.

MODEL <-> KERNEL TEXT
  T = kernels/triton/ladder.py, S = kernels/cuda/softmax.cu, L = kernels/cuda/layernorm.cu

  Triton tile model
    tile e B xs      offs = tl.arange(0, BLOCK); mask = offs < n_cols;
                     tl.load(..., mask=mask, other=<identity>): lane i holds xs ! i if
                     i < n, else the identity e (other=-inf for max, other=0.0 for sum)
                                                                T 86-89, 99-101
    tree f k l       tl.max(x, axis=0) (T 90), tl.sum(.., axis=0) (T 93, 102, 106), as the
                     README's balanced halving tree: tree [a] = a, and
                     tree (l @ r) = f (tree l) (tree r) with |l| = |r| = 2^(k-1)
    tileReduce       tree over the tile of width BLOCK = 2^k; block_for (T 47-71) rounds
                     n_cols up to a power of two, which is the hypothesis n <= 2^kB

  CUDA block model
    strided W n c    the loop  for (int c = tid; c < n_cols; c += width),
                     S 34, 51, 64; L 43, 49, 56. Defined by its own recursion equation,
                     strided W n c = (if c < n then c # strided W n (c + W) else []),
                     as a partial function: for W = 0 and c < n the C loop does not
                     terminate, and neither does this recursion. strided_terminates proves
                     it terminates for every W >= 1.
    partial0         phase 1 of thread t: start from the identity, fold the thread's
                     columns in loop order with acc = acc (+) x[c], store partial[tid]:
                     S 33-37 (m = -inf; m = fmaxf(m, x); partial[tid] = m),
                     S 50-54 (acc = 0; acc += expf(..); partial[tid] = acc),
                     L 42-45 and 48-52 (acc, sq), stored by block_sum at L 18
    step f s p t     one level of the shared-memory tree:
                       if (tid < s) partial[tid] = partial[tid] (+) partial[tid + s]
                     S 40-42, 57-59; L 21-23
    levels f k       step 1 o step 2 o ... o step (2^(k-1)), literally (foldr of o over
                     the list [step 1, step 2, ..., step (2^(k-1))]): the level
                     s = 2^(k-1) = W/2 runs first, s = 1 last, none when k = 0
    treeLoop f s p   the loop  for (int s = width >> 1; s > 0; s >>= 1),  literally:
                     S 39, 56; L 20. blockReduce_eq_loop proves it equals levels.
    blockReduce      levels applied to the W = 2^k partials, read at partial[0]:
                     S 45 (row_max), S 62 (denom), L 26 (total). The launch contract
                     makes W a power of two (S 14, L 11; threads_for in
                     kernels/cuda/ladder.py, lines 44-52).

  Kernels, over an abstract carrier 'f with fadd fzero fsub fmul fdiv fexp fsqrt fcast feps
  fmax, all variables of the statements
    maxbot fmax      max on 'f option with None = -inf as identity: negative_infinity()
                     (S 19-21, 33) and other=-float("inf") (T 89). The row max is read
                     back into 'f with "the"; on the empty row it is "the None", an
                     unspecified value, but then no column is written (both kernels store
                     nothing when n_cols = 0), so the theorems need no n >= 1.
    softmaxSpec      y = exp (x - M) / S, M the row max, S = spec of exp (x - M)
    softmaxTriton    _softmax_kernel, T 84-93: max tile with masked lanes None (T 89-90);
                     sum tile with masked lanes fzero directly, the README's reading of
                     exp(-inf - M) = 0 (T 90-93; side lemma triton_sum_tile); output T 93
    softmaxCuda      softmax_rows: row max S 31-45, denominator S 49-62, output S 64-66
    layernormSpec    mean = spec x / n; var = spec ((x - mean) * (x - mean)) / n;
                     y = (x - mean) / sqrt (var + eps)
    layernormTriton  _layernorm_kernel, T 97-107: x tile other=0.0 (T 101), mean (T 102),
                     var over centered * centered (T 105-106) with masked lanes fzero
                     directly (side lemma triton_var_tile: the kernel's lanes hold
                     0.0 * 0.0), output centered / tl.sqrt(var + EPS) (T 107)
    layernormCuda    layernorm_rows: mean L 41-46, var L 48-53, sqrtf(var + eps) L 54,
                     output L 56-58
  The output stores (Triton's masked tl.store, CUDA's third strided loop) are a map over
  the row: every column is written exactly once (the mask; strided_cover), and the value
  written depends only on x[c] and on block-uniform scalars.

WHY THE FUNCTIONAL step IS THE IN-PLACE LOOP BODY
  At the level with stride s (s = 2^j, j < k, so 2s <= W), thread t < s runs
  partial[t] = partial[t] (+) partial[t + s] and threads t >= s do nothing. Every write lands
  at an index < s, and index t is written by thread t alone. Every read is at t, by thread
  t itself before its own write, or at t + s, an index in [s, 2s) that no thread writes at
  this level. So no read observes a write of the same level, under any interleaving, and
  the level maps the array p to step f s p: p t (+) p (t + s) below s, p t elsewhere. The
  __syncthreads() closing each level (S 43, 60; L 24) hands exactly that array to the next
  level, and the one after the phase-1 store (S 38, 55; L 19) orders phase 1 before the
  tree. The model is this barrier-enforced schedule; concurrency itself is out of scope
  (README). Reads stay below 2s <= W, so the values of p at indices >= W never matter.
*)

theory CrossSetup
  imports Main "HOL-Library.Multiset"
begin

type_synonym 'a binop = "'a \<Rightarrow> 'a \<Rightarrow> 'a"


section \<open>The models\<close>

subsection \<open>The specification\<close>

text \<open>README: \<open>spec xs = foldl (\<oplus>) e xs\<close>.\<close>

definition spec :: "'m binop \<Rightarrow> 'm \<Rightarrow> 'm list \<Rightarrow> 'm" where
  "spec f e xs = foldl f e xs"


subsection \<open>Triton: the tile model\<close>

text \<open>Lane \<open>i\<close> of \<open>tl.load(.., mask=offs < n_cols, other=e)\<close>, for \<open>i \<leftarrow> 0 .. B-1\<close>.\<close>

definition tile :: "'m \<Rightarrow> nat \<Rightarrow> 'm list \<Rightarrow> 'm list" where
  "tile e B xs = map (\<lambda>i. if i < length xs then xs ! i else e) [0..<B]"

text \<open>The balanced halving tree over a list of length \<open>2^k\<close>.\<close>

fun tree :: "'m binop \<Rightarrow> nat \<Rightarrow> 'm list \<Rightarrow> 'm" where
  "tree f 0 l = hd l"
| "tree f (Suc k) l = f (tree f k (take (2 ^ k) l)) (tree f k (drop (2 ^ k) l))"

definition tileReduce :: "'m binop \<Rightarrow> 'm \<Rightarrow> nat \<Rightarrow> 'm list \<Rightarrow> 'm" where
  "tileReduce f e k xs = tree f k (tile e (2 ^ k) xs)"


subsection \<open>CUDA: the block model\<close>

text \<open>\<open>for (int c = t; c < n; c += W)\<close>: the columns visited, in order. A partial function,
  like the loop: it is left unspecified where the loop does not terminate (\<open>W = 0\<close>, \<open>c < n\<close>).\<close>

function (domintros) strided :: "nat \<Rightarrow> nat \<Rightarrow> nat \<Rightarrow> nat list" where
  "strided W n c = (if c < n then c # strided W n (c + W) else [])"
  by pat_completeness auto

text \<open>Phase 1: thread \<open>t\<close> folds its columns from the identity, \<open>acc = acc \<oplus> x[c]\<close>.\<close>

definition partial0 :: "'m binop \<Rightarrow> 'm \<Rightarrow> nat \<Rightarrow> 'm list \<Rightarrow> nat \<Rightarrow> 'm" where
  "partial0 f e W xs t = foldl (\<lambda>acc c. f acc (xs ! c)) e (strided W (length xs) t)"

text \<open>One level of the in-place tree: \<open>if (t < s) partial[t] = partial[t] \<oplus> partial[t + s]\<close>.\<close>

definition step :: "'m binop \<Rightarrow> nat \<Rightarrow> (nat \<Rightarrow> 'm) \<Rightarrow> nat \<Rightarrow> 'm" where
  "step f s p t = (if t < s then f (p t) (p (t + s)) else p t)"

text \<open>\<open>levels f k = step f 1 \<circ> step f 2 \<circ> \<dots> \<circ> step f (2^(k-1))\<close>: the level \<open>s = 2^(k-1)\<close>
  runs first, and \<open>k = 0\<close> runs no level.\<close>

definition levels :: "'m binop \<Rightarrow> nat \<Rightarrow> (nat \<Rightarrow> 'm) \<Rightarrow> nat \<Rightarrow> 'm" where
  "levels f k = foldr (\<circ>) (map (\<lambda>j. step f (2 ^ j)) [0..<k]) id"

definition blockReduce :: "'m binop \<Rightarrow> 'm \<Rightarrow> nat \<Rightarrow> 'm list \<Rightarrow> 'm" where
  "blockReduce f e k xs = levels f k (partial0 f e (2 ^ k) xs) 0"

text \<open>The kernels' tree loop as written: \<open>for (s = width >> 1; s > 0; s >>= 1) step s\<close>.\<close>

function treeLoop :: "'m binop \<Rightarrow> nat \<Rightarrow> (nat \<Rightarrow> 'm) \<Rightarrow> nat \<Rightarrow> 'm" where
  "treeLoop f s p = (if 0 < s then treeLoop f (s div 2) (step f s p) else p)"
  by pat_completeness auto
termination by (relation "measure (\<lambda>(f, s, p). s)") auto

text \<open>Unfolding the loop equation under an undecided test would not terminate in the
  simplifier, so it is replaced by its two cases.\<close>

declare treeLoop.simps [simp del]

lemma treeLoop_stop [simp]: "treeLoop f 0 p = p"
  by (simp add: treeLoop.simps)

lemma treeLoop_level [simp]: "0 < s \<Longrightarrow> treeLoop f s p = treeLoop f (s div 2) (step f s p)"
  by (simp add: treeLoop.simps)


subsection \<open>The two kernels over an abstract carrier\<close>

text \<open>\<open>max\<close> on \<open>'f option\<close>, with \<open>None\<close> the bottom element \<open>-inf\<close>.\<close>

fun maxbot :: "'f binop \<Rightarrow> 'f option binop" where
  "maxbot fmax None b = b"
| "maxbot fmax (Some a) None = Some a"
| "maxbot fmax (Some a) (Some b) = Some (fmax a b)"

definition softmaxSpec ::
  "'f binop \<Rightarrow> 'f \<Rightarrow> 'f binop \<Rightarrow> 'f binop \<Rightarrow> ('f \<Rightarrow> 'f) \<Rightarrow> 'f binop \<Rightarrow> 'f list \<Rightarrow> 'f list"
where
  "softmaxSpec fadd fzero fsub fdiv fexp fmax xs =
     (let M = the (spec (maxbot fmax) None (map Some xs));
          S = spec fadd fzero (map (\<lambda>x. fexp (fsub x M)) xs)
      in map (\<lambda>x. fdiv (fexp (fsub x M)) S) xs)"

definition softmaxTriton ::
  "'f binop \<Rightarrow> 'f \<Rightarrow> 'f binop \<Rightarrow> 'f binop \<Rightarrow> ('f \<Rightarrow> 'f) \<Rightarrow> 'f binop \<Rightarrow> nat \<Rightarrow> 'f list
     \<Rightarrow> 'f list"
where
  "softmaxTriton fadd fzero fsub fdiv fexp fmax kB xs =
     (let M = the (tileReduce (maxbot fmax) None kB (map Some xs));
          S = tileReduce fadd fzero kB (map (\<lambda>x. fexp (fsub x M)) xs)
      in map (\<lambda>x. fdiv (fexp (fsub x M)) S) xs)"

definition softmaxCuda ::
  "'f binop \<Rightarrow> 'f \<Rightarrow> 'f binop \<Rightarrow> 'f binop \<Rightarrow> ('f \<Rightarrow> 'f) \<Rightarrow> 'f binop \<Rightarrow> nat \<Rightarrow> 'f list
     \<Rightarrow> 'f list"
where
  "softmaxCuda fadd fzero fsub fdiv fexp fmax kW xs =
     (let row_max = the (blockReduce (maxbot fmax) None kW (map Some xs));
          denom = blockReduce fadd fzero kW (map (\<lambda>x. fexp (fsub x row_max)) xs)
      in map (\<lambda>x. fdiv (fexp (fsub x row_max)) denom) xs)"

definition layernormSpec ::
  "'f binop \<Rightarrow> 'f \<Rightarrow> 'f binop \<Rightarrow> 'f binop \<Rightarrow> 'f binop \<Rightarrow> ('f \<Rightarrow> 'f) \<Rightarrow> (nat \<Rightarrow> 'f) \<Rightarrow> 'f
     \<Rightarrow> 'f list \<Rightarrow> 'f list"
where
  "layernormSpec fadd fzero fsub fmul fdiv fsqrt fcast feps xs =
     (let n = fcast (length xs);
          mean = fdiv (spec fadd fzero xs) n;
          var = fdiv (spec fadd fzero (map (\<lambda>x. fmul (fsub x mean) (fsub x mean)) xs)) n
      in map (\<lambda>x. fdiv (fsub x mean) (fsqrt (fadd var feps))) xs)"

definition layernormTriton ::
  "'f binop \<Rightarrow> 'f \<Rightarrow> 'f binop \<Rightarrow> 'f binop \<Rightarrow> 'f binop \<Rightarrow> ('f \<Rightarrow> 'f) \<Rightarrow> (nat \<Rightarrow> 'f) \<Rightarrow> 'f
     \<Rightarrow> nat \<Rightarrow> 'f list \<Rightarrow> 'f list"
where
  "layernormTriton fadd fzero fsub fmul fdiv fsqrt fcast feps kB xs =
     (let n = fcast (length xs);
          mean = fdiv (tileReduce fadd fzero kB xs) n;
          var = fdiv (tileReduce fadd fzero kB (map (\<lambda>x. fmul (fsub x mean) (fsub x mean)) xs)) n
      in map (\<lambda>x. fdiv (fsub x mean) (fsqrt (fadd var feps))) xs)"

definition layernormCuda ::
  "'f binop \<Rightarrow> 'f \<Rightarrow> 'f binop \<Rightarrow> 'f binop \<Rightarrow> 'f binop \<Rightarrow> ('f \<Rightarrow> 'f) \<Rightarrow> (nat \<Rightarrow> 'f) \<Rightarrow> 'f
     \<Rightarrow> nat \<Rightarrow> 'f list \<Rightarrow> 'f list"
where
  "layernormCuda fadd fzero fsub fmul fdiv fsqrt fcast feps kW xs =
     (let n = fcast (length xs);
          mean = fdiv (blockReduce fadd fzero kW xs) n;
          var = fdiv (blockReduce fadd fzero kW (map (\<lambda>x. fmul (fsub x mean) (fsub x mean)) xs)) n
      in map (\<lambda>x. fdiv (fsub x mean) (fsqrt (fadd var feps))) xs)"


section \<open>The strided loop covers the row\<close>

lemma strided_terminates:
  assumes "0 < W"
  shows "strided_dom (W, n, c)"
proof (induction c rule: measure_induct_rule[where f = "\<lambda>c. n - c"])
  case (less c)
  show ?case
  proof (rule strided.domintros)
    assume "c < n"
    with assms have "n - (c + W) < n - c" by simp
    then show "strided_dom (W, n, c + W)" by (rule less)
  qed
qed

text \<open>The loop's own recursion equation, for every \<open>W \<ge> 1\<close>.\<close>

lemma strided_simps:
  "0 < W \<Longrightarrow> strided W n c = (if c < n then c # strided W n (c + W) else [])"
  by (rule strided.psimps[OF strided_terminates])

text \<open>Its two cases, as conditional rewrite rules (the equation itself would make the
  simplifier unfold the loop forever under an undecided test \<open>c < n\<close>).\<close>

lemma strided_Cons: "0 < W \<Longrightarrow> c < n \<Longrightarrow> strided W n c = c # strided W n (c + W)"
  using strided_simps[of W n c] by simp

lemma strided_Nil: "0 < W \<Longrightarrow> \<not> c < n \<Longrightarrow> strided W n c = []"
  using strided_simps[of W n c] by simp

lemma mod_eq_next:
  fixes c x W :: nat
  assumes "c < x" and "x mod W = c mod W"
  shows "c + W \<le> x"
proof -
  have "W dvd x - c" using assms mod_eq_dvd_iff_nat[of c x W] by simp
  then have "W \<le> x - c" using assms(1) by (simp add: dvd_imp_le)
  then show ?thesis using assms(1) by simp
qed

text \<open>Thread \<open>t\<close> visits exactly the columns \<open>x < n\<close> with \<open>x \<ge> t\<close> and \<open>x \<equiv> t (mod W)\<close>.\<close>

lemma set_strided:
  assumes "0 < W"
  shows "set (strided W n c) = {x. c \<le> x \<and> x < n \<and> x mod W = c mod W}"
proof (induction c rule: measure_induct_rule[where f = "\<lambda>c. n - c"])
  case (less c)
  show ?case
  proof (cases "c < n")
    case True
    have IH: "set (strided W n (c + W)) = {x. c + W \<le> x \<and> x < n \<and> x mod W = c mod W}"
      using less[of "c + W"] True assms by simp
    have "{x. c \<le> x \<and> x < n \<and> x mod W = c mod W}
        = insert c {x. c + W \<le> x \<and> x < n \<and> x mod W = c mod W}"
    proof
      show "{x. c \<le> x \<and> x < n \<and> x mod W = c mod W}
          \<subseteq> insert c {x. c + W \<le> x \<and> x < n \<and> x mod W = c mod W}"
      proof
        fix x assume x: "x \<in> {x. c \<le> x \<and> x < n \<and> x mod W = c mod W}"
        show "x \<in> insert c {x. c + W \<le> x \<and> x < n \<and> x mod W = c mod W}"
        proof (cases "x = c")
          case False
          with x have "c < x" by simp
          with x have "c + W \<le> x" by (simp add: mod_eq_next)
          with x show ?thesis by simp
        qed simp
      qed
      show "insert c {x. c + W \<le> x \<and> x < n \<and> x mod W = c mod W}
          \<subseteq> {x. c \<le> x \<and> x < n \<and> x mod W = c mod W}"
        using True by auto
    qed
    moreover have "strided W n c = c # strided W n (c + W)" using assms True by (rule strided_Cons)
    ultimately show ?thesis using IH by simp
  next
    case False
    with assms show ?thesis by (auto simp: strided_Nil)
  qed
qed

lemma distinct_strided:
  assumes "0 < W"
  shows "distinct (strided W n c)"
proof (induction c rule: measure_induct_rule[where f = "\<lambda>c. n - c"])
  case (less c)
  show ?case
  proof (cases "c < n")
    case True
    have "distinct (strided W n (c + W))" using less[of "c + W"] True assms by simp
    moreover have "c \<notin> set (strided W n (c + W))" using assms by (simp add: set_strided)
    ultimately show ?thesis using True assms by (simp add: strided_Cons)
  next
    case False
    with assms show ?thesis by (simp add: strided_Nil)
  qed
qed

text \<open>The README's "ascending".\<close>

lemma sorted_strided:
  assumes "0 < W"
  shows "sorted_wrt (<) (strided W n c)"
proof (induction c rule: measure_induct_rule[where f = "\<lambda>c. n - c"])
  case (less c)
  show ?case
  proof (cases "c < n")
    case True
    have "sorted_wrt (<) (strided W n (c + W))" using less[of "c + W"] True assms by simp
    moreover have "\<forall>x \<in> set (strided W n (c + W)). c < x" using assms by (simp add: set_strided)
    ultimately show ?thesis using True assms by (simp add: strided_Cons)
  next
    case False
    with assms show ?thesis by (simp add: strided_Nil)
  qed
qed

lemma distinct_concat_map_upt:
  assumes "\<And>t. t < m \<Longrightarrow> distinct (g t)"
    and "\<And>s t. s < m \<Longrightarrow> t < m \<Longrightarrow> s \<noteq> t \<Longrightarrow> set (g s) \<inter> set (g t) = {}"
  shows "distinct (concat (map g [0..<m]))"
  using assms
proof (induction m)
  case 0
  then show ?case by simp
next
  case (Suc m)
  have "distinct (concat (map g [0..<m]))"
    by (rule Suc.IH) (use Suc.prems in auto)
  moreover have "distinct (g m)" using Suc.prems(1) by simp
  moreover have "set (concat (map g [0..<m])) \<inter> set (g m) = {}"
  proof -
    have "set (g t) \<inter> set (g m) = {}" if "t < m" for t
      using that Suc.prems(2)[of t m] by simp
    then show ?thesis by (auto simp: disjoint_iff)
  qed
  ultimately show ?case by simp
qed

text \<open>\<^bold>\<open>strided_cover\<close>: for \<open>W \<ge> 1\<close>, the concatenation over \<open>t = 0 .. W-1\<close> of \<open>strided W n t\<close>
  is a permutation of \<open>[0, n)\<close> (in Isabelle, \<open>mset xs = mset ys\<close>; HOL-Combinatorics' \<open>perm\<close> is
  an input abbreviation for it): every column is visited by exactly one thread, once.\<close>

theorem strided_cover:
  assumes "1 \<le> W"
  shows "mset (concat (map (strided W n) [0..<W])) = mset [0..<n]"
proof -
  have W: "0 < W" using assms by simp
  have "distinct (concat (map (strided W n) [0..<W]))"
  proof (rule distinct_concat_map_upt)
    show "distinct (strided W n t)" for t using W by (rule distinct_strided)
    show "set (strided W n s) \<inter> set (strided W n t) = {}" if "s < W" "t < W" "s \<noteq> t" for s t
      using that W by (auto simp: set_strided)
  qed
  moreover have "set (concat (map (strided W n) [0..<W])) = set [0..<n]"
  proof
    show "set (concat (map (strided W n) [0..<W])) \<subseteq> set [0..<n]"
      using W by (auto simp: set_strided)
    show "set [0..<n] \<subseteq> set (concat (map (strided W n) [0..<W]))"
    proof
      fix x assume "x \<in> set [0..<n]"
      then have "x \<in> set (strided W n (x mod W))" using W by (simp add: set_strided)
      moreover have "x mod W < W" using W by simp
      ultimately show "x \<in> set (concat (map (strided W n) [0..<W]))" by auto
    qed
  qed
  ultimately show ?thesis
    using set_eq_iff_mset_eq_distinct[of "concat (map (strided W n) [0..<W])" "[0..<n]"] by simp
qed


section \<open>Structural facts about the tile and the tree levels\<close>

lemma length_tile [simp]: "length (tile e B xs) = B"
  by (simp add: tile_def)

text \<open>The tile is the row's first \<open>B\<close> columns, padded with the identity.\<close>

lemma tile_eq: "tile e B xs = take B xs @ replicate (B - length xs) e"
  by (rule nth_equalityI) (auto simp: tile_def nth_append min_def)

lemma foldr_comp: "foldr (\<circ>) gs g = foldr (\<circ>) gs id \<circ> g"
  by (induction gs) auto

lemma levels_Suc: "levels f (Suc j) p = levels f j (step f (2 ^ j) p)"
proof -
  have "levels f (Suc j) = foldr (\<circ>) (map (\<lambda>i. step f (2 ^ i)) [0..<j]) (step f (2 ^ j))"
    by (simp add: levels_def)
  also have "\<dots> = levels f j \<circ> step f (2 ^ j)"
    unfolding levels_def by (rule foldr_comp)
  finally show ?thesis by simp
qed


section \<open>Folding over a commutative monoid\<close>

text \<open>Inside HOL's locale \<open>comm_monoid\<close> (parameters \<open>f\<close>, written \<open>\<^bold>*\<close>, and \<open>z\<close>, written
  \<open>\<^bold>1\<close>); the six theorems below take \<open>comm_monoid f e\<close> as an explicit premise and interpret it.\<close>

context comm_monoid
begin

lemma foldl_init: "foldl f a xs = a \<^bold>* foldl f \<^bold>1 xs"
proof (induction xs arbitrary: a)
  case Nil
  show ?case by simp
next
  case (Cons x xs)
  have "foldl f a (x # xs) = foldl f (a \<^bold>* x) xs" by simp
  also have "\<dots> = (a \<^bold>* x) \<^bold>* foldl f \<^bold>1 xs" by (rule Cons.IH)
  also have "\<dots> = a \<^bold>* (x \<^bold>* foldl f \<^bold>1 xs)" by (rule assoc)
  also have "\<dots> = a \<^bold>* foldl f x xs" using Cons.IH[of x] by simp
  also have "\<dots> = a \<^bold>* foldl f \<^bold>1 (x # xs)" by simp
  finally show ?case .
qed

lemma spec_Nil [simp]: "spec f \<^bold>1 [] = \<^bold>1"
  by (simp add: spec_def)

lemma spec_Cons [simp]: "spec f \<^bold>1 (x # xs) = x \<^bold>* spec f \<^bold>1 xs"
  using foldl_init[of x xs] by (simp add: spec_def)

lemma spec_append: "spec f \<^bold>1 (xs @ ys) = spec f \<^bold>1 xs \<^bold>* spec f \<^bold>1 ys"
  by (induction xs) (simp_all add: assoc)

lemma spec_replicate: "spec f \<^bold>1 (replicate m \<^bold>1) = \<^bold>1"
  by (induction m) simp_all

lemma spec_concat: "spec f \<^bold>1 (concat L) = spec f \<^bold>1 (map (spec f \<^bold>1) L)"
  by (induction L) (simp_all add: spec_append)

lemma spec_map_op:
  "spec f \<^bold>1 (map (\<lambda>t. g t \<^bold>* h t) l) = spec f \<^bold>1 (map g l) \<^bold>* spec f \<^bold>1 (map h l)"
  by (induction l) (simp_all add: ac_simps)

text \<open>Permutation invariance: the fold of a commutative monoid does not see the order.\<close>

lemma spec_perm:
  assumes "mset xs = mset ys"
  shows "spec f \<^bold>1 xs = spec f \<^bold>1 ys"
proof -
  have "List.fold (\<lambda>x a. a \<^bold>* x) xs = List.fold (\<lambda>x a. a \<^bold>* x) ys"
    using assms by (intro fold_multiset_equiv) (auto simp: fun_eq_iff ac_simps)
  then show ?thesis by (simp add: spec_def foldl_conv_fold)
qed

text \<open>The spec is HOL's list fold of the monoid (\<open>comm_monoid_list.F\<close>, i.e. \<open>sum_list\<close> for \<open>+\<close>).\<close>

lemma spec_foldr: "spec f \<^bold>1 xs = foldr f xs \<^bold>1"
  by (induction xs) simp_all


subsection \<open>The tile\<close>

lemma tree_eq_spec: "length l = 2 ^ k \<Longrightarrow> tree f k l = spec f \<^bold>1 l"
proof (induction k arbitrary: l)
  case 0
  then show ?case by (cases l) auto
next
  case (Suc k)
  have take: "length (take (2 ^ k) l) = 2 ^ k" using Suc.prems by simp
  have drop: "length (drop (2 ^ k) l) = 2 ^ k" using Suc.prems by simp
  have "tree f (Suc k) l = tree f k (take (2 ^ k) l) \<^bold>* tree f k (drop (2 ^ k) l)" by simp
  also have "\<dots> = spec f \<^bold>1 (take (2 ^ k) l) \<^bold>* spec f \<^bold>1 (drop (2 ^ k) l)"
    by (simp only: Suc.IH[OF take] Suc.IH[OF drop])
  also have "\<dots> = spec f \<^bold>1 l" by (simp add: spec_append[symmetric])
  finally show ?case .
qed


subsection \<open>The block\<close>

lemma sum_step: "spec f \<^bold>1 (map (step f s p) [0..<s]) = spec f \<^bold>1 (map p [0..<s + s])"
proof -
  have live: "map (step f s p) [0..<s] = map (\<lambda>t. p t \<^bold>* p (t + s)) [0..<s]"
    by (rule map_cong) (simp_all add: step_def)
  have shift: "map (\<lambda>t. p (t + s)) [0..<s] = map p [s..<s + s]"
    by (simp add: map_add_upt[of s s, symmetric])
  have halves: "[0..<s + s] = [0..<s] @ [s..<s + s]"
    by (simp add: upt_add_eq_append[of 0 s s])
  show ?thesis
    by (simp only: live spec_map_op shift halves map_append spec_append)
qed

lemma levels_spec: "levels f j p 0 = spec f \<^bold>1 (map p [0..<2 ^ j])"
proof (induction j arbitrary: p)
  case 0
  show ?case by (simp add: levels_def)
next
  case (Suc j)
  have "levels f (Suc j) p 0 = levels f j (step f (2 ^ j) p) 0"
    by (simp add: levels_Suc)
  also have "\<dots> = spec f \<^bold>1 (map (step f (2 ^ j) p) [0..<2 ^ j])" by (rule Suc.IH)
  also have "\<dots> = spec f \<^bold>1 (map p [0..<2 ^ j + 2 ^ j])" by (rule sum_step)
  also have "\<dots> = spec f \<^bold>1 (map p [0..<2 ^ Suc j])" by (simp add: mult_2)
  finally show ?case .
qed

lemma partial0_spec:
  "partial0 f \<^bold>1 W xs t = spec f \<^bold>1 (map ((!) xs) (strided W (length xs) t))"
  by (simp add: partial0_def spec_def foldl_map)

lemma partial0_fun:
  "partial0 f \<^bold>1 W xs = (\<lambda>t. spec f \<^bold>1 (map ((!) xs) (strided W (length xs) t)))"
  by (rule ext) (rule partial0_spec)

lemma spec_partials:
  assumes "1 \<le> W"
  shows "spec f \<^bold>1 (map (partial0 f \<^bold>1 W xs) [0..<W]) = spec f \<^bold>1 xs"
proof -
  have "spec f \<^bold>1 (map (partial0 f \<^bold>1 W xs) [0..<W])
      = spec f \<^bold>1 (map ((!) xs) (concat (map (strided W (length xs)) [0..<W])))"
    by (simp add: partial0_fun spec_concat map_concat comp_def)
  also have "\<dots> = spec f \<^bold>1 (map ((!) xs) [0..<length xs])"
    using strided_cover[OF assms] by (intro spec_perm) simp
  also have "\<dots> = spec f \<^bold>1 xs" by (simp add: map_nth)
  finally show ?thesis .
qed

end


section \<open>The four reduction theorems\<close>

text \<open>\<^bold>\<open>tile_reduce_drops_tail\<close>: the tile model reduces the first \<open>2^k\<close> columns, for every row;
  a row wider than the tile loses its tail (the measured tile-tail defect).\<close>

theorem tile_reduce_drops_tail:
  assumes "comm_monoid f e"
  shows "tileReduce f e k xs = spec f e (take (2 ^ k) xs)"
proof -
  interpret comm_monoid f e by (fact assms)
  have "tileReduce f e k xs = spec f e (tile e (2 ^ k) xs)"
    by (simp add: tileReduce_def tree_eq_spec)
  also have "\<dots> = spec f e (take (2 ^ k) xs)"
    by (simp add: tile_eq spec_append spec_replicate)
  finally show ?thesis .
qed

text \<open>\<^bold>\<open>tile_reduce_eq_spec\<close>: when the tile holds the whole row, the tile model is the spec.\<close>

theorem tile_reduce_eq_spec:
  assumes "comm_monoid f e" and "length xs \<le> 2 ^ k"
  shows "tileReduce f e k xs = spec f e xs"
  using tile_reduce_drops_tail[OF assms(1)] assms(2) by simp

text \<open>\<^bold>\<open>block_reduce_eq_spec\<close>: the block model is the spec, for every \<open>k\<close> and every row: no
  relation between \<open>W = 2^k\<close> and \<open>n\<close> is needed.\<close>

theorem block_reduce_eq_spec:
  assumes "comm_monoid f e"
  shows "blockReduce f e k xs = spec f e xs"
proof -
  interpret comm_monoid f e by (fact assms)
  have "blockReduce f e k xs = spec f e (map (partial0 f e (2 ^ k) xs) [0..<2 ^ k])"
    by (simp add: blockReduce_def levels_spec)
  also have "\<dots> = spec f e xs"
    by (rule spec_partials) simp
  finally show ?thesis .
qed


section \<open>Softmax and layer normalization\<close>

lemma comm_monoid_maxbot:
  assumes max_comm: "\<And>a b. fmax a b = fmax b a"
    and max_assoc: "\<And>a b c. fmax (fmax a b) c = fmax a (fmax b c)"
  shows "comm_monoid (maxbot fmax) None"
proof unfold_locales
  fix a b c :: "'a option"
  show "maxbot fmax (maxbot fmax a b) c = maxbot fmax a (maxbot fmax b c)"
    by (cases a; cases b; cases c) (simp_all add: max_assoc)
  show "maxbot fmax a b = maxbot fmax b a"
    by (cases a; cases b) (simp_all add: max_comm)
  show "maxbot fmax a None = a"
    by (cases a) simp_all
qed

text \<open>\<^bold>\<open>softmax_consistent\<close>: for every \<open>kB\<close> with \<open>n \<le> 2^kB\<close> and every \<open>kW\<close>, the Triton and the
  CUDA softmax models return the spec's row, column for column.\<close>

theorem softmax_consistent:
  assumes add_cm: "comm_monoid fadd fzero"
    and max_comm: "\<And>a b. fmax a b = fmax b a"
    and max_assoc: "\<And>a b c. fmax (fmax a b) c = fmax a (fmax b c)"
    and fits: "length xs \<le> 2 ^ kB"
  shows "softmaxTriton fadd fzero fsub fdiv fexp fmax kB xs
           = softmaxSpec fadd fzero fsub fdiv fexp fmax xs
       \<and> softmaxCuda fadd fzero fsub fdiv fexp fmax kW xs
           = softmaxSpec fadd fzero fsub fdiv fexp fmax xs"
proof -
  have max_cm: "comm_monoid (maxbot fmax) None"
    using max_comm max_assoc by (rule comm_monoid_maxbot)
  show ?thesis
    using fits
    by (simp add: softmaxTriton_def softmaxCuda_def softmaxSpec_def Let_def
        tile_reduce_eq_spec[OF max_cm] tile_reduce_eq_spec[OF add_cm]
        block_reduce_eq_spec[OF max_cm] block_reduce_eq_spec[OF add_cm])
qed

text \<open>\<^bold>\<open>layernorm_consistent\<close>: the same for layer normalization.\<close>

theorem layernorm_consistent:
  assumes add_cm: "comm_monoid fadd fzero"
    and fits: "length xs \<le> 2 ^ kB"
  shows "layernormTriton fadd fzero fsub fmul fdiv fsqrt fcast feps kB xs
           = layernormSpec fadd fzero fsub fmul fdiv fsqrt fcast feps xs
       \<and> layernormCuda fadd fzero fsub fmul fdiv fsqrt fcast feps kW xs
           = layernormSpec fadd fzero fsub fmul fdiv fsqrt fcast feps xs"
  using fits
  by (simp add: layernormTriton_def layernormCuda_def layernormSpec_def Let_def
      tile_reduce_eq_spec[OF add_cm] block_reduce_eq_spec[OF add_cm])


section \<open>Bridges between the README's model and the kernel text\<close>

text \<open>\<open>levels\<close> is the README's composition, widest level first; e.g. for \<open>W = 8\<close>:\<close>

lemma levels_three: "levels f 3 p = step f 1 (step f 2 (step f 4 p))"
  by (simp add: levels_def numeral_3_eq_3)

text \<open>The literal loop \<open>for (s = W >> 1; s > 0; s >>= 1)\<close> started at \<open>W = 2^k\<close> is \<open>levels\<close>.\<close>

lemma treeLoop_levels: "treeLoop f (2 ^ k div 2) p = levels f k p"
proof (induction k arbitrary: p)
  case 0
  show ?case by (simp add: levels_def)
next
  case (Suc k)
  have "treeLoop f (2 ^ Suc k div 2) p = treeLoop f (2 ^ k div 2) (step f (2 ^ k) p)" by simp
  also have "\<dots> = levels f k (step f (2 ^ k) p)" by (rule Suc.IH)
  also have "\<dots> = levels f (Suc k) p" by (simp add: levels_Suc)
  finally show ?case .
qed

lemma blockReduce_eq_loop:
  "blockReduce f e k xs = treeLoop f (2 ^ k div 2) (partial0 f e (2 ^ k) xs) 0"
  by (simp add: blockReduce_def treeLoop_levels)

text \<open>The tree needs \<open>W\<close> to be a power of two: with six threads the loop runs the levels
  \<open>s = 3, 1\<close> and never folds \<open>partial[2]\<close> or \<open>partial[5]\<close> into \<open>partial[0]\<close>.\<close>

lemma width_six_drops_partials:
  fixes p :: "nat \<Rightarrow> nat"
  defines "p \<equiv> \<lambda>t. if t < 6 then 1 else 0"
  shows "treeLoop (+) (6 div 2) p 0 = 4" and "spec (+) 0 (map p [0..<6]) = 6"
proof -
  show "treeLoop (+) (6 div 2) p 0 = 4" by (simp add: p_def step_def)
  show "spec (+) 0 (map p [0..<6]) = 6" by (simp add: p_def spec_def eval_nat_numeral)
qed

text \<open>For a non-empty row the max is finite, and the spec is the textbook softmax.\<close>

lemma row_max_finite:
  "spec (maxbot fmax) None (map Some (x # rest)) = Some (foldl fmax x rest)"
proof -
  have "foldl (maxbot fmax) (Some y) (map Some rest) = Some (foldl fmax y rest)" for y
    by (induction rest arbitrary: y) simp_all
  then show ?thesis by (simp add: spec_def)
qed

lemma softmaxSpec_textbook:
  "softmaxSpec fadd fzero fsub fdiv fexp fmax (x # rest) =
     (let M = foldl fmax x rest;
          S = spec fadd fzero (map (\<lambda>y. fexp (fsub y M)) (x # rest))
      in map (\<lambda>y. fdiv (fexp (fsub y M)) S) (x # rest))"
  by (simp only: softmaxSpec_def row_max_finite option.sel)

text \<open>A lane-wise map that sends the masked value to the masked value commutes with the tile.\<close>

lemma map_tile: "g e = e' \<Longrightarrow> map g (tile e B xs) = tile e' B (map g xs)"
  by (simp add: tile_def)

text \<open>\<open>_softmax_kernel\<close>, lines 89-93, literally: the loaded tile holds \<open>-inf\<close> (\<open>None\<close>) in
  its masked lanes, and the kernel sums \<open>exp (c - M)\<close> over every lane of it. With \<open>lane\<close> that
  per-lane computation on \<open>'f option\<close>, the kernel's sum is the model's as soon as
  \<open>exp(-inf - M) = 0\<close>: a premise of this lemma only (README: IEEE behaviour, evidenced by PBT).\<close>

lemma triton_sum_tile:
  assumes "lane None = fzero" and "\<And>x. lane (Some x) = fexp (fsub x M)"
  shows "tree fadd kB (map lane (tile None (2 ^ kB) (map Some xs)))
       = tileReduce fadd fzero kB (map (\<lambda>x. fexp (fsub x M)) xs)"
  using assms by (simp add: tileReduce_def map_tile comp_def)

text \<open>\<open>_layernorm_kernel\<close>, lines 105-106, literally: centre under the mask (masked lanes
  \<open>0.0\<close>), then square every lane of the tile, masked ones included, and sum. That is the
  model's sum as soon as \<open>0.0 * 0.0 = 0.0\<close>: a premise of this lemma only.\<close>

lemma triton_var_tile:
  assumes "fmul fzero fzero = fzero"
  shows "tree fadd kB (map (\<lambda>c. fmul c c) (tile fzero (2 ^ kB) (map (\<lambda>x. fsub x mean) xs)))
       = tileReduce fadd fzero kB (map (\<lambda>x. fmul (fsub x mean) (fsub x mean)) xs)"
  using assms by (simp add: tileReduce_def map_tile comp_def)


section \<open>Audit\<close>

text \<open>Each line prints the oracles a theorem depends on; an empty list means none.\<close>

thm_oracles strided_cover
thm_oracles tile_reduce_eq_spec
thm_oracles tile_reduce_drops_tail
thm_oracles block_reduce_eq_spec
thm_oracles softmax_consistent
thm_oracles layernorm_consistent

thm_oracles strided_simps sorted_strided blockReduce_eq_loop width_six_drops_partials
  softmaxSpec_textbook triton_sum_tile triton_var_tile

text \<open>The same check, made binding and written to the session's build log
  (\<open>$ISABELLE_HOME_USER/heaps/*/log/CrossSetup.gz\<close>): one line per theorem listing its oracles,
  and the build fails if any of the six depends on one.\<close>

ML \<open>
  val audited =
    ["strided_cover", "tile_reduce_eq_spec", "tile_reduce_drops_tail",
     "block_reduce_eq_spec", "softmax_consistent", "layernorm_consistent"];
  fun oracle_names name =
    Thm_Deps.all_oracles (Proof_Context.get_thms \<^context> name)
    |> map (fn ((oracle, _), _) => oracle);
  val _ = audited |> List.app (fn name =>
    let val names = oracle_names name in
      Output.physical_writeln ("thm_oracles " ^ name ^ ": [" ^ commas names ^ "]");
      if null names then () else error ("theorem " ^ name ^ " depends on: " ^ commas names)
    end);
\<close>

end
