# Cross-setup reductions in Isabelle/HOL

`CrossSetup.thy` proves the six theorems of [`../README.md`](../README.md) in **Isabelle2025-2,
HOL only**: `Main` plus `HOL-Library.Multiset` (for `mset`), which ships with Isabelle. It uses
no AFP entry. The theory adds nothing to HOL except definitions. Every abstract operation is a
variable of the statement, and every law it needs is an explicit premise of the theorem that
uses it. All six build in the default strict mode, in which an unfinished proof is an error.

## What is proved

Here `comm_monoid f e` is HOL's own locale predicate from `Groups.thy`. It says that `f` is
associative and commutative and that `e` is a right identity. The reduction theorems hold for
every such `(f, e)`, and `spec f e xs = foldl f e xs`.

| theorem | exported statement | in words |
|---|---|---|
| `strided_cover` | `1 ≤ W ⟹ mset (concat (map (strided W n) [0..<W])) = mset [0..<n]` | The CUDA loop `for (c = t; c < n; c += W)`, run by threads `0 .. W-1`, visits every column of `[0, n)` exactly once. This holds for every `W ≥ 1`, not only powers of two. Equal `mset`s is Isabelle's notion of a permutation; HOL-Combinatorics' `xs <~~> ys` is an input abbreviation for it. |
| `tile_reduce_eq_spec` | `comm_monoid f e ⟹ length xs ≤ 2 ^ k ⟹ tileReduce f e k xs = spec f e xs` | When the row fits in the tile, the Triton tile model computes the spec. |
| `tile_reduce_drops_tail` | `comm_monoid f e ⟹ tileReduce f e k xs = spec f e (take (2 ^ k) xs)` | For every row, the tile model reduces only the first `2^k` columns. A wider row loses its tail, which is the tile-tail defect measured on hardware. |
| `block_reduce_eq_spec` | `comm_monoid f e ⟹ blockReduce f e k xs = spec f e xs` | The CUDA block model computes the spec for every `k` and every row. `W = 2^k` and `n` need no relation. |
| `softmax_consistent` | `comm_monoid fadd fzero ⟹ (⋀a b. fmax a b = fmax b a) ⟹ (⋀a b c. fmax (fmax a b) c = fmax a (fmax b c)) ⟹ length xs ≤ 2 ^ kB ⟹ softmaxTriton … kB xs = softmaxSpec … xs ∧ softmaxCuda … kW xs = softmaxSpec … xs` | The Triton and CUDA softmax models both return the spec's row, for every `kB` with `n ≤ 2^kB` and every `kW`. Here `…` stands for `fadd fzero fsub fdiv fexp fmax`. |
| `layernorm_consistent` | `comm_monoid fadd fzero ⟹ length xs ≤ 2 ^ kB ⟹ layernormTriton … kB xs = layernormSpec … xs ∧ layernormCuda … kW xs = layernormSpec … xs` | The same holds for layer normalization. Here `…` stands for `fadd fzero fsub fmul fdiv fsqrt fcast feps`. |

The last two hold over any type `'f`. The operations `fadd`, `fzero`, `fsub`, `fmul`, `fdiv`,
`fexp`, `fsqrt`, `fcast :: nat ⇒ 'f`, `feps` and `fmax` are universally quantified variables.
Only three things are assumed: `(fadd, fzero)` is a commutative monoid, and `fmax` is
commutative and associative. Nothing is assumed about `fexp`, `fsqrt`, `fsub`, `fmul`, `fdiv`,
`fcast` or `feps`.

The file also proves these lemmas, which tie the README's model to the kernel text:

| lemma | says |
|---|---|
| `strided_simps` | for `W ≥ 1`, `strided W n c = (if c < n then c # strided W n (c + W) else [])`: the loop's own recursion |
| `sorted_strided` | for `W ≥ 1`, each thread's columns are strictly ascending (the README's "ascending") |
| `blockReduce_eq_loop` | `blockReduce` equals `treeLoop`, the literal `for (s = width >> 1; s > 0; s >>= 1)` loop started at `W = 2^k` |
| `levels_three` | `levels f 3 p = step f 1 (step f 2 (step f 4 p))`: the composition order, widest level first |
| `width_six_drops_partials` | with six threads the literal loop runs `s = 3, 1` and returns `4` on six partials of `1`, whose sum is `6` |
| `softmaxSpec_textbook` | on a non-empty row, `softmaxSpec` is `exp (x - M) / S` at every column, with `M = foldl fmax x rest` the finite row max |
| `triton_sum_tile` | the kernel's literal sum tile, `exp (c - M)` over a tile whose masked lanes hold `-inf`, reduces to the model's sum, given that the masked lane's value is `fzero` |
| `triton_var_tile` | the kernel's literal variance tile (lines 105–106: centre under the mask, then square every lane, masked lanes included) reduces to the model's sum, given `fmul fzero fzero = fzero` |

## Check it

```bash
cd proofs/cross-setup-reductions/isabelle && isabelle build -d . -v CrossSetup
```

The command exits with code 0 and prints `Finished CrossSetup`. It writes nothing into this
directory. Heaps, logs and the session database go under `~/.isabelle/Isabelle2025-2/`.

- **Build time.** Six clean builds (`isabelle build -c -d . -v CrossSetup`) on 2026-10-13 took
  11.1, 11.8, 12.2, 15.5, 12.3 and 11.8 s of wall-clock time (`/usr/bin/time -p`), so about
  12 s. The machine's load average was 14–18 at the time. Isabelle reports
  `Finished CrossSetup (0:00:06 elapsed time)`, or 0:00:08 on the slowest run. The prebuilt HOL
  heap is reused, but `HOL-Library.Multiset` is loaded from source on every build. That takes
  about 13.7 s of CPU time across threads, against about 2 s for `CrossSetup` itself. Machine:
  Apple Silicon macOS, Poly/ML 5.9.2 arm64_32.
- **Repeat builds.** `isabelle build` skips a session that is already up to date and then prints
  no `Finished CrossSetup` line. Pass `-c` to force a rebuild. `tests/proofs/test_cross_setup_isabelle.py`
  builds a fresh copy in a temporary directory, which always rebuilds. It forces the strict mode
  on the command line and passed on 2026-10-13.

### The oracle audit

At the end of the theory:

- One `thm_oracles` command per theorem prints the oracles that theorem depends on. All six print
  an empty list. Batch builds do not echo this output, but Isabelle/jEdit shows it.
- An `ML` block makes the same check binding. It writes one line per theorem to the session log,
  `~/.isabelle/Isabelle2025-2/heaps/polyml-5.9.2_arm64_32-darwin/log/CrossSetup.gz`, and fails the
  build if any of the six depends on an oracle. A passing build logs:

  ```
  thm_oracles strided_cover: []
  thm_oracles tile_reduce_eq_spec: []
  thm_oracles tile_reduce_drops_tail: []
  thm_oracles block_reduce_eq_spec: []
  thm_oracles softmax_consistent: []
  thm_oracles layernorm_consistent: []
  ```

I tested the check in scratch copies:

- In the strict default mode, a skipped proof is rejected where it stands ("Cheating requires …
  mode"), so the build fails before the audit runs.
- In the permissive mode, all six theorems depend on `Pure.skip_proof`, whether or not the file
  contains a skipped proof. HOL's definitional packages (`fun`, `function`) and the
  `HOL-Library` theories loaded into this session then skip their internal proofs, so the audit
  fails the build.

The audit therefore passes only for a strict build in which none of the six theorems rests on
any oracle.

### The proof is not vacuous

I mutated the models in scratch copies outside the repo, one mutation per copy, and confirmed
that each build fails. The first lemma to fail is given in parentheses.

1. The stride `c + W` changed to `c + 2 * W`, in the definition only (`strided_simps`).
2. The same stride change made consistently in every statement of the loop equation, so that only
   the arithmetic can fail (`set_strided`).
3. The Triton mask `i < length xs` changed to `i ≤ length xs` (`tile_eq`).
4. The tree read `p (t + s)` changed to `p (t + s + 1)` (`sum_step`).
5. The level order reversed, so that `s = 1` runs first: `[0..<k]` changed to `rev [0..<k]`
   (`levels_Suc`).
6. The phase-1 start column `t` changed to `t + 1`, consistently (`spec_partials`).
7. The Triton halving split `drop (2 ^ k)` changed to `drop (2 ^ k + 1)` (`tree_eq_spec`).
8. `softmaxCuda`'s denominator uses `fsub row_max x` instead of `fsub x row_max`
   (`softmax_consistent`).
9. The hypothesis `n ≤ 2^k` weakened to `n ≤ 2^k + 1` (`tile_reduce_eq_spec`).
10. A width that is not a power of two: `2^k + 1` threads with `k` levels (`block_reduce_eq_spec`).

## How the definitions model the kernels

The header comment of `CrossSetup.thy` maps every definition to kernel lines. In the table,
S = `kernels/cuda/softmax.cu`, L = `kernels/cuda/layernorm.cu` and T = `kernels/triton/ladder.py`.

| definition | kernel text | lines |
|---|---|---|
| `tile e B xs` | `offs = tl.arange(0, BLOCK)`, `mask = offs < n_cols`, `tl.load(..., mask, other=<identity>)`: lane `i` is `xs ! i` if `i < n`, else `e` | T 86–89, 99–101 |
| `tree f k l` | `tl.max` / `tl.sum` over the tile, as the README's balanced halving tree (`take` / `drop` at `2^(k-1)`) | T 90, 93, 102, 106 |
| `tileReduce f e k xs` | `tree k (tile e (2^k) xs)`, with `BLOCK = 2^k` from `block_for` (hence `n ≤ 2^kB`) | T 47–71 |
| `strided W n c` | `for (int c = tid; c < n_cols; c += width)`, as its recursion equation | S 34, 51, 64; L 43, 49, 56 |
| `partial0 f e W xs t` | phase 1: start from the identity, `acc = acc ⊕ x[c]` in loop order, `partial[tid] = acc` | S 33–37, 50–54; L 42–45, 48–52, 18 |
| `step f s p t` | `if (tid < s) partial[tid] = partial[tid] ⊕ partial[tid + s]` | S 40–42, 57–59; L 21–23 |
| `levels f k` | `step 1 ∘ step 2 ∘ … ∘ step (2^(k-1))`, literally (`foldr (∘)` over that list) | S 39, 56; L 20 |
| `treeLoop f s p` | `for (int s = width >> 1; s > 0; s >>= 1)`, literally | S 39, 56; L 20 |
| `blockReduce f e k xs` | the levels applied to the `W = 2^k` partials, read at `partial[0]`; `W` is a power of two by the launch contract (`threads_for`, `kernels/cuda/ladder.py` 44–52) | S 14, 45, 62; L 11, 26 |
| `maxbot fmax` | `max` on `'f option`, with `None = -inf` as identity | S 19–21, 33; T 89 |
| `softmaxSpec` / `softmaxTriton` / `softmaxCuda` | row max, then `S = Σ exp (x - M)`, then `exp (x - M) / S` per column | T 84–93; S 31–66 |
| `layernormSpec` / `layernormTriton` / `layernormCuda` | mean, then the variance of the centred row, then `(x - mean) / sqrt (var + eps)` | T 97–107; L 41–58 |

Each model describes one row; the kernels run one program or block per row, and rows are
independent. The output stores (Triton's masked `tl.store`, CUDA's third strided loop) are a
`map` over the row: every column is written exactly once (the mask; `strided_cover`), and the
value written depends only on `x[c]` and on block-uniform scalars.

**The in-place tree.** At level `s`, thread `t < s` runs
`partial[t] = partial[t] ⊕ partial[t + s]`, and threads `t ≥ s` do nothing. Every write lands at
an index `< s`, and only thread `t` writes index `t`. Every read is at `t`, by thread `t` itself
before its own write, or at `t + s`, an index in `[s, 2s)` that no thread writes at this level.
So no read sees a write from the same level, whatever the interleaving, and the level is the
pure function `step f s p`. The `__syncthreads()` that closes each level (S 43, 60; L 24)
hands exactly that array to the next level. The one after the phase-1 store (S 38, 55; L 19)
orders phase 1 before the tree. The model is this barrier-enforced schedule. Concurrency itself
is out of scope, as `../README.md` says. Reads stay below `2s ≤ W`, so the values of `p` at
indices `≥ W` never matter.

## Isabelle-specific modelling choices

- **The commutative monoid.** The six theorems take HOL's locale predicate `comm_monoid f e` as
  an explicit premise. Its laws are `assoc`, `commute` and `comm_neutral`, and the left identity
  is derived. The supporting lemmas live in `context comm_monoid` (parameters `f`, written `❙*`,
  and `z`, written `❙1`). Each theorem's proof runs `interpret comm_monoid f e`. The same
  locale is instantiated twice by the kernels: `(fadd, fzero)` from the premise, and
  `(maxbot fmax, None)` by `comm_monoid_maxbot`. Permutation invariance (`spec_perm`) comes from
  HOL-Library's `fold_multiset_equiv`. `spec_foldr` relates `spec` to HOL's list fold
  `foldr f xs e`, which is `comm_monoid_list.F` and so `sum_list` for `+`.
- **The bottom element for max.** `-inf` is `None` in `'f option`, and `maxbot fmax` extends
  `fmax` with `None` as identity. Commutativity and associativity of `fmax` are enough for the
  monoid laws. The row max is read back into `'f` with HOL's `the`. On the empty row it is
  `the None`, an unspecified value of `'f`, but then no column is written, as with both kernels
  when `n_cols = 0`. So no `n ≥ 1` is needed. For `n ≥ 1`, `row_max_finite` and
  `softmaxSpec_textbook` show that `M` is the finite row max.
- **The in-place tree.** The shared array is a function `nat ⇒ 'm`, and a level is `step f s`, a
  pure function, for the reason above. `levels` is the README's composition, written literally
  as `foldr (∘) (map (λj. step f (2 ^ j)) [0..<k]) id`. The literal C loop `treeLoop` is a
  `function` that terminates on the measure `s`. Its recursion equation is replaced in the
  simplifier by two cases, `s = 0` and `0 < s`. Unfolding it under an undecided test would make
  the simplifier diverge.
- **The strided loop** is defined by its own recursion equation, with no fuel. It is a partial
  `function (domintros)`: for `W = 0` and `c < n` the C loop does not terminate, and the
  definition leaves `strided 0 n c` unspecified there. `strided_terminates` proves that the
  recursion terminates for every `W ≥ 1`, which is all that `strided_cover` and the block model
  use, since `2^k ≥ 1`. Lean and Rocq give the same loop fuel `n` instead.
- **`tree f 0 l = hd l`.** On the one-element lists it is applied to, this is `tree [a] = a`.
- **Names.** The arithmetic variables are prefixed with `f` (`fadd`, `fdiv`, `fexp`, …) so that
  they do not clash with HOL constants such as `div` and `max`. `'a binop` abbreviates
  `'a ⇒ 'a ⇒ 'a`. The models are top-level definitions with explicit parameters, so each global
  constant's arguments are visible in the theorem statements.

## Deviations from `../README.md`'s statement

None of these weakens the statement.

1. **`softmax_consistent` has no `n ≥ 1` hypothesis.** All three models read the max back with
   `the`, so on the empty row all three return `[]`. The Lean proof drops the hypothesis too.
   The Rocq proof keeps it but does not use it. `softmaxSpec_textbook` covers what the
   hypothesis was for: with `n ≥ 1`, `M` is the finite row max.
2. **`max` need not be idempotent.** Only commutativity and associativity are premises.
3. **Masked lanes.** As the README's table says, the Triton models put the identity in the
   masked lanes. For the softmax sum that is `0` directly. For the re-masked variance tile it is
   `0`, where the kernel computes `0.0 * 0.0`. The two IEEE facts are premises of the side
   lemmas `triton_sum_tile` (`exp(-inf - M) = 0`) and `triton_var_tile`
   (`fmul fzero fzero = fzero`) only. Neither is a premise of the six theorems.
4. **The conclusions are equalities of whole output lists.** That is stronger than agreement on
   every column, because it also fixes the length.
5. **`strided_cover` is stated with `mset`.** It is Isabelle's permutation relation, and
   HOL-Combinatorics' `perm` abbreviates it. `strided W n t` is unspecified for `W = 0`, which
   the README's `W ≥ 1` excludes.
6. **The reduction theorems carry `comm_monoid f e` as a premise.** The README fixes the monoid
   in the surrounding text. Lean passes it as a structure argument and Rocq as a section
   variable.

## What the proof does not show

- **Floating point.** Rounding, `expf` against Triton's `exp`, and IEEE special values are left
  to PBT, as `../README.md`'s table says.
- **`W` must be a power of two.** The kernel genuinely needs this; it is not a convenience of the
  proof. `width_six_drops_partials` checks the counterexample: with `W = 6` the loop never folds
  `partial[2]` or `partial[5]` into `partial[0]`. Mutation 10 shows that the block theorem then
  fails. `threads_for` guarantees a power of two in `[32, 1024]`.
- **Unbounded indices.** Indices are `nat`. The kernels use `int c`, which overflows only when
  `n_cols > INT_MAX - width`.
- **Triton's reduction order.** The model is the README's halving tree. In exact arithmetic every
  bracketing gives `spec` (`spec_perm` with associativity), so the consistency claim does not
  depend on it. Rounding does, and that is PBT's job.
