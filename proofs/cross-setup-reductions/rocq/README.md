# Cross-setup reductions in Rocq

`CrossSetup.v` proves the six theorems of [`../README.md`](../README.md) in **Rocq 9.3, Stdlib
only**. It uses no `Admitted`, no `admit` and no top-level assumptions. Every abstract operation
and every law is a section variable (`Context`), so once the sections close, each theorem is
universally quantified over them. `Print Assumptions` reports `Closed under the global context`
for all six.

## What is proved

The reduction theorems hold over any commutative monoid `(M, op, e)`. The
`is_comm_monoid op e` argument holds the monoid laws.

| theorem | statement in the file | in words |
|---|---|---|
| `strided_cover` | `1 <= W -> Permutation (flat_map (strided W n) (seq 0 W)) (seq 0 n)` | Across threads `0 .. W-1`, the CUDA loop `for (c = t; c < n; c += W)` visits every column of `[0, n)` exactly once. This holds for every `W >= 1`, not only powers of two. |
| `tile_reduce_eq_spec` | `length xs <= 2 ^ k -> tileReduce op e k xs = spec op e xs` | The Triton tile model gives the sequential fold whenever the row fits in the tile. |
| `tile_reduce_drops_tail` | `tileReduce op e k xs = spec op e (firstn (2 ^ k) xs)` | For every row, the tile model reduces only the first `2^k` columns, so a wider row loses its tail. |
| `block_reduce_eq_spec` | `blockReduce op e k xs = spec op e xs` | The CUDA block model gives the sequential fold for every `k` and every row, with no relation between `W = 2^k` and `n`. |
| `softmax_consistent` | `1 <= length xs -> length xs <= 2 ^ kB -> softmaxTriton kB xs = softmaxSpec xs /\ softmaxCuda kW xs = softmaxSpec xs` | The Triton and CUDA softmax models both equal the spec, as whole output lists. |
| `layernorm_consistent` | `length xs <= 2 ^ kB -> layernormTriton kB xs = layernormSpec xs /\ layernormCuda kW xs = layernormSpec xs` | The same holds for layer normalization. |

The last two quantify over the carrier `F` and the operations `add`, `zero`, `sub`, `mul`, `div`,
`exp`, `sqrt`, `cast : nat -> F`, `eps` and `fmax` (each theorem over the ones it uses). They
assume only three things:

- `(F, add, zero)` is a commutative monoid.
- `fmax` is commutative.
- `fmax` is associative.

Nothing is assumed about `exp`, `sqrt`, `sub`, `mul`, `div`, `cast` or `eps`.

Helper results that are worth reading:

- `strided_loop_fuel`: once the fuel is at least `n - c`, adding more fuel never changes the loop's output.
- `step_level`: one tree level preserves the fold of the live partials.
- `levels_spec`: the whole tree computes the fold of all `2^k` partials.
- `spec_partials`: the fold of the partials equals the fold of the row.
- `fold_perm`: a fold over a commutative monoid is invariant under permutation.
- `omax_cm`: max extended with a bottom element is a commutative monoid.
- `softmax_row_max_finite`: for `n >= 1` the row max is finite.
- `triton_var_tile_literal`: see "Deviations" below.

## Check it

```bash
cd proofs/cross-setup-reductions/rocq && rocq compile CrossSetup.v
```

The command exits with code 0 and prints `Closed under the global context` six times, once for
each `Print Assumptions` line, in the order of the table above. It prints no warnings.

**Compile time:** about 1.3 s wall clock. Four runs measured 1.22 to 1.28 s with
`/usr/bin/time -p`, the slowest being the run from a clean directory. All used Rocq 9.3.0 from
Homebrew (OCaml 5.5.1) on Apple Silicon macOS.

**Build artefacts:** the compile writes four files next to the source. They are not committed and
should be git-ignored:

- `CrossSetup.vo`
- `CrossSetup.glob`
- `.CrossSetup.aux`
- `.lia.cache` (written by `lia`)

It writes no `.vos` or `.vok` files.

## How the definitions model the kernels

The header comment of `CrossSetup.v` maps each definition to the kernel lines it models, and
explains why the functional tree equals the in-place loop. In summary:

| definition | models | kernel lines |
|---|---|---|
| `tile e B xs` | `offs = tl.arange(0, BLOCK)`, `mask = offs < n_cols`, `tl.load(..., mask, other=<identity>)` | `ladder.py` 86-89, 99-101 |
| `tree k l` | `tl.max` / `tl.sum`, as the README's balanced halving tree (`firstn` / `skipn` at `2^(k-1)`) | `ladder.py` 90, 93, 102, 106 |
| `tileReduce k xs` | `tree k (tile (2^k) xs)`, with `BLOCK = 2^k` from `block_for` | `ladder.py` 47-71 |
| `strided_loop`, `strided` | `for (int c = tid; c < n_cols; c += width)`, transcribed as written | `softmax.cu` 34, 51, 64; `layernorm.cu` 43, 49, 56 |
| `partial0 W xs t` | phase 1: start from the identity, `acc = acc ⊕ x[c]` in loop order, `partial[tid] = acc` | `softmax.cu` 33-37, 50-54; `layernorm.cu` 42-45, 48-52, 18 |
| `step s p t` | `if (tid < s) partial[tid] = partial[tid] ⊕ partial[tid + s]` | `softmax.cu` 40-42, 57-59; `layernorm.cu` 21-23 |
| `levels k p` | `for (s = width >> 1; s > 0; s >>= 1)`: `s = 2^(k-1)` first, `1` last, no level when `k = 0` | `softmax.cu` 39, 56; `layernorm.cu` 20 |
| `blockReduce k xs` | the tree over `partial0 (2^k) xs`, then `partial[0]` | `softmax.cu` 45, 62; `layernorm.cu` 26 |
| `softmaxTriton` / `softmaxCuda` | row max, then the sum of `exp (x - M)`, then `exp (x - M) / S` per column | `ladder.py` 86-93; `softmax.cu` 31-66 |
| `layernormTriton` / `layernormCuda` | mean, then the variance of the centred row, then `(x - mean) / sqrt (var + eps)` | `ladder.py` 99-107; `layernorm.cu` 41-57 |

Each model describes one row. The kernels run one program or block per row, and rows are
independent.

### Modelling choices specific to Rocq

- **The commutative monoid.** `is_comm_monoid op e` is a `Prop` record with three fields:
  associativity, commutativity and left identity. The right identity is derived from these. In
  `Section Reductions` it is the section variable `HM`, so every reduction theorem takes it as an
  explicit argument. Softmax and layer normalization instantiate it twice: once with
  `add_cm : is_comm_monoid add zero`, and once with `omax_cm`.
- **The bottom element for max (`-inf`).** The max is taken over `option F`, where `None` stands
  for the bottom element ⊥. `omax` extends `fmax` with `None` as its identity. `omax_cm` proves
  that this is a commutative monoid from `fmax` commutativity and associativity alone.
  `unbot` reads the row max back into `F`. Its `None` branch returns `zero`, but that branch is
  unreachable for `n >= 1`, as `softmax_row_max_finite` shows. The Triton max tile
  pads with `None`, which models `other=-float("inf")`. The CUDA phase-1 fold starts from `None`,
  which models `m = negative_infinity()`.
- **The in-place tree.** `p : nat -> M` is the shared array, and `step s p` is one whole level
  as a pure function. This equals the in-place loop body for three reasons:
  - At stride `s`, thread `t < s` writes only `partial[t]`, an index below `s`.
  - The thread reads its own cell before writing it, and reads `partial[t + s]`, an index in
    `[s, 2s)` that no thread writes at this level. So no read sees a write from the same level.
  - The `__syncthreads()` after each level orders the levels.

  Reads stay below `2s <= W`, so the values of `p` at or above `W` are never consulted.
- **The strided loop** is a `Fixpoint` with fuel `n`, because Rocq needs a decreasing argument.
  The fuel is never what stops the loop: `strided_loop_fuel` proves that once the fuel is at
  least `n - c`, adding more fuel changes nothing. For `W >= 1` the loop runs at most `n - t`
  times.
- **The output loops**, Triton's masked store and CUDA's third strided loop, are modelled as a
  `map` over the row. On the CUDA side, `strided_cover` is the fact that justifies this: every
  column is written by exactly one thread, once.

## Deviations from the README statement

All of these either strengthen the statement or make an assumption explicit. None weakens it.

1. **`layernorm_consistent` has no `n >= 1` hypothesis.** The README says "the same", but layer
   normalization has no max that needs to be finite, and `/ cast n` is the same term on both
   sides. The theorem as proved implies the version with `n >= 1`.
2. **`softmax_consistent` keeps `1 <= length xs`, as the README states it, but the proof does
   not use it.** The `None` branch of `unbot` gives the same `zero` on all three sides, and for
   `n = 0` all three outputs are `[]`. The hypothesis is there only so that `M` means the finite
   row max.
3. **`fmax` idempotence is not assumed.** The README allows it, but the monoid on `option F` needs
   only commutativity and associativity.
4. **The conclusions are equalities of whole output lists.** This is stronger than agreement on
   every column, because it also says the outputs have the same length.
5. **Masked lanes of Triton's variance tile.** Following the README, the model puts the identity
   `0` in the masked lanes of the tile of squares. The kernel actually computes
   `centered * centered` with `centered = tl.where(mask, x - mean, 0.0)`, so a masked lane holds
   `0.0 * 0.0`. `triton_var_tile_literal` proves that the kernel's lane-wise squared tile equals
   the model's tile whenever `mul zero zero = zero`. That equation is a premise of this lemma
   only; it is not an assumption of any of the six theorems. It is IEEE behaviour of the same
   kind as `exp(-inf) = 0`, and the README's table does not list it yet.
