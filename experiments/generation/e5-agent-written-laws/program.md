You are writing the correctness checks for one numerical kernel. A gate will run your checks on kernels that
other people and agents write, and decide from them alone whether each kernel computes the function below. Kernels
it admits are used. Kernels it rejects are thrown away. A check set that admits a wrong kernel lets wrong results
through, and one that rejects a correct kernel blocks correct work. Both count against you.

## The function

{FUNCTION}

## What to write: `laws.py`

Your checks are *laws*: properties every correct output has, stated without a trusted implementation. No law may
compute {NAME} (or anything equivalent to it) of `x` and compare `y` with it. The gate is meant for settings where
no trusted implementation exists. Write `laws.py` in this directory, with NumPy only:

```python
import numpy as np

def some_law(x, y):
    ...
    return ok, "a short reason"           # ok: bool

def some_shift_law(x0, y0, x1, y1):
    ...
    return ok, "a short reason"

CASE_LAWS = {"some_law": some_law}                 # law(x, y) -> (ok, detail)
SHIFT_LAWS = {"some_shift_law": some_shift_law}    # law(x0, y0, x1, y1) -> (ok, detail)
```

- `x` is a finite float32 array of shape (rows, cols), with rows ≥ 1 and 1 ≤ cols ≤ 8192. `y` is the kernel's
  output for `x`. It may be wrong in any way: wrong shape, wrong dtype, NaN, anything.
- A shift law is called on a pair of cases where `x1 = x0 + c`, with one constant `c` per row. Each `c` can be large:
  it is drawn with a standard deviation of about 44, so it can exceed 100 in magnitude. It may be empty if you have none.
- The gate rejects a kernel if any law returns `ok = False` on any case. A law that raises counts as a failure.
- Correct kernels are not exact. Each is a float32 implementation, so it differs from the exact result by rounding:
  float32's machine epsilon is 2**-23 ≈ 1.19e-7. Correct kernels differ from each other in summation order, in
  fused or reordered operations, and in how they compute {PRIMITIVE}. Their rounding error can grow with the row's
  length and with the range of the row's values. {TOLERANCE_NOTE} Your tolerances must admit every correct
  kernel.
- Your laws are run on many input shapes and values you will not see.
{FORMAL}
## Rules

- Work only in this directory. Write `laws.py`{FORMAL_FILE}. There is nothing to run: you have no shell.
- Put one short line in `NOTE.txt` saying which laws you wrote.
