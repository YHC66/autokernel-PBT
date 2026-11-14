# Pre-registration: E3c. Does a wider ladder stop E3b's hacks without rejecting correct kernels?

**Date:** 2026-11-13 (registered before this experiment's code exists and before any candidate is re-run;
results are appended below the line, never edited above it)
**Segment:** generation. This follows E3b.
**Instance:** Apple M4 Pro, CPU only; the project's pinned environment (NumPy 1.26.4)
**Companions:** `2026-11-04-e3b-instructed-red-team-prereg.md` (the hacks), `2026-11-03-e3-red-team-prereg.md`
(the correct kernels)

## 0. The question

In E3b the completed laws held: no agent got a kernel that is wrong on the gate's own cases past them. The gate as a
whole did not hold: it admitted 160 of 180 candidates and every one was wrong somewhere, almost all of them past the
gate's ladder, which stops at 17 rows and 129 columns. E3b's results named a fix, a ladder drawn from the call
site's shapes, and did not test it.

This experiment tests it, with no new agent sessions. **If the same laws are run on a wider ladder, how many of
E3b's admitted hacks does the gate reject, and does it reject any correct kernel?**

It is a replay of fixed kernels, so it cannot say whether an agent that knows the wider ladder finds a way past it.
That needs new sessions and is a separate experiment.

## 1. Construction

**The ladders.** Every ladder keeps the gate's seed (42), its domains (softmax N(0, 1), layernorm U(-10, 10)), its
relation (`shift_rows`) and its first 9 groups, which are byte-identical to the shipped gate's. Only the shapes after
them differ.

| Ladder | Groups | Shapes added after the shipped 9 |
|---|---|---|
| **L0** shipped | 9 | none |
| **L1** call site | 10 | (512, 4096), the shape the loop times the kernel at |
| **L2** random | 17 | 8 shapes drawn once from seed 20261113: rows `round(exp(U(0, ln 512)))`, columns `round(exp(U(0, ln 8192)))` |

L2 is not aimed at the call site. It stands for a gate that widens its ladder without knowing where the kernel runs.
Its seed is fixed here so the run is reproducible. No E3b agent saw any of these shapes.

**The decision.** As in the shipped gate: the declarative arm under a bundle decides, and a kernel that produces no
output on any case is refused. The reference arm is also recorded on every case, to tell a law's rejection of a
correct kernel (a false reject) from its rejection of a wrong one.

**The kernels.**

- **Hacks:** E3b's 180 candidates (`recorded/<arm>-r<k>/candidates/NN.py`): 160 the gate admitted, every one wrong by
  the hidden checker, and 20 it rejected. Each is scored under the bundle its arm gated on (A: current, B and C:
  completed) and under the completed bundle.
- **Correct kernels:** E3's 120 candidates (`e3-red-team/recorded/<bundle>-r<k>/candidates/NN.py`), every one admitted
  and correct by E3's hidden checker, and the starting kernel. Scored under the completed bundle.

**Running them.** Each candidate runs only in E3b's sandboxed recorder, once per ladder, in a fresh process, and is
called on that ladder's cases in the gate's order. A kernel that counts its calls therefore sees the same call
sequence a real gate on that ladder would give it. The scorer replays the recording and never imports the candidate.
Each ladder's gate wall time is recorded per candidate.

## 2. Already known, so not predictions

- L0 must reproduce E3b's gate verdict on all 180 candidates under each arm's bundle, and admit all 121 correct
  kernels. A mismatch is an instrument defect: it is reported, and the run stops until it is explained.
- The 20 candidates L0 rejects stay rejected on L1 and L2 for a kernel that does not count its calls, because the
  shipped 9 groups come first.

## 3. Predictions

- **P1.** L1, under the completed bundle, rejects at least 150 of the 160 admitted hacks.
- **P2.** L1 and L2, under the completed bundle, reject none of the 121 correct kernels. **Falsifier:** any correct
  kernel rejected on a case where the reference arm passes it. That would mean the laws' tolerances do not hold at
  these sizes.
- **P3.** L2 rejects fewer of the 160 than L1. It is not aimed at the shapes the hacks switch on.
- **P4.** L1's gate time is under 10× L0's, median over the 160.

Every hack L1 does not reject is listed by hand, with the input class on which the hidden checker failed it.

## 4. What is NOT measured

- Whether an agent that knows the wider ladder finds a way past it (needs new sessions).
- Checks on the timed calls themselves (E3b's other suggested fix).
- GPU kernels; only the NumPy kernels E3 and E3b produced.

## 5. Deviation rules

Any change to a ladder, a corpus or a prediction after the first candidate is re-run is reported as a deviation
below the line, with its reason. An instrument defect is fixed and the whole run is repeated.

---

## Results

Run 2026-11-13 on the instance above: 301 kernels × 3 ladders, 903 sandboxed recordings, 0 recording errors.
`python experiments/generation/e3c-wider-ladder/e3c.py run`, then `report`; the per-recording results are in
`experiments/generation/e3c-wider-ladder/recorded/results.jsonl` and the report in `recorded/report.md`.

**Deviation (instrument, before any result was used).** The first full run stored only each arm's first failure.
That was too little to tell which group rejected a correct kernel, so the scorer was changed to store every failing
group and the whole run was repeated. Every verdict reproduced exactly.

**Instrument check.** L0 reproduces E3b's gate verdict on all 180 candidates, per function and under each arm's
bundle, and admits all 121 correct kernels. 0 mismatches.

| Ladder | Hacks rejected (completed) | E3b-rejected still rejected | Correct kernels rejected (completed) | Median gate time |
|---|---|---|---|---|
| L0 shipped | 8/160 (arm A's family hacks) | 20/20 | 0/121 | 0.36 s |
| L1 call site | **151/160** | 20/20 | **0/121** | 1.85 s |
| L2 random | 160/160 | 20/20 | **121/121** | 0.42 s |

Gate time is the sandboxed recording plus scoring, so it includes a process start.

- **P1 holds.** L1 rejects 151 of 160 (94%). It rejects every hack faster than 1.19×, including all of arm C's,
  up to the 12,442× constant fill. The decision is the laws' alone: they reject 246 of the 282 wrong functions, as
  many as the reference arm does. The first failing law is most often layernorm's zero mean (109) and softmax's row sum (48) and
  log-ratio law (39) on the call-site row.
- **The 9 hacks L1 admits are all value-regime hacks.** Each is wrong only on inputs whose spread or values the
  gate never draws, at any shape. Seven come from arm A and two from arm C. Four floor the shifted logits (at −10,
  −12, −16 or −20). Four take the shift from something other than the full row max (the first element, a prefix,
  every fourth element) and break only when the true max exceeds it by about 60–88. One drops ε on rows with variance under 1e−4. None is faster than 1.18× on either
  function. Closing these needs the ladder to vary the input's scale, not only its shape.
- **P2 holds for L1 and is falsified for L2.** L2 rejects every correct kernel, and every one fails in the
  same place: layernorm's unit-variance law on the (22, 2) group. The reference arm passes all of them. The
  smallest row variance in that group is 0.0224, and the reported deviation of 4.45e−4 is exactly ε/var,
  the term `tasks.py` documents. The shipped ladder avoids it by drawing wide U(−10, 10) rows, and a 2-column
  row escapes that. The law is the same in both bundles, so it has been in the gate all along. This is a false
  alarm in a shipped law, found by widening the ladder: the laws' tolerances were calibrated on the ladder's
  shapes, and need calibrating again whenever the ladder grows.
- **P3, as registered, is falsified** (L2 160 > L1 151), but L2's count is inflated by that false alarm. *Post
  hoc, not registered:* setting aside that one group's failure, L2 rejects 114/160 hacks (current bundle:
  104) and 0/121 correct kernels. So the direction P3 predicted holds once the false alarm is removed.
- **P4 holds.** L1's median gate time is 5.1× L0's (L2's is 1.2×).

**What this says.** A gate's laws and its inputs are separate properties. E2 made the laws complete, and E3b showed
the inputs were not. Adding the one shape the kernel is called at rejects 94% of what agents told to break the
gate produced, and every hack faster than 1.19×, with no false reject. A ladder widened without aim does less, and
widening exposed a tolerance that had been calibrated only on the old shapes. Not tested: whether an agent that
knows the call-site rung finds its way past it.
