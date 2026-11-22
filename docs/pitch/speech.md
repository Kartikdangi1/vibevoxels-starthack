# 3-minute pitch script — Feyer: Faster Physics

Timing: 357 spoken words ≈ 2:39 at ~135 words per minute, plus ~6 s for the demo run and ~10 s
to switch to the terminal and back ≈ 2:55 in total.
**Bold** marks the words to stress; *[cues]* are actions, not spoken. Slide numbers refer to
`pitch.pdf`.

| Part | Time | Slides |
|---|---|---|
| Problem | 0:00 – 0:30 | 1 → 3 |
| Solution | 0:30 – 1:45 | 4 → 8 |
| Demo | 1:45 – 2:25 | terminal, then 9 |
| Any setup, what's next | 2:25 – 3:00 | 10 → 13 |

---

## 0:00 – 0:30 · Problem  *(≈ 70 words)*

*[Slide 1]*

Gravitational-wave detectors are designed by optimisation: change 193 mirror and laser
parameters, simulate, follow the gradient, repeat. Every one of those steps is a full Differometor
evaluation.

*[Slide 3]*

And one evaluation took **35 seconds**. More than half of that was not physics at all: it was
**compiling**. The rest was dense linear algebra on a matrix that is **99.5 percent zeros**:
606 factorisations, mostly of nothing.

## 0:30 – 1:45 · Solution  *(≈ 175 words)*

*[Slide 4]*

So we stopped storing zeros. We assemble only the 2,500 nonzeros and factorise **once** per
frequency. That one factorisation serves all four solves, forward and adjoint: 606 factorisations
become 101.

*[Slide 5]*

Then one line of algebra. The noise term needed a second solve, but c-Hermitian times A-inverse is
just w-Hermitian. So the noise becomes **w-Hermitian N w**: a whole solve deleted. It holds for any
detector topology, not just this one.

*[Slide 6]*

Next, compile time. We run the physics on the CPU inside a single custom VJP, so XLA compiles two
tiny stubs instead of the whole graph. We did **not** re-derive the physics by hand: we transpose
JAX's own derivative program, so the gradients are JAX's.

*[Slide 7]*

The solver is our own C++ sparse LU, shipped prebuilt, so nothing compiles at install. The pivot
order from the first frequency is reused for the other hundred.

*[Slide 8]*

And the last 70 milliseconds came from XLA compiler flags and a CPU-tuned build. They are
**bit-identical**: not one output bit changes.

## 1:45 – 2:25 · Demo  *(≈ 80 words; the run takes ~6 s)*

*[Switch to terminal. Run:]* `DEMO_SSH="ssh -p <port> root@<host>" ./docs/pitch/demo.sh`

This is the official benchmark, unmodified, on the AMD EPYC we measured on, pinned to **one** CPU
core. It includes everything: imports, compilation, forward pass, sensitivity and the gradient.

*[Point at "Complete first-run total"]*

**Zero point five eight seconds.** And here, *[point at PASS]*, the official correctness check:
**PASS**.

*[Slide 9]*

Same machine, one session, every version side by side: 35 seconds down to 0.585. **Sixty times
faster.**

## 2:25 – 3:00 · Any setup, what's next  *(≈ 80 words)*

*[Slide 10]*

And the final scoring uses a **different** setup, so we built nothing for this one. On the official
aLIGO benchmark: **21 seconds to under half a second**, PASS. Ten more detectors, 45 times faster on
average.

*[Slide 11]*

Every output and every gradient matches the official code, and on ill-conditioned detectors we
are **more accurate** than the dense code.

*[Slide 12 → 13]*

What's left is mostly `import jax`. Same physics. Same gradients. **Any detector.** Thank you.

---

## Alternative demo: the race video (`race_100_steps.mp4`)

Use it instead of the live run (no network needed), or right after it. It shows 100 real optimisation
steps, original vs final, on the same EPYC core, from measured timestamps
(`results/race/*.jsonl`). The first seconds play in real time; then it fast-forwards, and the speed
is shown on screen.

Say over it (~40 s): *"Here is what that means for design work: 100 optimisation steps, gradient and
Adam each step, the original on top, ours below, started at the same moment. Ours is done in under
eight seconds. The original is still compiling. We now fast-forward, and the speed is on screen.
And look at the loss curves: they lie on top of each other. Same physics, same design, just
**seconds instead of minutes**."* *[Read the final ratio off the end card.]*

## If the demo fails

- **No network / SSH refused:** run `./docs/pitch/demo.sh` without `DEMO_SSH` (local container).
  Say: *"On this laptop it's slower, about 1 to 2 seconds, because of the disk mount; the
  0.585 is from the EPYC."*
- **Anything else:** skip to slide 9 and say: *"Here is the recorded run from that machine:
  21 runs, all PASS."* That loses no time.
- **If asked to show the new setup live:** `DEMO_SETUP=aligo DEMO_SSH=... ./docs/pitch/demo.sh`
  runs the official `benchmark_aligo.py` (about 0.49 s first run, PASS).
- **Before going on stage:** run the demo once (it warms the disk cache; the measured number
  doesn't change) and keep the terminal open at the right folder with a large font.

## Likely questions (keep answers to about 15 seconds)

- **Is it still correct?** All 21 runs PASS, the error is 5.374e-6 (the same as the original),
  the gradient cosine is 1.0000000000, and the finite-difference error is 4.3e-9 vs 7.0e-8 for the
  original. An independent validation against the official code matched every output array and all
  gradients: 1e-10 on aLIGO, 1e-7 on the grid (`FEYER_FINAL_VALIDATION.md`).
- **Is it specific to this setup?** No. The official aLIGO benchmark passes at 0.490 s (from
  20.78 s), and a 10-setup suite (aLIGO, Voyager, uIFO, 5–301 frequencies) is 45× faster on average,
  every case matching the original. Sparsity patterns, orderings and pivots come from the setup. A
  custom component falls back to traced JAX automatically.
- **Do the judges need a C++ compiler?** No. The solver ships prebuilt for Linux x86-64 (baseline,
  AVX2, AVX-512, chosen from the CPU flags), so `pip install` compiles nothing. With no usable
  library it falls back to SciPy and still passes.
- **Is the C++ solver really used, or does it silently fall back?** We instrumented the official
  scripts: every solve ran on the C++ solver, with zero SciPy or JAX fallback.
- **What do you lose?** Forward-mode AD on the fast path. `DIFFEROMETOR_BACKEND=jax` keeps the full
  JAX path.
- **Are the compiler flags a trick?** They are set by the package, recorded in the benchmark's own
  JSON, and bit-identical. The `fast` mode is the only one that is not, and it is opt-in.
