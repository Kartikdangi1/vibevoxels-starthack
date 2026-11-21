# 3-minute pitch script — Feyer: Faster Physics

Timing: 354 spoken words ≈ 2:37 at ~135 words per minute, plus ~6 s for the demo run and ~10 s to
switch to the terminal and back ≈ 2:55 in total.
**Bold** marks the words to stress; *[cues]* are actions, not spoken. Slide numbers refer to
`pitch.pdf`.

| Part | Time | Slides |
|---|---|---|
| Problem | 0:00 – 0:30 | 1 → 3 |
| Solution | 0:30 – 1:45 | 4 → 8 |
| Demo | 1:45 – 2:30 | terminal, then 9 |
| What's next | 2:30 – 3:00 | 10 → 12 |

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

The solver is our own C++ sparse LU. The pivot order from the first frequency is reused for the
other hundred.

*[Slide 8]*

And the last 70 milliseconds came from XLA compiler flags and a CPU-tuned build. They are
**bit-identical**: not one output bit changes.

## 1:45 – 2:30 · Demo  *(≈ 85 words; the run takes ~6 s)*

*[Switch to terminal. Run:]* `DEMO_SSH="ssh -p <port> root@<host>" ./docs/pitch/demo.sh`

This is the official benchmark, unmodified, on the AMD EPYC we measured on, pinned to **one** CPU
core. It includes everything: imports, compilation, forward pass, sensitivity and the gradient.

*[Point at "Complete first-run total"]*

**Zero point five eight seconds.** And here, *[point at PASS]*, the official correctness check:
**PASS**.

*[Slide 9]*

Same machine, one session, every version side by side: 35 seconds down to 0.585. **Sixty times
faster**, and with the opt-in fast mode, 69 times.

## 2:30 – 3:00 · What's next  *(≈ 70 words)*

*[Slide 10]*

The log-sensitivity error is identical to the original, and on ill-conditioned detectors we are
actually **more accurate** than the dense code.

*[Slide 11]*

What's next: half of our remaining time is just `import jax`. On the compute side, all 101
frequencies share one pivot order, so we can vectorise the LU **across frequencies**, still
bit-identical.

*[Slide 12]*

Same physics. Same gradients. **Thirty-five seconds to half a second.** Thank you.

---

## If the demo fails

- **No network / SSH refused:** run `./docs/pitch/demo.sh` without `DEMO_SSH` (local container).
  Say: *"On this laptop it's slower, about 1 to 2 seconds, because of the disk mount; the
  0.585 is from the EPYC."*
- **Anything else:** skip to slide 9 and say: *"Here is the recorded run from that machine:
  21 runs, all PASS."* That loses no time.
- **Before going on stage:** run the demo once (it warms the disk cache; the measured number
  doesn't change) and keep the terminal open at the right folder with a large font.

## Likely questions (keep answers to about 15 seconds)

- **Is it still correct?** All 21 runs PASS, the error is 5.374e-6 (the same as the original),
  the gradient cosine is 1.0000000000, and the finite-difference error is 4.3e-9 vs 7.0e-8 for the
  original.
- **Is it specific to this setup?** No: the identity is exact for any topology, and we checked
  aLIGO, Voyager, UIFO and random topologies against the JAX path. A custom component falls back to
  traced JAX automatically.
- **What do you lose?** Forward-mode AD on the fast path. `DIFFEROMETOR_BACKEND=jax` keeps the full
  JAX path.
- **Are the compiler flags a trick?** They are set by the package, recorded in the benchmark's own
  JSON, and bit-identical. The `fast` mode is the only one that is not, and it is opt-in.
