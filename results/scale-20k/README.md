# 20,000-frequency sweeps: final/combined-v2 vs the official Differometor

Full simulation, sensitivity and reverse-mode gradient at **20,000 log-spaced frequencies from
10 Hz to 20 kHz**, on the shipped commit `c8066b8` (prebuilt C++ solver, nothing compiled).
Machine: AMD EPYC 9334, one pinned core, 22 November 2026.

| Setup | Frequencies | Design variables | First run (incl. imports, compile) | Warm | Peak memory |
|---|---:|---:|---:|---:|---:|
| Development 3×3 grid | 1,000 | 193 | 1.30 s | 0.80 s | 2.1 GiB |
| Development 3×3 grid | 5,000 | 193 | 5.13 s | 4.49 s | 9.3 GiB |
| **Development 3×3 grid** | **20,000** | **193** | **20.07 s** | **18.81 s** | **28.6 GiB** |
| **aLIGO** | **20,000** | **29** | **1.79 s** | **1.33 s** | **2.3 GiB** |

Every run: all sensitivities positive, all outputs and gradients finite, every gradient entry nonzero.

**Official code, for scale.** It needs 19.5 s (grid) and 10.1 s (aLIGO) for just 100 of these
frequencies, with 6.2 GiB peak on the grid. It builds dense 751×752 complex matrices per frequency
(14.4 GiB peak at 101 frequencies on the official benchmark), so 20,000 frequencies are out of reach
for it. That is roughly 2.8 TB by linear extrapolation.

## Correctness

The official code ran on every 200th frequency of the same grid (100 frequencies, 10 Hz to 18.5 kHz):

| | Grid | aLIGO |
|---|---:|---:|
| Our 20k run vs official, max \|Δlog S\| over the 100 shared frequencies | 1.7e-8 | 1.3e-10 |
| Same 100-frequency sample: gradient relative error vs official | 4.6e-8 | 3.8e-10 |
| Gradient cosine vs official | 1.0000000000 | 1.0000000000 |
| Our 20k run vs our own 100-frequency run (batch independence) | 3.8e-10 | 7.1e-15 |

Note: on the aLIGO sample, the official gradient has 28 of 29 entries nonzero and ours 29. The
extra entry is an optical length whose gradient is at roundoff level (about 1e-34, as in
FEYER_FINAL_VALIDATION.md). In ours it is a tiny nonzero, in the official code an exact zero.

The loss for these sweeps is the official readout with reference 1, i.e. mean log10 S. No public
reference exists for these frequency grids, so the official implementation is the reference.

## Files

- `*_20k`, `dev_1k`, `dev_5k`: our runs. `*_sub_opt` / `*_sub_ref`: the 100-frequency sample, ours / official.
- `*.json`: timings, memory and checks. `*.npz`: frequencies, sensitivity, gradient, loss. `*.log`: console output.
- `comparison.txt`: output of `scripts/scale_compare.py`.
- `scripts/`: `scale.py` (one run), `run_scale.sh` (all runs), `scale_compare.py`, plus the `vcase.py`/`vprobe.py` helpers they import.

Reproduce, from a checkout of `final/combined-v2`:
`python scripts/scale.py dev 20000 10 20000 out/dev_20k` (run in the repository root; needs about 30 GiB RAM).
