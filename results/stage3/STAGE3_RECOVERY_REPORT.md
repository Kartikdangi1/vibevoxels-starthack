# Stage 3 — Post-crash recovery report

Written 2026-11-21 ~20:05 CET, after the Mac rebooted at 19:56:51 CET.
Timestamps in the previous session's transcript are UTC (CET − 1 h); they are converted to CET below.

## A. Repository state

| Item | Value |
|---|---|
| Main worktree | `/Users/samarthshetty/vibevoxels-starthack`, branch `feyer-sparse-host-solver`, HEAD `ad9603c` (= protected commit, = `origin/feyer-sparse-host-solver`) |
| Main worktree dirty files | `container.bash` mode change 644→755 only (no content change; predates Stage 3). Untracked: `research/`, PDFs, `.DS_Store`, an empty file `differometor/reserach` (0 bytes, 19:38, typo'd name — harmless, left in place) |
| Experimental worktree | `/Users/samarthshetty/vv-exp`, branch `experiment/local-component-plan`, HEAD `ad9603c` (**no commits made** by Stage 3) |
| Experimental uncommitted changes | `M differometor/host.py` (+2/−1), `?? differometor/native.py` (359 lines), `?? tests/check_native.py` (94 lines) |
| Other branches | `main`, `submission` (= `e3dfa35`). No stashes. No detached HEADs. |

The protected branch was **not** modified by Stage 3.

## B. Recovered previous-session activity

Source: Claude Code transcript `~/.claude/projects/-Users-samarthshetty-vibevoxels-starthack/03116445-b6ab-4b22-916e-f493f71e2ee9.jsonl`
(last write 19:50:30 CET). A condensed, complete timeline of every tool call and result is saved at
[`research/stage3/prev_session_03116445_timeline.txt`](stage3/prev_session_03116445_timeline.txt).

| CET | Action | Outcome |
|---|---|---|
| 19:42 | Read memory, git state, Stage 2 report, OPTIMIZATION_LOG, host.py; env check (Py 3.12.10, JAX 0.9.0, NumPy 2.5.5, SciPy 1.18.1, load avg ≈ 2.1) | done |
| 19:42:55 | `git worktree add -b experiment/local-component-plan ../vv-exp ad9603c`; confirmed the editable install resolves `differometor` to the cwd (vv-exp) | done |
| 19:43:07 | Baseline: 5 sequential `benchmark.py --repeats 3` runs of unmodified ad9603c | all PASS: 1.437 (cold disk cache), 0.681, 0.739, 0.773, 0.954 s |
| 19:43:53 | Profiling harness (wraps host.py functions, runs benchmark unchanged) | first-run: `trace_function` 85.8 ms (16 fns; incl. `jvp make_jaxpr` 47.6 ms, `_self_check` 38.1 ms); `forward_lower` ≈ 97 ms |
| 19:46:22 | Per-function trace probe | surface 18.4 ms, corrected_om_right 10.3, … total ≈ 82 ms |
| 19:46:29 | **Gate A decision: GO** (~82 ms removable) | |
| 19:47:05–19:47:19 | Wrote `differometor/native.py` (Wirtinger-dual NumPy kernels for all 19 built-in component functions), hooked into `Plan.__init__` | uncommitted |
| 19:47:38–19:49:09 | Wrote/iterated `tests/check_native.py` (native vs `jax.vjp`, complex/physical/boundary inputs, FD check) | see D |
| 19:49:09 | Benchmark `DIFFEROMETOR_NATIVE=surface` and `=all` (1 run each) | PASS 0.650 s / PASS **0.518 s** |
| 19:49:31 | Full saved-array comparison, `all` vs baseline run b1 | bit-identical carrier/signal/noise/powers/sensitivity; gradient normrel 1.6e-16, maxabs 7.3e-12 |
| 19:50:03 | Wrote 13-topology harness `generality.py` (native vs traced vs in-graph JAX, 20 freqs) | first attempt: harness bug (`optimized_parameters` is None), failed instantly |
| **19:50:27** | Patched harness and **launched it** (foreground, 600 s timeout; all 13 cases × 3 modes in one process) | **no result ever recorded** — machine hung, then panicked at 19:57:05 |

## C. Existing experimental changes

* `differometor/native.py` — `Dual` sparse Wirtinger dual number (holomorphic partials `a`, anti-holomorphic `b`, only `conj` swaps them); kernels re-expressing each component function in the same operation order; `NativeFunction` with the `TracedFunction` interface (`__call__`, `vjp`, `out_shape=(4,)`, `unsupported`). Used only when `FUNCTIONS[fid]` is the original function object; otherwise falls back to JAX tracing. `DIFFEROMETOR_NATIVE=none|<names>` for ablation. Linking functions are still JAX-traced.
* `differometor/host.py` — import + `native_function(FUNCTIONS[fid]) or trace_function(...)`.
* `tests/check_native.py` — component-level validation.
* Lost with the reboot (`/private/tmp` scratchpad): baseline JSON/NPZ, profiling harness, `cmp.py`, `generality.py`. Their source and printed results survive only in the transcript.

Recovery review of the diff (this session): syntactically complete; each kernel matches `components.py` term by term (argument order for surface/transmissivity/reflectivity_right incl. the swapped refractive indices, `conj` of `tuning_correction`, squeezer conj outputs, susceptibility double negation, `x**2` → `x*x` as JAX's `integer_pow`). VJP formula `ct_i = A ct + conj(B ct)` is the transpose of the JVP in JAX's convention. Nothing appears partially written.

## D. Completed experiments

| Experiment | Status | Evidence |
|---|---|---|
| Baseline ad9603c, 5 runs | COMPLETED (transcript only) | median 0.773 s; 0.616 s was **not** reproduced in that session (load avg 2–3.6) |
| Profiling / Gate A | COMPLETED | ≈ 82 ms Plan construction removable |
| Native kernels, component level | COMPLETED, 2 flagged rows | all 19 functions ≤ 1e-10 vs JAX except `space_modulation(_lower)` `zero_alpha` VJP 7.9e-10 / 1.7e-10 (threshold 1e-10). Attributed by the previous session to frequency-derivative cancellation; **not yet confirmed which column** |
| Native kernels, public benchmark | COMPLETED UNVERIFIED for speed (1 run) | PASS, 0.518 s vs baseline runs 0.68–0.95 s in the same window, `forward_lower` 90→10 ms |
| Full-output comparison | COMPLETED | forward bit-identical, gradient 1.6e-16 normrel |

## E. Incomplete experiments

* **13-topology generality validation** (aligo×3 modes, voyager×3, uifo3 seeds 0–2, uifo2 seed 5, uifo_20_5000): started 19:50:27, interrupted by the crash, no output. **Must be redone**, safely (see §G and §I).
* Repeated alternating baseline/candidate benchmarks: not started.
* Challenger audit of `native.py`: not started.
* Candidate B (KLU) / C (SCC): not started.

## F. Benchmark artifacts

None of the Stage 3 JSON/NPZ files survived (written to the wiped scratchpad). The numbers above are copied from the recorded tool output in the transcript, which is reliable evidence that those commands ran and printed those lines. `research/stage3/` was created at 19:43 and was empty. Historical results in `results/` (V0–E9) are untouched.

## G. Crash evidence

* `/Library/Logs/DiagnosticReports/ResetCounter-2026-11-21-195709.diag`: `Boot faults: wdog,reset_in_1`.
* Panic report `panic-full-2026-11-21-195705.0002.panic` (read once; macOS then rotated it away, only `.contents.panic` remains):
  * `panic(cpu 5 …): watchdog timeout: no checkins from watchdogd in 92 seconds` — a **userspace-hang watchdog panic**, not a kernel bug in a driver.
  * `Compressor Info: 42% of compressed pages limit (OK) … with 7 swapfiles and LOW swap space`.
  * Memory snapshot: free 3 206 pages (≈ 50 MB), active 15, inactive 12 pages, wired 1.3 GB, compressor holding ≈ 399k pages — physical RAM fully exhausted.
  * Largest resident process: **`Python` pid 58629, 3.2 GB**; Chrome helpers ≈ 5 GB combined; VS Code ≈ 1.4 GB. Machine RAM: **8 GB**.
* Transcript stops 19:50:27 (generality launch); hang ≈ 19:50–19:55; panic 19:57:05.

**Conclusion (with uncertainty):** the evidence is consistent with system-wide memory exhaustion (RAM + swap) starving `watchdogd` until the watchdog fired. The only heavy job running was the 13-topology harness, and a Python process at 3.2 GB was the single largest consumer, so that harness very likely pushed an already-loaded 8 GB machine (≈ 7 GB of other RSS) over the edge. I cannot prove pid 58629 was the harness, nor which case/mode grew it (candidates: the eager, un-jitted `value_and_grad` through the in-graph `DIFFEROMETOR_BACKEND=jax` path; 39 runs accumulating in one process; uifo_20_5000). It was not a hardware fault signature and not specifically a JAX crash.

## H. Last verified checkpoint

* **Protected & passing:** `ad9603c` on `feyer-sparse-host-solver` (Challenger APPROVE, unchanged).
* **Most advanced experimental state:** uncommitted native kernels in `vv-exp`, validated at component level and on the public benchmark, **not** validated on alternate topologies.

## I. Safe continuation point

1. Commit the native-kernel work on `experiment/local-component-plan` as an explicitly *unvalidated* checkpoint (preserves it).
2. Re-run `tests/check_native.py` (cheap) and resolve the two flagged `space_modulation` rows.
3. Re-run the topology validation **one case and one mode per process**, under an RSS guard that kills the child above 1.5 GB, skipping the in-graph JAX reference on the large cases unless it is shown to be small.
4. Then alternating baseline/candidate benchmarks, one process at a time.
5. Challenger (read-only), then Candidate B only if justified.

## Stage 3 checkpoint table

| # | Step | Status |
|---|---|---|
| 1 | Verify repo & env | COMPLETED VERIFIED |
| 2 | Read Stage 1/2 reports | COMPLETED |
| 3 | Reproduce baseline | COMPLETED (median 0.77 s under load; artifacts lost) |
| 4 | Profile component-JVP compilation | COMPLETED |
| 5 | Decide Candidate A | COMPLETED — GO |
| 6 | Implement local derivative rule | COMPLETED UNVERIFIED (all 19 functions, uncommitted) |
| 7 | Validate physics & gradients | PARTIAL — public setup verified; alternate topologies INTERRUPTED |
| 8 | Repeated first-run benchmarks | NOT STARTED |
| 9 | Expand if beneficial | N/A (already covers all functions) |
| 10 | KLU symbolic reuse | NOT STARTED |
| 11 | SCC decomposition | NOT STARTED |
| 12 | Challenger audit | NOT STARTED |

## Resource-safety policy in force from now on

* One Python workload at a time; no parallel agents running code.
* Every validation child runs under a watchdog wrapper polling RSS every 0.2 s and killing it at a fixed cap; benchmarks use `/usr/bin/time -l` to record peak RSS.
* Topology validation: one case × one mode per process (no accumulation).
* No dense 101×751×752 references; no in-graph JAX reference on large cases.
* Stop all heavy work if free memory drops below ~1.5 GB or swap use climbs sharply.
