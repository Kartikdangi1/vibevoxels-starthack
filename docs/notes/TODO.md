# Speed-up TODO (Differometor, Faster Physics challenge)

Target: **complete first-run total** of `python benchmark.py --cpus 1` on a Xeon Platinum 8468,
pinned to **one logical CPU**, no GPU. The baseline is about 44 s:

| Phase | Baseline |
|---|---|
| imports | 2.6 s |
| build | 1.0 s |
| forward lower | 2.0 s |
| **forward compile** | **15.0 s** |
| **forward run** | **13.3 s** |
| gradient compile | 5.2 s |
| gradient run | 4.3 s |

Source: `benchmarks/cpu-baseline.json`.

## Implementation status (2026-11-21)

Implemented and benchmark-validated in this checkout: T1 factor reuse with an implicit `custom_linear_solve` derivative, the static function grouping and linking portions of T3, frequency-independent update hoisting and the single-column carrier path from T4, T6 per-frequency `lax.map` with checkpointing, and the lazy Plotly / lookup / NumPy build changes from T7. `tools/inspect_structure.py`, `tools/compare_arrays.py`, and `tools/check_factored_solve.py` are present. The official benchmark passed reference sensitivity and finite-difference gradient checks on the available WSL2 i5-1145G7 host. It used one logical CPU (`requested_logical_cpus: 1`, `cpu_affinity_count: 1`) and took 20.36 s on its cold first run; see [`APPROACH.md`](APPROACH.md) and [`../../bench-custom-solve-cpu1.json`](../../bench-custom-solve-cpu1.json).

This does not complete the TODO. Input-port elimination and nested-dissection solves remain open; a cell-wise Schur prototype passed correctness but was removed because cold first-run time increased. Separated matrix/RHS storage and sparse noise are also not implemented. The available host has Python 3.14.4 rather than the requested 3.12. The reference fixture line endings are pinned to LF in `.gitattributes`, so the checked-in benchmark now runs without a temporary harness. The current WSL measurements are not a direct comparison against the Xeon baseline. See `APPROACH.md` for one-CPU measurements and rejected cold-start experiments.

## Ground rules (read first)
- **Do not edit `benchmark.py`**. Its SHA-256 is recorded in the report. It only calls
  `df.run_build_step(...)` and `df.simulate(**arrays)`, so anything inside `differometor/`
  may change as long as those two calls keep working:
  `run_build_step` returns `(arrays_dict, ports, ...)`, and `simulate(**arrays)` returns
  `(carrier, signal, noise)` with the same shapes ((286,1), (751,101), (101,)).
- Keep float64/complex128, all 101 frequencies and all 193 parameters.
  The pass criterion is `|ln(sens) - ln(ref)| <= 1e-5`, plus the finite-difference gradient
  check on `mb11.tuning`.
- After every task: `python benchmark.py --cpus 1 --repeats 3 --output bench-<task>.json`
  must print PASS. Commit the JSON files so we can compare before and after.
- **Static data trick** (needed by T3 and others): `arrays` is passed to `jax.jit`, so every
  array in it is *traced*. Index data we want to be compile-time constants goes into a
  hashable object registered with `jax.tree_util.register_static` (no array leaves), e.g.
  `arrays["plan"] = StaticPlan(...)`. Then `simulate(..., plan)` sees plain NumPy/ints at trace
  time. The arg-count check in `run_build_step` (simulate.py ~L460) must be updated to match.

---

## T0. Environment and measurement (blocker)
- [ ] Set up Python 3.12 venv on **WSL2/Linux** (JAX does not import on our Windows Py3.15).
      `pip install -r requirements.txt && pip install -e .`
- [ ] Run baseline: `python benchmark.py --cpus 1 --output bench-baseline.json --save-arrays baseline.npz`.
- [ ] New script `tools/inspect_structure.py`: build the setup and print, for the signal matrix,
      nnz per block (U-U, U-L, L-U, U-S, S-U, L-S, S-L, S-S) and which entries depend on frequency.
      **Confirms the arrowhead structure assumed in T2** (no direct U↔L coupling).
- [ ] New script `tools/compare_arrays.py`: compare a candidate `.npz` against `baseline.npz`
      (carrier/signal/noise/gradient, max relative error). It's a stricter check than the benchmark's.
- [ ] Optional: `JAX_LOG_COMPILES=1` and `XLA_FLAGS=--xla_dump_to=/tmp/xla` to see the compiled graph.

## T1. Factorise the signal matrix once, reuse it 3× (easy, about 2–3× forward)
File: `differometor/simulate.py`
- [ ] L335 `signal = vmap(solve)(...)`: replace with `lu = vmap(lu_factor)(A)`, then `vmap(lu_solve)(lu, b)`.
- [ ] L370–375: drop `jnp.transpose(jnp.conjugate(...))` and the conjugate-transpose solve.
      Use `lu_solve(lu, B, trans=2)`, which solves Aᴴx = b without a copy.
- [ ] L407: reuse the same `lu` for the covariance solve.
- [ ] Solve all detectors at once: give the right-hand sides shape `(F, N, D)` instead of
      `vmap(vmap(solve), in_axes=(None, 0))`. That's one BLAS3 `trsm` instead of D `trsv` calls.
      Same for the `einsum` at L403: `"bij,bjd->bid"`.
- [ ] Check that the gradient still matches (`jax.scipy.linalg.lu_factor/lu_solve` are differentiable).

## T2a. Eliminate input-port unknowns exactly (halves every optical block, about 8× per LU)
Files: `differometor/build.py` (index layout), `differometor/simulate.py` (assembly + back-substitution)
- [ ] Structure (from build.py): the matrix starts as **identity** (L652/L660). **Components** write into
      *output-port rows / input-port columns* (e.g. surfaces: rows `idx+[1,3,5,7]`, cols `idx+[0,2,4,6]`,
      around L980), and those values don't depend on frequency. **Spaces** write exactly one entry per
      direction into *input-port rows / output-port columns* (L1288–1291), `-exp(-i2πfLn/c)`, and these
      are **the only frequency-dependent optical entries**.
- [ ] Reordered as (in, out): `[[I, P(f)], [K, I]]`, where P(f) is a permutation times diagonal phases. Substitute:
      `x_in = b_in − P(f) x_out`, then `(I − K·P(f)) x_out = b_out − K b_in`. Per sideband: 286 → about 143 unknowns.
- [ ] `K·P(f)` = K with permuted, phase-scaled columns, so it costs O(nnz) to build. Build K once (ties into T4).
- [ ] Apply the same substitution to the signal-block couplings (B, C rows/cols that touch input ports).
- [ ] T0 structure script must confirm: (a) no component writes into an input-port row, (b) no space
      writes into an output-port row, (c) unconnected ports (L1379), lasers and detectors (index
      special cases at L1276–1288) give pure identity input rows.
- [ ] Exact algebra (a Schur complement on an identity block), so physics and gradients are unchanged.

## T2. Block/Schur solve for the arrowhead signal system (largest gain, about 8× per factorisation)
Files: `differometor/simulate.py` (new helper), `differometor/build.py` (layout)
- [ ] New function `arrowhead_factor(A_U, A_L, B_U, B_L, C_U, C_L, S)` and `arrowhead_solve(fac, b, trans)`
      in `simulate.py` (or a new `differometor/linalg.py`):
      LU of A_U and A_L (286×286), Schur complement `S - C_U A_U⁻¹ B_U - C_L A_L⁻¹ B_L` (179×179), its LU.
      It also needs a conjugate-transpose solve (for the noise weights): write it as the block
      formula with `trans=2` on each factor.
- [ ] Split the slices at `N = carrier_size` and `2N` in `simulate`. Better still, build the five blocks
      directly (see T5) so the dense 751×751 matrix never exists.
- [ ] Sizes are chosen to fit L2 (2 MB/core): 286² × 16 B = 1.3 MB, 179² × 16 B = 0.5 MB.
- [ ] If the U/L blocks turn out very sparse (T0), consider `klujax` later (T9). Check whether extra deps are allowed.
- [ ] Combined with T2a: about 143 (U) + 143 (L) + 179 (S). Roughly 2·143³ + 179³ ≈ 1.2e7 vs 4.2e8 for dense 751³,
      **about 35× less LU work**. The 179×179 signal block then dominates.

## T2c. Cell-wise nested dissection for the 3×3 grid (exact, optional after T2a/T2)
Files: `differometor/build.py` (cell membership of every port), new `differometor/linalg.py`
- [ ] The UIFO setup (arXiv:2312.04258) is a **planar 3×3 grid of cells**. For grid/planar graphs, nested
      dissection gives the lowest fill: eliminate each cell's interior first, then the separators (links between cells).
- [ ] Build time: tag each unknown with its cell id, or "separator" if it sits on a link between cells.
      Store the permutation and block boundaries in the static plan.
- [ ] Runtime: per frequency, batched LU of the 9 interior blocks (one `vmap` over cells if padded to equal size),
      Schur complement onto the separators, small dense LU there, back-substitute.
- [ ] Only worth it if, after T2a/T2, the U/L blocks still dominate the profile. Measure first.

## T3. Static indices + group component functions by type (main compile-time fix)
Files: `differometor/build.py` (`pairs_to_arrays` L1445, `combine_arrays` L1615),
`differometor/simulate.py` (`update` L62–112, linking at L276–283)
- [ ] Today `vmap(lambda i, x: lax.switch(i, FUNCTIONS, x))` (L109) has a *traced, batched* `i`, so
      XLA compiles and runs **all 19 `FUNCTIONS` for every entry**. The same happens for
      `LINKING_FUNCTIONS` at L278.
- [ ] In `pairs_to_arrays`: group entries by `function_indices` with NumPy and store per group
      `(fn_id, input_idx, output_idx, matrix_rows, matrix_cols)` in the static plan (see Ground rules).
- [ ] New `update`: `for fn_id, g in plan.groups: vals = vmap(FUNCTIONS[fn_id])(params[g.input_idx])`,
      then concatenate and do **one** scatter per matrix.
- [ ] Same for linking: group by linking function, call each once.
- [ ] Sort scatter indices row-major at build time (better locality).
- [ ] Fix the global mutation at `build.py` L401/L1418 (`LINKING_FUNCTIONS.append`). Keep the
      functions in the plan instead, so repeated builds don't grow a global list.

## T4. Hoist frequency-independent work out of the 101× loop
Files: `differometor/simulate.py` (`expand_parameters` L114, the `vmap(update)` calls at L290/L304/L378),
`differometor/build.py` (classify entries)
- [ ] In the plan, tag each group as **f-dependent** (functions that take the frequency parameter:
      `space`, `space_lower`, `space_modulation*`, `susceptibility`, `signal_function`, ...) or
      **f-independent** (mirrors, beamsplitters, lasers, noise entries, ...).
      Do it from data: an entry is f-dependent iff its input indices contain `signal_changing_parameter_indices`.
- [ ] Evaluate f-independent groups **once** on the un-expanded parameter row. Build a base matrix
      (or base blocks) once, then broadcast and scatter only the f-dependent values per frequency.
- [ ] Noise matrix (L378): probably fully f-independent. Compute it once and keep it **sparse**
      (values + indices). Replace the dense `einsum` with a gather/segment-sum.
- [ ] Carrier: `carrier_changing_values` has 1 column in the benchmark. Skip `expand_parameters` for it.

## T5. Memory layout / fewer full-size copies
Files: `differometor/build.py` (L651, L659, L663, `set_instructions` L94), `differometor/simulate.py`
- [ ] Store matrix and RHS **separately** instead of augmented `(N, N+1)` (build.py L651/L659).
      The `[:, :, :-1]` / `[:, :, -1]` slices at L298/L335/L370/L407 create strided copies.
- [ ] Sideband conjugation (simulate.py L325–333): three `.at[block].set(conj(block))` passes over
      the whole dense array. Instead, mark the affected entries at build time and **conjugate the
      value vector before scattering**.
- [ ] Carrier scaling of signal entries (L318–323): fold it into the same single scatter.
- [ ] Remove `jnp.tile` / `jnp.broadcast_to` at L356/L371 once T1 is done.

## T6. Per-frequency processing + gradient memory (do after T1–T5)
File: `differometor/simulate.py`
- [ ] Restructure as `lax.map(per_frequency, freqs)`: build → solve → noise for one frequency at a time,
      so its working set stays in L2/L3 instead of streaming 0.9 GB per stage.
- [ ] Try `jax.checkpoint` on the per-frequency body to cut the 14 GiB AD tape. Measure it,
      because recomputation costs time.
- [ ] Compare `lax.map` vs `vmap` on compile time *and* run time. The scored number includes both.

## T7. Startup overhead
- [ ] `differometor/__init__.py` L3: `from differometor.plot import visualize_setup` imports plotly
      on every run. Make it lazy (module `__getattr__` or import inside the function).
- [ ] `build.py` L1414: `parameter_names.index(...)` in a loop is quadratic. Use a dict.
- [ ] Profile the 1.0 s build with `python -X importtime` / `cProfile` and fix only what shows up.
- [ ] `build.py` L1288 calls the **JAX** function `space(jnp.array(...))` eagerly once per edge. Each eager
      JAX op dispatches separately, and the first call of each op compiles a small kernel. Use a NumPy
      version (`-np.exp(-1j*2*np.pi*F*L*n/C_LIGHT)`). Same check for any other eager `jnp` call in build/setups.

## T9. (Optional, needs organiser OK) `klujax` sparse direct solver
- [ ] KLU (SuiteSparse) for JAX: CPU-only, float64/complex128 only, LGPL-2.1, v0.5.3. COO input
      `solve(Ai, Aj, Ax, b)` with a shared batch dimension (all 101 frequencies in one call).
- [ ] `analyze` once (fixed pattern), outside jit. Use **`solve_with_symbol`** for differentiable paths:
      raw `factor`/`solve_with_numeric` are forward-only. Don't create handles inside jit (they leak memory).
- [ ] Check whether it can do the conjugate-transpose solve needed for the noise weights.
- [ ] Ask the organisers whether a new dependency in `requirements.txt` is allowed.
- [ ] Timebox: half a day. Only pursue it if it beats T2a + T2 on the same setup.

## T10. (Optional, needs organiser OK) XLA compiler flags for compile time
- [ ] `XLA_FLAGS` is read when the backend initialises, which happens *after* `import differometor` in benchmark.py.
      So `differometor/__init__.py` could set e.g. a lower backend optimisation level or disable expensive LLVM
      passes. Verify the exact flag names for jaxlib 0.9.0 (`--xla_backend_optimization_level`,
      `--xla_llvm_disable_expensive_passes`).
- [ ] This trades runtime for compile time, so measure both. The report records `xla_flags` from the
      environment, so it's visible to judges. Ask if this counts as fair.

## T11. Hand-written adjoint (`jax.custom_vjp` on `simulate`), a likely differentiator
File: `differometor/simulate.py`
- [ ] Today JAX's autodiff records every scatter, conjugation and solve, then generates the backward program
      (5.2 s compile, 4.3 s run, 14 GiB tape).
- [ ] Adjoint method for `A(θ) x = b(θ)`: forward stores only the **LU factors + solutions**. Backward:
      one adjoint solve `Aᴴ λ = ∂L/∂x̄` with the *same* LU (`trans=2`), then
      `∂L/∂θ = Re(λᴴ (∂b/∂θ − ∂A/∂θ x))`. ∂A/∂θ is just the derivative of the few component functions
      at their scattered positions, so use `jax.vjp` on the small `update` value vector, not on the matrix.
- [ ] Chain it through the three stages: noise → signal → carrier (signal entries depend on the carrier
      solution, the noise on both).
- [ ] `benchmark.py` calls `jax.vjp(simulate)` and passes cotangents for all three outputs
      (carrier, signal, noise), so the custom VJP has to accept all three.
- [ ] Validate against the existing gradient with `compare_arrays.py` (all 193 entries, not just mb11.tuning).

## T14. Drop the second noise solve: `noise² ∝ Re(vᴴ Σ v)` (exact, from thesis eq. 2.74–2.75)
File: `differometor/simulate.py` (current per-frequency body, about L460–476)
- [ ] Source: the Differometor thesis (`media/documentation.pdf`, §2.3.5): `V_o = M⁻¹ V_i M⁻ᴴ`, and the
      noise is `S = s_cᴴ V_o s_c`. The current code computes it as two solves:
      `v = M⁻ᴴ s` (`noise_weights`, trans=2), then `c = M⁻¹ (Σ v)` (`covariances`), then `Re(sᴴ c)`.
- [ ] Identity: `sᴴ M⁻¹ (Σ v) = (M⁻ᴴ s)ᴴ (Σ v) = vᴴ Σ v`. So replace
      `covariances = solve_factored(system, factors, noise_sources)` and the `projected` sum with
      `projected = jnp.real(jnp.sum(jnp.conjugate(noise_weights) * noise_sources, axis=0))`.
      That removes one forward + backward triangular solve pair per detector per frequency, plus its AD residuals.
- [ ] Exact algebra, but check `compare_arrays.py` and the benchmark: floating-point rounding changes slightly.
- [ ] Combine with T4's sparse noise: the thesis (Fig. 3.4) shows `V_i` is **diagonal** (vacuum and loss noise)
      plus small **upper↔lower sideband blocks for squeezers**. So `vᴴ Σ v = Σᵢ dᵢ|vᵢ|² + squeezer cross terms`,
      which is O(n) instead of a dense n² matvec.

## Findings from the papers (for planning and the write-up)
Sources: the Differometor thesis by Klimesch & Krenn (`media/documentation.pdf`) and Krenn, Drori &
Adhikari, "Digital Discovery of interferometric GW Detectors", arXiv:2312.04258 / PRX 15, 021012 (2025).
- **The authors' own profiling** (thesis §3.4.3): *"steps 3 to 7 take the maximum amount of time"*. Those steps
  are the matrix updates. JAX does these updates *out-of-place*, returning a modified copy of the whole
  matrix each time. This confirms T3/T5 (group functions, one scatter per matrix, fewer full-matrix copies).
- **Sparsification is the authors' planned next step** (thesis §3.4.3, §4.3, Outlook): *"Reducing the size of the
  modified interferometer matrix using sparsification promises another performance boost"*. They also note
  CPU Differometor is slower than Finesse, which uses sparse matrices and in-place updates. Use this in
  `APPROACH.md`: T2a/T2/T14 *implement the authors' own outlook*. UIFO size 6 already ran out of
  memory for them, which supports T6/T11.
- **Finesse's quantum-noise method** (thesis §2.3.5, "MISO", multiple-input single-output): one solve per
  *detector*, not per noise source, because there are far fewer detectors than sources. We already do this;
  T14 removes the redundant second half of it.
- **Function forms useful for T4 / precomputation** (thesis §3.3):
  - space `−exp(−i2π·Δf·nL/c)` uses only the *offset* Δf;
  - susceptibility `−1/(m(2πf)²)`, so precompute `1/(2πf_k)²` once and divide by the mass;
  - `f_correct = exp(iφ·f_s/f₀)`, where `f_s/f₀` is about 1e-11.
  Keep all of these **exact**: the 1e-5 log tolerance and the "omitted calculations" rule rule out approximating them.
- **The Outlook's "dynamically choose which systems are necessary"**: not applicable here. The benchmark
  needs the carrier, signal and noise systems, and the full `(751,101)` signal output.
- **The PRX paper (UIFO)**: simulated with Finesse through PyKat, about 1.5 M CPU-hours in total, using BFGS
  without autodiff and 100 frequencies (*"If we would use less, … pathological solutions"*). There's no speed
  trick in it we can use. Its "simplification" step (snapping transmissions to 0/1, deleting spaces) runs
  *after* optimisation and changes the physics, so it's **not allowed** here: we must keep all 193 gradients.
- **Frequency-sweep literature** (rational Krylov / model order reduction, Woodbury updates): approximate,
  or not low-rank for this topology. Not usable (details in the research notes).

## Likely common vs. differentiating
Other teams use the same AI tools and will likely get similar suggestions. Expect most teams to do T1, T3
and the lazy imports. **Differentiators:** T2a + T2 combined (exploiting physics structure), T14 (noise
identity, a cheap win most teams will miss), T11 (hand-written adjoint), T4 with sparse noise, T2c, and
careful phase-by-phase profiling. The score is the
*first-run total*, so compile time counts as much as runtime. A team with a faster solver but a 20 s compile still loses.

## T8. Submission
- [ ] Before/after JSON reports and plots (`bench-baseline.json/png`, `bench-final.json/png`).
- [ ] Short write-up in `docs/notes/APPROACH.md`: what changed, why it helps on one SPR core,
      correctness evidence (benchmark PASS + `compare_arrays.py` output).
- [ ] Exact reproduction commands.

---

### Suggested order and split
1. **T0** (everyone needs it), then **T1** (quick win, low risk).
2. In parallel: **T3** (compile time, person A) and **T2a → T2** (solver, person B).
   They touch different parts of `simulate.py`, so agree on the static-plan format first.
3. **T4 + T5** (they build on T3's plan), then **T7** (anytime, small).
4. **T11** (adjoint) once the forward structure is stable. **T2c / T6 / T9 / T10** only if time remains.

---

## Optional: micro-optimisations (each roughly 10–100 ms, unmeasured)
Do these only after the big items, and keep any that measurably help. With the baseline at about 44 s
they don't matter now, but if the big items get the total down to a few seconds, a handful of these
adds up to a real share. Estimates are guesses; measure each.

**Startup / imports / initialisation**
- [ ] Lazy-import `plotly` (T7). Probably more than 50 ms, confirm with `python -X importtime -c "import differometor"`.
- [ ] Check `-X importtime` for anything else pulled in by `differometor` (e.g. `optax`, `scipy` submodules,
      `matplotlib` via `setups.py`/`plot.py`). Import only what `simulate` needs.
- [ ] Don't call `jax.config.update(...)` at module import in `simulate.py` if the benchmark already sets it (free, tiny).
- [ ] Avoid module-level work at import: building global tables, `FUNCTIONS.index(...)` lookups in loops
      (use a precomputed dict).

**Build phase (currently 1.0 s)**
- [ ] Use NumPy everywhere in build and convert to JAX **once** at the end. Today about 25 separate
      `jnp.array(...)` calls (build.py L1424–1439, L1637–1676) each do a host→device transfer and dispatch.
      Return NumPy arrays: `jit` accepts them directly.
- [ ] Replace per-edge eager `space(...)` (L1288) with NumPy (T7).
- [ ] `parameter_names.index` → dict (L1414). Same for any `list.index` / `in list` inside loops in `build`/`pairs_to_arrays`.
- [ ] `Setup.from_data` / `constrain_inter_grid_cell_spaces` (setups.py L513, L1082): profile, and remove
      deep copies or graph rebuilds if they show up.
- [ ] Build the matrices directly in the final layout (separate RHS, eliminated input ports, block order),
      so `simulate` never reshuffles.

**Compile size (each op removed shrinks forward *and* backward programs)**
- [ ] Use int32 instead of int64 for index arrays (x64 makes them int64 by default). Smaller constants,
      cheaper gathers and scatters.
- [ ] Remove `astype(jnp.complex128)` in `solve` (L57–58) when inputs are already complex128.
- [ ] Skip `expand_parameters` for the carrier when `carrier_changing_values` has one column (true in the benchmark).
- [ ] Replace `lax.scatter_mul` for the homodyne-detector phases (L345) with a static gather/multiply once the indices are static.
- [ ] Don't keep `parameters` as complex128 (`build.py` L1424) unless needed. Real-valued parameter maths
      is cheaper and makes simpler gradients; cast only the outputs of component functions.
- [ ] Merge all scatters into one per matrix (also T3/T5): fewer HLO scatter ops, faster compile.

**Memory / cache layout**
- [ ] Sort scatter indices row-major at build time (T3), so writes are sequential.
- [ ] Separate RHS from matrix (T5): no strided `[:, :, :-1]` copies.
- [ ] Contiguous batch layout `(F, N, N)` with the frequency axis outermost (already the case; keep it after refactors).
- [ ] **Lower peak memory = fewer page faults.** First touch of fresh memory costs about 0.25 µs per 4 KB page,
      so the 14 GiB baseline peak is about 3.7 M pages, roughly 1 s just in page faults. Every GB of peak
      removed saves about 60 ms (T6, T11, T2a).
- [ ] Avoid materialised `jnp.tile` / `jnp.broadcast_to` (L356, L371). Rely on broadcasting in the arithmetic.
- [ ] Noise readout (L410): `jnp.einsum` / `vdot` instead of `sum(real(a * conj(b)))`, so the product isn't materialised.

**Execution**
- [ ] Merge small elementwise steps after the solves (sqrt, constants `UNIT_VACUUM * H_PLANCK * F0 * 0.25`) into
      one precomputed constant multiplier.
- [ ] Check host transfer: outputs are small ((751,101) complex), so nothing to gain unless something large is returned by accident.
- [ ] Check that no Python-side `float(...)` / `.item()` / `print` on traced values sits in the hot path
      (each forces a device sync).
