# Topology-cache prototype results

## Cold-input requirement

The topology cache must help the first random input in a fresh process. A cache populated by a prior run does not count for that use case. The random UIFO generator has a finite set of local interior-cell motifs (center component, orientation, and modulation mode), even though the full grid configuration varies.

## Prototype tested

I prototyped an in-memory cache of normalized cell graph records. On the first occurrence of a motif it built a template; later cells cloned its nodes, edges, and parameter names with the current coordinates substituted. The benchmark cleared the cache before each trial, then built the three size-3 UIFO setups used by `examples/uifo_optimization.py` (space, amplitude, and frequency modulation) with random seed 42. Cached and uncached output graphs matched exactly.

| Cold-cache setup construction | Median |
|---|---:|
| Direct Setup API calls | 1.414 ms |
| Build template on first use, clone later cells | 2.862 ms |

The prototype was about 2.0× slower, so it was removed instead of adding a first-run regression. This part of setup generation is only milliseconds; caching these Python graph records cannot materially reduce the roughly 40-second simulation/gradient workload.

## Better cache boundary

For a useful cold-start speedup, cache **symbolic solver templates**, not the Python graph objects. A template could hold a canonical motif's port ordering, sparsity pattern, elimination order, and scatter maps. Per-input reflectivities, lengths, phases, masses, and other physical values must still be applied at runtime; numerical LU factors cannot be reused when those values change. The assembled global solve is the expensive part, so realizing a payoff likely requires the block/Schur solver work in T2/T2a, with each subtopology represented as a reusable boundary-to-boundary solve.

Because the first input is random, a process-local cache alone is insufficient for motifs that do not repeat in that one input. Any compiled-template cache should be generated ahead of the run for the finite motif grammar, versioned by component/function definitions and dtype/backend, and included in the cold-start benchmark's load time. If topology can be arbitrary outside a known grammar, misses must fall back to the normal builder and the benchmark must include that cost.
