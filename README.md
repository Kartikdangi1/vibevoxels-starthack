# START Hack Munich 2026 — Feyer: Faster Physics

- **Event:** START Hack Munich, 21–22 November 2026
- **Challenge:** [Feyer · Faster Physics](challenge.pdf) ([judging criteria](judging-criteria.pdf))
- **Source:** https://github.com/feyerlabs/start-hack-munich-2026 (upstream setup/benchmark docs: [CHALLENGE_README.md](CHALLENGE_README.md))
- **Team:** TBD

## The challenge, in one line

Make a complete evaluation of **Differometor** (an open-source, JAX-based optical
interferometer simulator for gravitational-wave detector design) faster — setup, compilation,
forward simulation, sensitivity and gradients — while keeping the physics, all 101 frequencies,
all 193 design variables, double precision, and correct gradients exactly as they are.

No pitch, no demo, no business case here — it's pure "make this number smaller without
breaking correctness."

## What's timed vs. not

| Counts toward the timer | Excluded |
| --- | --- |
| imports, initialization, input prep, all compilation, forward simulation, sensitivity, gradients, result transfer | validation, benchmark reporting/plotting |

## Correctness bar (must hold after any change)

- All 101 frequencies, all 193 design variables, double precision, full setup kept.
- Sensitivities positive and within `abs(log(sensitivity) - log(reference)) <= 1e-5` at every frequency.
- Carrier, signal, noise, sensitivity, powers, loss and all gradients finite; gradient not identically zero.
- Gradients pass the supplied sanity check.

## Submission

- Code
- Before/after timing reports + plots (`benchmark.py` output)
- Reproduction commands
- Optional short note on approach/correctness checks (see [docs/notes](docs/notes))

Prizes: lowest complete-evaluation time, second-lowest, and a "most surprising solution" pick
by the Feyer team. AI coding tools are explicitly welcome.

## Environment

The upstream README asks for **Python 3.12 on Linux or WSL2** (native Windows isn't supported —
JAX/process-affinity bits lean on Linux). 32 GB RAM recommended, no GPU needed.

**Docker is not set up yet — planned for later** so we don't have to fight WSL2 distro setup by
hand; Docker Desktop is already running on this machine, so a container is the simpler path to a
matching Linux environment. Until then, follow [CHALLENGE_README.md](CHALLENGE_README.md) directly
(venv + pip, run inside WSL2).

Quick start (inside Linux/WSL2):

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
python start_here.py                 # sanity walkthrough
python benchmark.py --cpus 1 --output benchmark-baseline.json   # full, ~5 min
python benchmark.py --cpus 1 --repeats 3 --output benchmark-check.json  # quick correctness check
```

## Layout

| Path | Purpose |
| --- | --- |
| `differometor/` | The simulator package — this is what we optimize |
| `examples/` | Example setups (aLIGO, cavity, uIFO, Voyager, ...) |
| `benchmarks/` | Upstream CPU baseline (`cpu-baseline.json` / `.png`) |
| `media/` | Upstream docs/images (incl. `documentation.pdf`) |
| `docs/notes/` | Our own working notes, profiling findings, approach writeup |
| `benchmark.py`, `start_here.py`, `plot_timings.py` | Upstream entry points, unmodified |
| `challenge.pdf`, `challenge.docx`, `judging-criteria.pdf` | Official brief |
| `CHALLENGE_README.md` | Upstream setup/benchmark instructions, unmodified |

## Status

- [ ] Get a working Linux/WSL2 (or Docker) environment running `start_here.py`
- [ ] Run unmodified baseline benchmark, save `benchmark-baseline.json/.png`
- [ ] Profile to find the bottleneck
- [ ] Optimize `differometor/`, keep correctness checks passing
- [ ] Re-run benchmark, compare before/after
- [ ] Write the short approach note
- [ ] Set up Docker for a reproducible environment
