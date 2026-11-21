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

**Use the Docker container — one command, works the same on Windows, macOS and Linux**, via Docker
Desktop. This sidesteps hand-installing a WSL distro on Windows or juggling a venv on everyone's own
machine:

```bash
# Windows (PowerShell)
.\container.ps1              # build (first time) + open a shell in the container
.\container.ps1 exec python start_here.py
.\container.ps1 exec python benchmark.py --cpus 1 --output benchmark-baseline.json

# macOS / Linux / WSL2
./container.bash              # build (first time) + open a shell in the container
./container.bash exec python start_here.py
./container.bash exec python benchmark.py --cpus 1 --output benchmark-baseline.json
```

See [container.bash](container.bash) / [container.ps1](container.ps1) for the full option list
(`-b`/`--build` to force a rebuild, `-s`/`--stop`, `-d`/`--delete`). Apple Silicon: works as-is; if a
pinned wheel ever lacks a `linux/arm64` build, add `platform: linux/amd64` under the `hack` service in
`docker/docker-compose.yaml` to build under Rosetta emulation.

Verified: the image builds cleanly with all pinned deps (jax 0.9.0, jaxlib 0.9.0, numpy 2.5.5, ...) on
Python 3.12. Running `start_here.py` end to end inside it is the next thing to confirm — see Status below.

**No Docker, or prefer it bare-metal?** Follow [CHALLENGE_README.md](CHALLENGE_README.md) directly
(venv + pip, run inside WSL2/Linux):

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

**Hackathon-wide resources (AMD GPU cloud, Token Factory, etc.)** — not needed for this challenge's
CPU-only benchmark, but documented in case it's useful: [docs/RESOURCES.md](docs/RESOURCES.md).

## Layout

| Path | Purpose |
| --- | --- |
| `differometor/` | The simulator package — this is what we optimize |
| `examples/` | Example setups (aLIGO, cavity, uIFO, Voyager, ...) |
| `benchmarks/` | Upstream CPU baseline (`cpu-baseline.json` / `.png`) |
| `media/` | Upstream docs/images (incl. `documentation.pdf`) |
| `docs/notes/` | Our own working notes, profiling findings, approach writeup |
| `docs/RESOURCES.md` | Hackathon-wide resources (AMD GPU cloud, Token Factory) |
| `docker/` | Dockerfile + compose file for the Linux dev container |
| `container.bash` / `container.ps1` | One-command container launcher (macOS/Linux/WSL2 / Windows) |
| `benchmark.py`, `start_here.py`, `plot_timings.py` | Upstream entry points, unmodified |
| `challenge.pdf`, `challenge.docx`, `judging-criteria.pdf` | Official brief |
| `CHALLENGE_README.md` | Upstream setup/benchmark instructions, unmodified |

## Status

- [x] Get a working Docker environment running `start_here.py` (also works bare-metal in WSL2/Linux)
- [ ] Run unmodified baseline benchmark, save `benchmark-baseline.json/.png`
- [ ] Profile to find the bottleneck
- [ ] Optimize `differometor/`, keep correctness checks passing
- [ ] Re-run benchmark, compare before/after
- [ ] Write the short approach note
