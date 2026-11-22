"""Large frequency sweeps: full simulation + sensitivity + gradient at N frequencies.

    cd <code dir>; python scale.py CASE N FMIN FMAX OUT_PREFIX [--subset K]

CASE: dev (official 3x3 grid, 193 vars) or aligo (official aLIGO, 29 vars). Frequencies are N
log-spaced points FMIN..FMAX. The loss is the official readout with reference 1 (mean log10 S);
the gradient is jax.value_and_grad of it. With --subset K only every K-th frequency is run (used
to evaluate the official code on a sample of the same grid). Writes OUT_PREFIX.json/.npz.
"""
import json
import os
import resource
import sys
import time
from pathlib import Path

T0 = time.perf_counter()
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vcase  # noqa: E402  (imports jax with x64, differometor from cwd)

jax, jnp, np, df = vcase.jax, vcase.jnp, vcase.np, vcase.df


def build(case, frequencies):
    from copy import deepcopy
    if case == "aligo":
        setup, _ = vcase.aligo(mode="space_modulation")
        fixture = deepcopy(setup.to_data())
        pairs = [list(p) for p in setup.parameters if p[1] in vcase.PROPERTIES]
    else:
        fixture = json.loads((vcase.DATA / "uifo_800_3000.json").read_text())
        setup = df.Setup.from_data(fixture)
        pairs = vcase.constrain_inter_grid_cell_spaces(setup.parameters, vcase.PROPERTIES)
    initial = []
    for pair in pairs:
        members = pair if isinstance(pair[0], list) else [pair]
        initial.append(fixture["nodes" if members[0][0] in fixture["nodes"] else "edges"][members[0][0]]["properties"][members[0][1]])
    parameters = jnp.asarray(initial, dtype=jnp.float64)
    f = jnp.asarray(frequencies)
    arrays, ports, *_ = df.run_build_step(setup, [("f", "frequency")], f, pairs)
    arrays["optimized_parameters"] = parameters
    return arrays, jnp.asarray(ports), parameters, pairs


def main():
    case, n, fmin, fmax, out = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
    k = int(sys.argv[sys.argv.index("--subset") + 1]) if "--subset" in sys.argv else 1
    frequencies = np.geomspace(fmin, fmax, n)[::k]
    t_imports = time.perf_counter() - T0

    t = time.perf_counter()
    arrays, ports, parameters, pairs = build(case, frequencies)
    t_build = time.perf_counter() - t
    expected = jnp.ones(frequencies.shape[0], dtype=jnp.float64)

    def full(p):
        outputs = df.simulate(**dict(arrays, optimized_parameters=p))
        loss, (sens, powers) = vcase.readout(outputs, ports, expected)
        return loss, (sens, powers)

    fn = jax.jit(jax.value_and_grad(full, has_aux=True))
    t = time.perf_counter()
    (loss, (sens, powers)), grad = fn(parameters)
    sens, grad = np.asarray(jax.device_get(sens)), np.asarray(jax.device_get(grad))
    t_first = time.perf_counter() - t
    warm = []
    for _ in range(3 if n <= 5000 or k > 1 else 2):
        t = time.perf_counter()
        (loss, (s2, _)), g2 = fn(parameters)
        jax.block_until_ready((s2, g2))
        warm.append(time.perf_counter() - t)
    record = {
        "case": case, "frequencies": int(frequencies.shape[0]), "f_min_hz": float(frequencies[0]), "f_max_hz": float(frequencies[-1]),
        "design_variables": int(parameters.shape[0]), "gradient_length": int(grad.shape[0]),
        "seconds_imports": t_imports, "seconds_build": t_build,
        "seconds_first_compile_and_run": t_first, "seconds_total_first": t_imports + t_build + t_first,
        "seconds_warm": warm, "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20,
        "finite": bool(np.all(np.isfinite(sens)) and np.all(np.isfinite(grad)) and np.isfinite(float(loss))),
        "sensitivity_positive": bool(np.all(sens > 0)), "gradient_nonzero": int(np.count_nonzero(grad)),
        "loss": float(loss),
    }
    np.savez(out + ".npz", frequencies=frequencies, sensitivity=sens, gradient=grad, loss=np.asarray(loss))
    Path(out + ".json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record))


if __name__ == "__main__":
    main()
