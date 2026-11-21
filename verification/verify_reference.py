"""End-to-end acceptance check against the PUBLIC reference data (the judging ground truth).

  * input fixtures are the unmodified public files (SHA-256)
  * full 101-frequency sensitivity matches the upstream reference to atol 1e-5 in log-space
    (the official criterion), reported with the margin and the upstream code's own error floor
  * the aLIGO example matches its stored reference curve
  * a fresh subprocess run reproduces the same numbers (no state leaking between runs)
  * the in-graph JAX path reproduces the same public sensitivity (second implementation)

    python verification/verify_reference.py [--n-freq 101]
"""
import argparse
import hashlib
import json
import subprocess
import sys

from common import DATA, ROOT, Report, aligo, df, jnp, load_public, np, set_backend, uifo_loss

FIXTURE_SHA256 = "ff4ba353a2a03cdce7f3e61bad729c2e00888d842c28ba9cf699f95afb87e138"
REFERENCE_SHA256 = "726f2cfe6986cb991fa48d5a28127b9330c3dc7e69d5bd193c8748acef03adc3"
R = Report("public reference acceptance")


def sensitivities(n_freq):
    case = load_public(n_freq)
    _, sens = uifo_loss(case["params"], case["arrays"], case["ports"], jnp.asarray(case["ref"]))
    return np.asarray(sens), case


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-freq", type=int, default=None, help="subset size (default: all 101)")
    ap.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.child:  # used by the fresh-process reproducibility test
        s, _ = sensitivities(args.n_freq)
        print("SENS_JSON=" + json.dumps(s.tolist()))
        return 0

    R.check("fixture SHA-256 unchanged", hashlib.sha256((DATA / "uifo_800_3000.json").read_bytes()).hexdigest() == FIXTURE_SHA256)
    R.check("reference SHA-256 unchanged",
            hashlib.sha256((DATA / "uifo_800_3000_test_data.json").read_bytes()).hexdigest() == REFERENCE_SHA256)

    set_backend("host")
    sens, case = sensitivities(args.n_freq)
    ref = case["ref"]
    R.check("one positive sensitivity per frequency", sens.shape == ref.shape and np.all(sens > 0), f"{sens.shape}")
    err = np.abs(np.log(sens) - np.log(ref))
    R.check("max |log S - log S_ref| <= 1e-5 (official criterion)", err.max() <= 1e-5,
            f"max={err.max():.3e} at f={case['freqs'][int(err.argmax())]:.4g} Hz, margin x{1e-5 / max(err.max(), 1e-300):.1f}")
    R.check("mean |log S - log S_ref| <= 5e-6", err.mean() <= 5e-6,
            f"mean={err.mean():.3e} (upstream's own error floor is ~5.4e-6 max)")
    R.check("relative sensitivity error <= 1e-5 everywhere", np.max(np.abs(sens / ref - 1)) <= 1e-5,
            f"{np.max(np.abs(sens / ref - 1)):.3e}")

    set_backend("jax")
    sens_jax, _ = sensitivities(args.n_freq)
    R.check("jax-sparse path also meets 1e-5", np.max(np.abs(np.log(sens_jax) - np.log(ref))) <= 1e-5,
            f"{np.max(np.abs(np.log(sens_jax) - np.log(ref))):.3e}")
    R.close("host == jax-sparse sensitivity", np.log(sens), np.log(sens_jax), atol=1e-6)
    set_backend("host")

    # aLIGO example vs stored curve
    stored = json.loads((DATA / "aligo_test_data.json").read_text())
    S, opt = aligo()
    f = jnp.asarray(stored["frequencies"])
    carrier, signal, noise, ports, *_ = df.run(S, [("f", "frequency")], f)
    keys = [k for k in stored if k != "frequencies"]
    if keys:
        det = df.signal_detector(carrier, signal)[ports].squeeze()
        got = np.asarray(noise.squeeze() / jnp.abs(det))
        want = np.asarray(stored[keys[0]], dtype=float)
        if want.shape == got.shape:
            R.close(f"aLIGO sensitivity vs stored '{keys[0]}'", np.log(got), np.log(want), atol=1e-5)
        else:
            R.check("aLIGO stored key shape", False, f"{want.shape} vs {got.shape}")

    # fresh process reproduces the numbers
    cmd = [sys.executable, str(ROOT / "verification" / "verify_reference.py"), "--child"]
    if args.n_freq:
        cmd += ["--n-freq", str(args.n_freq)]
    out = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    line = next((ln for ln in out.stdout.splitlines() if ln.startswith("SENS_JSON=")), None)
    if line is None:
        R.check("fresh process run", False, out.stderr[-300:])
    else:
        fresh = np.asarray(json.loads(line[len("SENS_JSON="):]))
        R.check("fresh process reproduces sensitivity bit-for-bit", np.array_equal(fresh, sens),
                f"max diff {np.max(np.abs(fresh - sens)):.2e}")
    return R.finish()


if __name__ == "__main__":
    sys.exit(main())
