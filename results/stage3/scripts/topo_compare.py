import numpy as np, glob, os, sys, json
D = sys.argv[1]
def nrel(a, b): return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))
rows = []
for f in sorted(glob.glob(f"{D}/*.native.npz")):
    tag = os.path.basename(f)[:-len(".native.npz")]
    nat = np.load(f)
    ok = all(np.all(np.isfinite(nat[k])) for k in "csng") and np.all(nat["sens"] > 0)
    row = {"case": tag, "finite_positive": bool(ok), "grad_nonzero": bool(np.any(nat["g"] != 0)), "n_params": int(nat["g"].size)}
    for ref in ("traced", "jax"):
        r = np.load(f"{D}/{tag}.{ref}.npz")
        row[ref] = dict(carrier=nrel(nat["c"], r["c"]), signal=nrel(nat["s"], r["s"]),
                        sens_log=float(np.abs(np.log(nat["sens"]) - np.log(r["sens"])).max()),
                        grad=nrel(nat["g"], r["g"]))
    rows.append(row)
    t, j = row["traced"], row["jax"]
    print(f"{tag:32s} ok={ok} P={row['n_params']:3d} | vs traced: c {t['carrier']:.1e} s {t['signal']:.1e} dlogS {t['sens_log']:.1e} g {t['grad']:.1e}"
          f" | vs jax: c {j['carrier']:.1e} s {j['signal']:.1e} dlogS {j['sens_log']:.1e} g {j['grad']:.1e}")
json.dump(rows, open(f"{D}/topo_compare.json", "w"), indent=1)
