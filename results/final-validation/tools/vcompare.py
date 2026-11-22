"""Compare two saved-array files (optimised vs official reference) array by array.

    python vcompare.py OPT.npz REF.npz [label]

Errors reported per array:
  max_abs            max |opt - ref|
  max_abs/scale      max |opt - ref| / max |ref|   (normwise; robust to near-zero entries)
  max_rel(>1e-6 sc)  max elementwise |opt - ref| / |ref| over entries with |ref| > 1e-6 * max|ref|
  max_rel_all        elementwise over all nonzero entries (dominated by near-zero entries)
plus max |dlog sensitivity| and the gradient's relative 2-norm error and cosine.
"""
import json
import sys

import numpy as np

opt, ref = np.load(sys.argv[1]), np.load(sys.argv[2])
label = sys.argv[3] if len(sys.argv) > 3 else ""
rows = {}
for key in sorted(set(opt.files) & set(ref.files)):
    a, b = np.asarray(opt[key]), np.asarray(ref[key])
    row = {"shape_opt": list(a.shape), "shape_ref": list(b.shape), "dtype_opt": str(a.dtype), "dtype_ref": str(b.dtype)}
    if a.shape != b.shape:
        row["error"] = "shape mismatch"
        rows[key] = row
        continue
    d = np.abs(a - b)
    scale = float(np.max(np.abs(b))) if b.size else 0.0
    row.update({
        "finite_opt": bool(np.all(np.isfinite(a))), "finite_ref": bool(np.all(np.isfinite(b))),
        "bit_identical": bool(np.array_equal(a, b)), "max_abs": float(d.max()) if d.size else 0.0,
        "max_abs_over_scale": float(d.max() / scale) if scale > 0 else 0.0,
    })
    big = np.abs(b) > 1e-6 * scale
    row["max_rel_entries_above_1e-6_scale"] = float(np.max(d[big] / np.abs(b[big]))) if np.any(big) else 0.0
    nz = np.abs(b) > 0
    row["max_rel_all_nonzero"] = float(np.max(d[nz] / np.abs(b[nz]))) if np.any(nz) else 0.0
    rows[key] = row
summary = {"label": label, "arrays": rows}
if "sensitivity" in rows:
    summary["max_abs_dlog_sensitivity"] = float(np.max(np.abs(np.log(opt["sensitivity"]) - np.log(ref["sensitivity"]))))
if "gradient" in rows:
    g, h = np.asarray(opt["gradient"]), np.asarray(ref["gradient"])
    summary["gradient_rel_2norm_error"] = float(np.linalg.norm(g - h) / np.linalg.norm(h))
    summary["gradient_cosine"] = float(g @ h / (np.linalg.norm(g) * np.linalg.norm(h)))
    summary["gradient_length"] = int(g.shape[0])
print(json.dumps(summary, indent=2))
