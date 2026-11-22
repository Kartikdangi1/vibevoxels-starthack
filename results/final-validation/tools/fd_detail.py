import json, numpy as np
A = "/root/validation/artifacts/"
for c in ["dev", "uifo20", "voyager_space"]:
    fd = json.load(open(A + f"fd/{c}_opt_fd.json")); o = np.load(A + f"vcase/{c}_opt.npz"); r = np.load(A + f"vcase/{c}_ref.npz"); gn = fd["gradient_inf_norm"]
    print(f"== {c}  |grad|_inf {gn:.3e}")
    for k in fd["checks"]:
        i = k["index"]
        print(f"  {k['parameter']:24s} p={o['parameters'][i]: .4e} ad_opt={k['autodiff']: .6e} ad_ref={r['gradient'][i]: .6e}  fd_best_rel={k['best_rel_err']:.1e} fd_abs/|g|inf={k['best_abs_err'] / gn:.1e}")
d = json.load(open(A + "fd/dev_opt_fd.json")); k = [k for k in d["checks"] if k["parameter"] == "boundary01.power"][0]
for s in k["steps"]:
    print(f"   h={s['step']:.3e} fd={s['central_difference']: .6e} rich={s.get('richardson', float('nan')): .6e}")
