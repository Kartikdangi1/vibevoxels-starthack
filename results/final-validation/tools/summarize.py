import json
A = "/root/validation/artifacts/"
for f in ["compare/dev_official.json", "compare/aligo_official.json"] + [f"compare/{c}_vcase.json" for c in ["dev", "aligo", "uifo20", "voyager_space", "voyager_am", "voyager_fm"]]:
    d = json.load(open(A + f))
    print("==", d["label"], "| dlogS %.2e  grad rel2 %.2e  cos %.12f  len %d" % (
        d.get("max_abs_dlog_sensitivity", -1), d.get("gradient_rel_2norm_error", -1), d.get("gradient_cosine", -1), d.get("gradient_length", -1)))
    for k, r in d["arrays"].items():
        if k in ("parameters", "frequencies"):
            continue
        print(f"   {k:12s} {str(r['shape_opt']):11s} finite {r['finite_opt']}/{r['finite_ref']}  abs {r['max_abs']:.2e}  abs/scale {r['max_abs_over_scale']:.2e}  rel(>1e-6 scale) {r['max_rel_entries_above_1e-6_scale']:.2e}  bitident {r['bit_identical']}")
print()
for c in ["dev", "aligo", "uifo20", "voyager_space", "voyager_am", "voyager_fm"]:
    o = json.load(open(A + f"vcase/{c}_opt.json")); r = json.load(open(A + f"vcase/{c}_ref.json"))
    fd = json.load(open(A + f"fd/{c}_opt_fd.json"))
    worst = max(fd["checks"], key=lambda x: x["best_rel_err"])
    print(f"{c:14s} f={o['frequency_count']} vars={o['design_variables']} gradlen={o['gradient_length']} finite={all(o['finite'].values())} pos={o['sensitivity_positive']} nonzero={o['gradient_nonzero']} "
          f"| jitter dlogS opt {o['jitter_1e-12_max_abs_dlogS']:.1e} ref {r['jitter_1e-12_max_abs_dlogS']:.1e} | ref finite {all(r['finite'].values())} pos {r['sensitivity_positive']} "
          f"| FD n={len(fd['checks'])} worst {worst['parameter']} rel {worst['best_rel_err']:.1e} abs {worst['best_abs_err']:.1e} | pubref {o.get('max_abs_log_sensitivity_error_vs_public_reference', '-')}")
