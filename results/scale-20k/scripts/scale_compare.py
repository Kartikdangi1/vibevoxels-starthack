import json, numpy as np
O = "/root/validation/scale/"
for c in ["dev", "aligo"]:
    big, so, sr = (np.load(O + f"{c}_{x}.npz") for x in ("20k", "sub_opt", "sub_ref"))
    assert np.array_equal(big["frequencies"][::200], sr["frequencies"])
    d_big = np.abs(np.log(big["sensitivity"][::200]) - np.log(sr["sensitivity"]))
    d_sub = np.abs(np.log(so["sensitivity"]) - np.log(sr["sensitivity"]))
    g, h = so["gradient"], sr["gradient"]
    worst = int(np.argmax(d_big))
    print(f"{c}: 20k run vs official at {len(sr['frequencies'])} shared freqs: max|dlogS| {d_big.max():.2e} (worst at {sr['frequencies'][worst]:.0f} Hz), median {np.median(d_big):.1e}")
    print(f"     same 100-freq sample, ours vs official: max|dlogS| {d_sub.max():.2e}; gradient rel2 {np.linalg.norm(g-h)/np.linalg.norm(h):.2e}, cosine {g@h/np.linalg.norm(g)/np.linalg.norm(h):.10f}")
    print(f"     20k run reproduces our own 100-freq run: max|dlogS| {np.max(np.abs(np.log(big['sensitivity'][::200]) - np.log(so['sensitivity']))):.1e}")
    for x in ("20k", "sub_opt", "sub_ref"):
        r = json.load(open(O + f"{c}_{x}.json"))
        print(f"     {x:8s} f={r['frequencies']:6d} first {r['seconds_total_first']:7.2f} s  warm {min(r['seconds_warm']):7.3f} s  peak {r['peak_rss_gib']:6.2f} GiB  finite {r['finite']} grad nonzero {r['gradient_nonzero']}/{r['gradient_length']}")
