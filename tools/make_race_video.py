#!/usr/bin/env python3
"""Render a side-by-side "race" video from two race_optimizer.py logs (measured timestamps).

    python tools/make_race_video.py results/race/baseline.jsonl results/race/final.jsonl \
        --out docs/pitch/race_100_steps.mp4

Timeline: real time (1x) until the fast run has finished plus a short pause, then a constant
fast-forward (shown on screen as "fast-forward xN") until the slow run finishes, then a summary.
Every number on screen comes from the logs: steps done at wall time t = steps whose timestamp is
<= t. Needs matplotlib and imageio-ffmpeg (bundled ffmpeg binary).
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np

SLOW_C, FAST_C = "#E69F00", "#0072B2"  # Okabe-Ito orange / blue (identity, not status)
INK, MUTED, TRACK = "#222222", "#6B7280", "#E5E7EB"


def load(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    steps = [r for r in rows if "step" in r]
    return dict(label=steps[0]["label"], t=np.array([r["t"] for r in steps]),
                loss=np.array([r["loss"] for r in steps]))


def clock(t):
    m, s = divmod(t, 60)
    return f"{int(m):d} min {s:04.1f} s" if m else f"{s:.1f} s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("slow")
    ap.add_argument("fast")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--ff-seconds", type=float, default=25.0, help="video length of the fast-forward part")
    ap.add_argument("--slow-name", default="Original Differometor")
    ap.add_argument("--fast-name", default="Our final build")
    ap.add_argument("--machine", default="AMD EPYC 9334, 1 CPU core")
    a = ap.parse_args()
    import imageio_ffmpeg
    plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    plt.rcParams["font.family"] = "DejaVu Sans"

    slow, fast = load(a.slow), load(a.fast)
    n = len(slow["t"])
    assert len(fast["t"]) == n, "both logs must have the same number of steps"
    T_fast, T_slow = fast["t"][-1], slow["t"][-1]
    A = T_fast + 2.5                                     # end of the real-time part (wall seconds)
    k = max(1, int(round((T_slow + 1 - A) / a.ff_seconds / 10.0)) * 10)  # fast-forward factor, multiple of 10
    title_s, end_s = 2.5, 6.0
    ff_s = (T_slow + 1 - A) / k
    total_s = title_s + A + ff_s + end_s

    def wall(tau):  # video time -> wall time
        if tau < title_s:
            return 0.0, "title"
        tau -= title_s
        if tau < A:
            return tau, "real"
        tau -= A
        if tau < ff_s:
            return A + k * tau, "ff"
        return T_slow + 1, "end"

    fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
    fig.patch.set_facecolor("white")
    fig.text(0.05, 0.93, "100 design-optimisation steps of a gravitational-wave detector", fontsize=30,
             fontweight="bold", color=INK)
    fig.text(0.05, 0.885, f"193 design parameters · 101 frequencies · gradient + Adam each step · {a.machine} · "
             "measured, not simulated", fontsize=16, color=MUTED)
    lanes = fig.add_axes([0.05, 0.47, 0.90, 0.30])
    lanes.set_xlim(0, 1); lanes.set_ylim(0, 1); lanes.axis("off")
    loss_ax = fig.add_axes([0.07, 0.08, 0.60, 0.33])
    side = fig.add_axes([0.70, 0.08, 0.27, 0.33]); side.axis("off")

    # lanes: name, track, bar, status
    lane_y = {"slow": 0.68, "fast": 0.22}
    bars, status, names = {}, {}, {}
    for key, name, col in (("slow", a.slow_name, SLOW_C), ("fast", a.fast_name, FAST_C)):
        y = lane_y[key]
        names[key] = lanes.text(0.0, y + 0.17, name, fontsize=22, fontweight="bold", color=INK)
        lanes.add_patch(plt.Rectangle((0, y - 0.07), 1.0, 0.14, color=TRACK, lw=0))
        bars[key] = lanes.add_patch(plt.Rectangle((0, y - 0.07), 0.0, 0.14, color=col, lw=0))
        status[key] = lanes.text(1.0, y + 0.17, "", fontsize=19, color=INK, ha="right")
    clock_txt = fig.text(0.05, 0.80, "", fontsize=30, fontweight="bold", color=INK)
    mode_txt = fig.text(0.95, 0.805, "", fontsize=22, color=MUTED, ha="right")

    # loss panel: both runs follow the same optimisation trajectory
    loss_ax.set_xlim(0, n + 1)
    lo, hi = min(slow["loss"].min(), fast["loss"].min()), max(slow["loss"].max(), fast["loss"].max())
    pad = 0.06 * (hi - lo)
    loss_ax.set_ylim((lo - pad) * 1e3, (hi + pad) * 1e3)
    loss_ax.set_xlabel("optimisation step", fontsize=15, color=INK)
    loss_ax.set_ylabel("loss  (×10⁻³, lower is better)", fontsize=15, color=INK)
    loss_ax.set_title("Same physics: the loss curves coincide", fontsize=17, color=INK, loc="left")
    for s in ("top", "right"):
        loss_ax.spines[s].set_visible(False)
    loss_ax.grid(alpha=0.25, ls="--")
    loss_ax.tick_params(labelsize=12)
    (l_slow,) = loss_ax.plot([], [], color=SLOW_C, lw=6, alpha=0.55, label=a.slow_name, solid_capstyle="round")
    (l_fast,) = loss_ax.plot([], [], color=FAST_C, lw=2, label=a.fast_name)
    loss_ax.legend(loc="upper right", fontsize=13, frameon=False)
    side_txt = side.text(0.0, 0.95, "", fontsize=19, color=INK, va="top", linespacing=1.6)

    steps_x = np.arange(1, n + 1)
    max_dloss = float(np.max(np.abs(slow["loss"] - fast["loss"])))

    def draw(frame):
        t, mode = wall(frame / a.fps)
        for key, run in (("slow", slow), ("fast", fast)):
            done = int(np.searchsorted(run["t"], t, side="right"))
            bars[key].set_width(done / n)
            if done == n:
                status[key].set_text(f"✓ 100 steps in {clock(run['t'][-1])}")
                status[key].set_fontweight("bold")
            elif done == 0:
                status[key].set_text("importing, building, compiling…" if t > 0 else "ready")
            else:
                status[key].set_text(f"step {done}/{n}")
        ds = int(np.searchsorted(slow["t"], t, side="right"))
        df_ = int(np.searchsorted(fast["t"], t, side="right"))
        l_slow.set_data(steps_x[:ds], slow["loss"][:ds] * 1e3)
        l_fast.set_data(steps_x[:df_], fast["loss"][:df_] * 1e3)
        clock_txt.set_text(f"elapsed  {clock(min(t, T_slow))}")
        mode_txt.set_text({"title": "starting both at the same moment", "real": "real time (1×)",
                           "ff": f"▶▶ fast-forward ×{k}", "end": "finished"}[mode])
        mode_txt.set_color(SLOW_C if mode == "ff" else MUTED)
        mode_txt.set_fontweight("bold" if mode == "ff" else "normal")
        if mode == "end":
            side_txt.set_text(f"{a.slow_name}:\n   {clock(T_slow)}\n{a.fast_name}:\n   {clock(T_fast)}\n"
                              f"→ {T_slow / T_fast:.0f}× faster\n"
                              f"max loss difference: {max_dloss:.1e}")
        elif df_ == n:
            side_txt.set_text(f"{a.fast_name}\nfinished all 100 steps\nin {clock(T_fast)}.\n\n"
                              f"{a.slow_name}\nis at step {ds}.")
        else:
            side_txt.set_text("")
        return []

    frames = int(total_s * a.fps)
    anim = animation.FuncAnimation(fig, draw, frames=frames, interval=1000 / a.fps, blit=False)
    anim.save(a.out, writer=animation.FFMpegWriter(fps=a.fps, bitrate=6000,
                                                    extra_args=["-pix_fmt", "yuv420p"]))
    print(f"wrote {a.out}: {total_s:.1f} s, {frames} frames, fast-forward x{k}; "
          f"slow {T_slow:.1f} s, fast {T_fast:.2f} s, ratio {T_slow / T_fast:.1f}, max |dloss| {max_dloss:.2e}")


if __name__ == "__main__":
    main()
