# Overlay exact Julia (SparseProp) voltage traces and every-step Brian2 traces for the same neurons.
#
# usage: python plot_traces.py [comparison/nodelay] [--dts 1e-5 1e-6] [--t-start 0] [--t-window 0.3] [--save traces.png]
# (run `julia export_for_brian.jl` and `python brian2_compare.py <case> --dt <dt>` first; compare.py does both dts)

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("case_dir", nargs="?", default="comparison/nodelay")
parser.add_argument("--dts", nargs="+", type=float, default=[1e-5], help="Brian2 time steps to overlay")
parser.add_argument("--t-start", type=float, default=0.0, help="start of the plotted window (s)")
parser.add_argument("--t-window", type=float, default=0.3, help="end of the plotted window (s)")
parser.add_argument("--save", default=None, help="write the figure to this file instead of showing it")
args = parser.parse_args()

params = {name: float(value) for name, value in np.loadtxt(os.path.join(args.case_dir, "params.csv"), delimiter=",", skiprows=1, dtype=str)}
v_drive, tau = params["v_drive"], params["mem_time_const"]

julia = np.loadtxt(os.path.join(args.case_dir, "traces_julia.csv"), delimiter=",", skiprows=1, ndmin=2)
trace_ids = np.unique(julia[:, 0]).astype(int)
brian = {dt: np.loadtxt(os.path.join(args.case_dir, f"traces_brian_dt{dt:g}.csv"), delimiter=",", skiprows=1, ndmin=2) for dt in args.dts}


def exact_curve(events, t_end, points_per_segment=20):
    """Julia trace between logged events: free relaxation v = I - (I - v0) exp(-(t - t0)/tau)."""
    events = events[events[:, 0] < t_end]
    bounds = np.append(events[:, 0], t_end)
    t, v = [], []
    for (t0, v0), t1 in zip(events, bounds[1:]):
        ts = np.linspace(t0, t1, points_per_segment)
        t.append(ts)
        v.append(v_drive - (v_drive - v0) * np.exp(-(ts - t0) / tau))
    return np.concatenate(t), np.concatenate(v)


colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
fig, axes = plt.subplots(len(trace_ids), 1, figsize=(11, 2.2 * len(trace_ids)), sharex=True, squeeze=False)
for ax, nid, col in zip(axes[:, 0], trace_ids, range(1, len(trace_ids) + 1)):
    t, v = exact_curve(julia[julia[:, 0] == nid, 1:], args.t_window)
    ax.plot(t, v, color="k", lw=1.2, label="Julia (exact)")
    for k, dt in enumerate(args.dts):
        b = brian[dt][(brian[dt][:, 0] >= args.t_start) & (brian[dt][:, 0] < args.t_window)]
        ax.plot(b[:, 0], b[:, col], color=colors[k], lw=0.8, ls="--", label=f"Brian2 dt = {dt:g} s")
    ax.axhline(params["v_threshold"], color="gray", ls=":", lw=0.6)
    ax.set_ylabel(f"v (neuron {nid})")
axes[0, 0].legend(loc="lower right", fontsize=8)
axes[0, 0].set_title(os.path.basename(os.path.normpath(args.case_dir)))
axes[-1, 0].set_xlim(args.t_start, args.t_window)
axes[-1, 0].set_xlabel("Time (s)")
fig.tight_layout()

if args.save:
    fig.savefig(args.save, dpi=130)
else:
    plt.show()
