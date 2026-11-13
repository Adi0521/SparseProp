# Spike histograms for Julia (SparseProp) vs Brian2 on the same network:
# population spike counts over time, per-neuron firing rates, inter-spike intervals,
# and the Brian2 - Julia spike-time error of matched spikes.
#
# usage: python plot_spike_hists.py [comparison/nodelay] [--dts 1e-5 1e-6] [--bin 0.005] [--save hists.png]
# (run `julia export_for_brian.jl` and `python compare.py` first)

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

MATCH_TOL = 1e-3  # same as compare.py: a k-th spike pair more than 1 ms apart is not the same spike

parser = argparse.ArgumentParser()
parser.add_argument("case_dir", nargs="?", default="comparison/nodelay")
parser.add_argument("--dts", nargs="+", type=float, default=[1e-5, 1e-6], help="Brian2 time steps to overlay")
parser.add_argument("--bin", type=float, default=0.005, help="time bin of the population histogram (s)")
parser.add_argument("--save", default=None, help="write the figure to this file instead of showing it")
args = parser.parse_args()

params = {name: float(value) for name, value in np.loadtxt(os.path.join(args.case_dir, "params.csv"), delimiter=",", skiprows=1, dtype=str)}
num_neurons, duration = int(params["num_neurons"]), params["duration"]


def load_spikes(name):
    return np.loadtxt(os.path.join(args.case_dir, name), delimiter=",", skiprows=1, ndmin=2)


def per_neuron(spikes):
    return [spikes[spikes[:, 0] == i, 1] for i in range(1, num_neurons + 1)]


def matched_errors(julia_trains, brian_trains):
    """Brian2 - Julia time of each neuron's k-th spike, for pairs within MATCH_TOL."""
    err = np.concatenate([tb[:min(len(tj), len(tb))] - tj[:min(len(tj), len(tb))] for tj, tb in zip(julia_trains, brian_trains)])
    return err[np.abs(err) <= MATCH_TOL], np.count_nonzero(np.abs(err) > MATCH_TOL)


runs = {"Julia (exact)": load_spikes("spikes_julia.csv")}
runs.update({f"Brian2 dt = {dt:g} s": load_spikes(f"spikes_brian_dt{dt:g}.csv") for dt in args.dts})
trains = {label: per_neuron(s) for label, s in runs.items()}

time_bins = np.arange(0.0, duration + args.bin, args.bin)
rates = {label: np.array([len(t) for t in tr]) / duration for label, tr in trains.items()}
rate_bins = np.arange(0, max(r.max() for r in rates.values()) + 2) - 0.5
isis = {label: np.concatenate([np.diff(t) for t in tr]) for label, tr in trains.items()}
isi_bins = np.linspace(0, max(i.max() for i in isis.values()), 40)

colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
style = {label: dict(color="0.6", alpha=0.5) if k == 0 else dict(histtype="step", color=colors[k - 1], lw=1.2)
         for k, label in enumerate(runs)}

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
ax = axes[0, 0]
for label, s in runs.items():
    ax.hist(s[:, 1], bins=time_bins, label=f"{label} ({len(s)} spikes)", **style[label])
ax.set_xlabel("Time (s)")
ax.set_ylabel(f"Spikes per {args.bin * 1e3:g} ms bin")
ax.set_title("Population spike histogram")
ax.legend(fontsize=8)

ax = axes[0, 1]
for label, r in rates.items():
    ax.hist(r, bins=rate_bins, label=f"{label} (mean {r.mean():.2f} Hz)", **style[label])
ax.set_xlabel("Firing rate (Hz)")
ax.set_ylabel("Number of neurons")
ax.set_title("Per-neuron firing rate")
ax.legend(fontsize=8)

ax = axes[1, 0]
for label, i in isis.items():
    ax.hist(i, bins=isi_bins, label=label, **style[label])
ax.axvline(params["mem_time_const"] * np.log(params["v_drive"] / (params["v_drive"] - params["v_threshold"])),
           color="k", ls=":", lw=0.8, label="free period (min ISI)")
ax.set_xlabel("Inter-spike interval (s)")
ax.set_ylabel("Count")
ax.set_title("Inter-spike intervals")
ax.legend(fontsize=8)

ax = axes[1, 1]
for k, dt in enumerate(args.dts):
    err, n_unmatched = matched_errors(trains["Julia (exact)"], trains[f"Brian2 dt = {dt:g} s"])
    ax.hist(err * 1e6, bins=60, histtype="step", color=colors[k], lw=1.2,
            label=f"dt = {dt:g} s: median {np.median(err) * 1e6:.1f} µs, {n_unmatched} unmatched")
ax.set_xlabel("Brian2 − Julia spike time (µs)")
ax.set_ylabel("Matched spikes")
ax.set_title(f"Spike-time error (k-th spike of each neuron, |Δt| ≤ {MATCH_TOL * 1e3:g} ms)")
ax.legend(fontsize=8)

fig.suptitle(os.path.basename(os.path.normpath(args.case_dir)))
fig.tight_layout()

if args.save:
    fig.savefig(args.save, dpi=130)
else:
    plt.show()
