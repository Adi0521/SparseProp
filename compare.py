# Compare Julia (SparseProp, event-driven, exact) against Brian2 (clock-driven) on the same network,
# across Brian2 time steps. If the two implement the same model, Brian2's error should shrink and
# divergence should come later as dt -> 0.
#
# usage: python compare.py [--cases nodelay delay] [--dts 1e-4 1e-5 1e-6] [--rerun]
# (run `julia export_for_brian.jl` first)

import argparse
import os
import subprocess
import sys

import numpy as np

DIVERGE_TOL = 1e-3  # a matched spike more than 1 ms off means the trajectories have separated

parser = argparse.ArgumentParser()
parser.add_argument("--cases", nargs="+", default=["nodelay", "delay"])
parser.add_argument("--dts", nargs="+", type=float, default=[1e-4, 1e-5, 1e-6])
parser.add_argument("--rerun", action="store_true", help="rerun Brian2 even if its outputs are up to date")
args = parser.parse_args()


def load_csv(path):
    return np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)


def run_brian(case_dir, dt):
    """Run brian2_compare.py unless its outputs are newer than the Julia export."""
    spike_file = os.path.join(case_dir, f"spikes_brian_dt{dt:g}.csv")
    julia_file = os.path.join(case_dir, "spikes_julia.csv")
    if args.rerun or not os.path.exists(spike_file) or os.path.getmtime(spike_file) < os.path.getmtime(julia_file):
        subprocess.run([sys.executable, "brian2_compare.py", case_dir, "--dt", repr(dt)], check=True, stdout=subprocess.DEVNULL)


def compare_spikes(julia, brian, num_neurons):
    """Match each neuron's k-th Julia spike with its k-th Brian2 spike.
    Returns (neurons with different spike counts, divergence time, |dt| of matched spikes before divergence)."""
    count_mismatch, diverge_time, matched = 0, np.inf, []
    for i in range(1, num_neurons + 1):
        tj, tb = julia[julia[:, 0] == i, 1], brian[brian[:, 0] == i, 1]
        n = min(len(tj), len(tb))
        diff = np.abs(tb[:n] - tj[:n])
        off = np.nonzero(diff > DIVERGE_TOL)[0]
        if len(tj) != len(tb):
            count_mismatch += 1
        if len(off):  # first spike that is too far off
            diverge_time = min(diverge_time, tj[off[0]], tb[off[0]])
        elif len(tj) != len(tb):  # first extra spike in either simulation
            diverge_time = min(diverge_time, np.concatenate([tj[n:], tb[n:]]).min())
        matched.append(np.column_stack([tj[:n], diff]))
    matched = np.concatenate(matched)
    return count_mismatch, diverge_time, matched[matched[:, 0] < diverge_time, 1]


for case in args.cases:
    case_dir = os.path.join("comparison", case)
    params = {name: float(value) for name, value in np.loadtxt(os.path.join(case_dir, "params.csv"), delimiter=",", skiprows=1, dtype=str)}
    num_neurons, duration = int(params["num_neurons"]), params["duration"]
    julia_spikes = load_csv(os.path.join(case_dir, "spikes_julia.csv"))
    julia_v = load_csv(os.path.join(case_dir, "voltages_julia.csv"))

    print(f"\n{case}: {num_neurons} neurons, {duration:g} s, Julia {len(julia_spikes)} spikes"
          + (f" (delays are multiples of {params['dt']:g} s; coarser Brian2 dt rounds them)" if params["with_delays"] else ""))
    print(f"{'dt (s)':>8} {'spikes':>7} {'count≠':>7} {'diverge (s)':>12} {'median|Δt| (s)':>15} {'max|Δt| (s)':>12} {'median|ΔV|':>11}")

    for dt in args.dts:
        run_brian(case_dir, dt)
        brian_spikes = load_csv(os.path.join(case_dir, f"spikes_brian_dt{dt:g}.csv"))
        brian_v = load_csv(os.path.join(case_dir, f"voltages_brian_dt{dt:g}.csv"))

        count_mismatch, diverge_time, spike_err = compare_spikes(julia_spikes, brian_spikes, num_neurons)

        # voltages on the shared sample times, before divergence only (afterwards they are different trajectories)
        m = min(len(julia_v), len(brian_v))
        assert np.allclose(julia_v[:m, 0], brian_v[:m, 0]), "voltage sample times differ"
        rows = julia_v[:m, 0] < diverge_time
        v_err = np.abs(julia_v[:m][rows, 1:] - brian_v[:m][rows, 1:])

        diverge_str = "none" if np.isinf(diverge_time) else f"{diverge_time:.4f}"
        fmt = lambda x, w: f"{x:>{w}.2e}" if np.size(x) else f"{'-':>{w}}"  # nothing matched before divergence
        print(f"{dt:>8g} {len(brian_spikes):>7d} {count_mismatch:>7d} {diverge_str:>12} "
              f"{fmt(np.median(spike_err) if spike_err.size else [], 15)} {fmt(spike_err.max() if spike_err.size else [], 12)} "
              f"{fmt(np.median(v_err) if v_err.size else [], 11)}")
