# Clock-driven Brian2 run of the exact network exported by export_for_brian.jl:
# same parameters, connectivity, delays and initial voltages as the Julia (SparseProp) run,
# with spikes and voltages saved in the same CSV formats for comparison.
#
# usage: python brian2_compare.py [comparison/nodelay] [--dt 1e-5] [--plot]

import argparse
import os

from brian2 import *
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("case_dir", nargs="?", default="comparison/nodelay", help="folder written by export_for_brian.jl")
parser.add_argument("--dt", type=float, default=None, help="simulation time step in seconds (default: dt in params.csv)")
parser.add_argument("--plot", action="store_true", help="overlay Brian2 and Julia rasters")
parser.add_argument("--trace-window", type=float, default=0.3, help="seconds of every-step voltage traces to save")
args = parser.parse_args()

# Parameters (Vt=1, Vreset=0 units, already converted by Julia)
params = {name: float(value) for name, value in np.loadtxt(os.path.join(args.case_dir, "params.csv"), delimiter=",", skiprows=1, dtype=str)}
N = int(params["num_neurons"])
tau = params["mem_time_const"] * second
v_threshold = params["v_threshold"]
v_reset = params["v_reset"]
v_drive = params["v_drive"]  # 1 + iext from the balance equation
w = params["w"]  # J0 / sqrt(K)
duration = params["duration"] * second
record_dt = params["record_dt"] * second
sim_dt = args.dt if args.dt is not None else params["dt"]
defaultclock.dt = sim_dt * second

# Connectivity: row = source, col = target (1-based in Julia), value = delay in s, -1 = no connection
adj = np.loadtxt(os.path.join(args.case_dir, "connectivity.csv"), delimiter=",", skiprows=1)
sources, targets = np.nonzero(adj >= 0)
delays = adj[sources, targets]

# Initial voltages from the Julia initial phases
v0 = np.loadtxt(os.path.join(args.case_dir, "initial_voltages.csv"), delimiter=",", skiprows=1)[:, 1]

# Dimensionless LIF, solved exactly between time steps
eqs = '''
dv/dt = (v_drive - v) / tau : 1
'''
P = NeuronGroup(N, eqs, threshold='v > v_threshold', reset='v = v_reset', method='exact')
P.v = v0

# Inhibitory delta synapses
Ci = Synapses(P, P, model='w_syn : 1', on_pre='v_post -= w_syn')
Ci.connect(i=sources, j=targets)
Ci.w_syn = w
Ci.delay = delays * second  # Brian2 rounds to whole time steps; export_for_brian.jl makes them exact multiples of dt

# Monitors: voltages at the same multiples of record_dt as voltages_julia.csv
spikemon = SpikeMonitor(P)
statemon = StateMonitor(P, 'v', record=True, dt=record_dt)
# every-step traces of the neurons Julia traced exactly (traces_julia.csv)
trace_file = os.path.join(args.case_dir, "traces_julia.csv")
trace_ids = np.unique(np.loadtxt(trace_file, delimiter=",", skiprows=1, ndmin=2)[:, 0]).astype(int) if os.path.exists(trace_file) else []
tracemon = StateMonitor(P, 'v', record=trace_ids - 1) if len(trace_ids) else None

run(duration)

# Save in the same formats as the Julia outputs (neuron ids 1-based)
tag = f"dt{sim_dt:g}"
spike_file = os.path.join(args.case_dir, f"spikes_brian_{tag}.csv")
np.savetxt(spike_file, np.column_stack([spikemon.i[:] + 1, spikemon.t[:] / second]),
           delimiter=",", header="neuron_id,spike_time_s", comments="", fmt=["%d", "%.17g"])
voltage_file = os.path.join(args.case_dir, f"voltages_brian_{tag}.csv")
np.savetxt(voltage_file, np.column_stack([statemon.t[:] / second, statemon.v[:].T]),
           delimiter=",", header="time_s," + ",".join(str(i) for i in range(1, N + 1)), comments="", fmt="%.17g")

if tracemon is not None:
    keep = tracemon.t[:] / second < args.trace_window
    np.savetxt(os.path.join(args.case_dir, f"traces_brian_{tag}.csv"),
               np.column_stack([tracemon.t[:][keep] / second, tracemon.v[:, keep].T]),
               delimiter=",", header="time_s," + ",".join(str(i) for i in trace_ids), comments="", fmt="%.17g")

print(f"{args.case_dir} (dt = {sim_dt:g} s): {spikemon.num_spikes} spikes, {len(sources)} synapses, "
      f"mean rate = {spikemon.num_spikes / N / float(duration):.4f} Hz")
print(f"wrote {spike_file} and {voltage_file}")

if args.plot:
    julia_spikes = np.loadtxt(os.path.join(args.case_dir, "spikes_julia.csv"), delimiter=",", skiprows=1)
    plt.figure(figsize=(10, 5))
    plt.plot(julia_spikes[:, 1], julia_spikes[:, 0], 'ok', ms=4, mfc='none', label='Julia (SparseProp)')
    plt.plot(spikemon.t / second, spikemon.i + 1, '.r', ms=3, label=f'Brian2 (dt = {sim_dt:g} s)')
    plt.xlabel('Time (s)')
    plt.ylabel('Neuron index')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()
