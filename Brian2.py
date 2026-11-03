# this was qifnet02inhibitoryTruelyDimensionlessHist
# this script has a rate of 15.037775589887717 Hz
# the voltage distribution is identical to the Brian implementation

#Change to LIF network implementation
from brian2 import *
import numpy as np
import matplotlib.pyplot as plt

def visualise_connectivity(S):
    Ns = len(S.source)
    Nt = len(S.target)
    figure(figsize=(10, 4))
    subplot(121)
    plot(zeros(Ns), arange(Ns), 'ok', ms=10)
    plot(ones(Nt), arange(Nt), 'ok', ms=10)
    for i, j in zip(S.i, S.j):
        plot([0, 1], [i, j], '-k')
    xticks([0, 1], ['Source', 'Target'])
    ylabel('Neuron index')
    xlim(-0.1, 1.1)
    ylim(-1, max(Ns, Nt))
    subplot(122)
    plot(S.i, S.j, 'ok')
    xlim(-1, Ns)
    ylim(-1, Nt)
    xlabel('Source neuron index')
    ylabel('Target neuron index')

# Simulation parameters
N = 40  # Number of neurons
K = 10 # number of synapses per neuron
J0 = 1.0 # coupling strength factor
ratewnt = 1.0*Hz # wanted firing rate according to balance equation
tau = 10*ms  # Membrane time constant (in time units)
Vt_d = 10#10  # Spike threshold (dimensionless)
Vreset_d = -1  # Reset potential (dimensionless)
deltaV = 1.0*1
Iext_d =  sqrt(K) * J0 * ratewnt*tau/2  + Vt_d# External input current (dimensionless)

print(f"Iext_d = {Iext_d}")


gI = 1/sqrt(K)  # Inhibitory conductance (dimensionless)



defaultclock.dt = 0.01 * ms
# Define the dimensionless LIF neuron model
eqs = '''
dv/dt = (v0 - v) / tau : 1  (unless refractory)
v0 : 1
'''

# Create neuron groups
P = NeuronGroup(N, eqs, threshold='v>Vt_d', reset='v = Vreset_d')

# Initialize membrane potentials
P.v = -rand(len(P))

#init v0
P.v0 = Iext_d

# Synapses (only inhibitory)
Ci = Synapses(P, P, model='w : 1', on_pre='v_post -= w')
Ci.connect(p=K/(N-1))  # Connect with probability K/N

# Define synaptic weights
Ci.w = gI

# Monitors
spikemon = SpikeMonitor(P)
statemon = StateMonitor(P, 'v', record=True)


# Run the simulation
duration=2.0*second
run(duration)

firing_rates = spikemon.count / duration
firing_rates_unitless = firing_rates / Hz

# Plot results
figure()
subplot(221)
plot(spikemon.t/ms, spikemon.i, '.k')
xlabel('Time (ms)')
ylabel('Neuron index')

subplot(222)
plot(statemon.t/ms, statemon.v[1])
xlabel('Time (ms)')
ylabel('Membrane potential (dimensionless)')
ylim(-2,2)
plt.subplot(223)
plt.hist(firing_rates_unitless, bins=20, edgecolor='black')
plt.xlabel('Firing rate (Hz)')
plt.ylabel('Number of neurons')
title(mean(firing_rates_unitless))
plt.subplot(224)
plt.hist(statemon.v.ravel(), bins=1000, density=True)
tight_layout()

visualise_connectivity(Ci)
show()