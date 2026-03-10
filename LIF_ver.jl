using DataStructures, RandomNumbers.Xorshifts, StatsBase, PyPlot, Random, DelimitedFiles

function lifnet_edited(num_neurons, num_spikes, in_degree, synaptic_weight, target_rate, mem_time_const, seed_init, seed_topo, input_delays_sec::Vector{Float64}=Float64[])
    ext_input = mem_time_const * sqrt(in_degree) * synaptic_weight * target_rate # ext_input given by balance equation
    phase_speed, ptc_amplitude = 1 / log(1.0 + 1 / ext_input), synaptic_weight / sqrt(in_degree) / (1.0 + ext_input) # phase velocity LIF
    phase_threshold, current_time = 1.0, 0.0 # threshold for LIF
    rng = Xoroshiro128Plus(seed_init) # init. random number generator
    # Heap 1 — spike-time heap (min-heap): absolute next-fire times
    fire_time_heap = MutableBinaryHeap{Float64,DataStructures.FasterForward}(phase_threshold .- rand(rng, num_neurons))

    spike_neuron_ids = Int64[] # initialize spike record
    # Heap 2 — arrival-time heap (min-heap): scheduled delivery times
    arrival_heap = MutableBinaryHeap{Float64,DataStructures.FasterForward}()
    pending_targets = Dict{Int,Vector{Int}}() # heap handle → post-synaptic indices

    spike_times = Float64[] # spike raster
    target_neurons = rand(Int, in_degree)

    # convert per-neuron delays from seconds to internal phase units
    if isempty(input_delays_sec)
        neuron_delays = zeros(num_neurons)
    else
        @assert length(input_delays_sec) == num_neurons "input_delays_sec must have length num_neurons"
        neuron_delays = input_delays_sec .* phase_speed / mem_time_const
    end

    while length(spike_neuron_ids) < num_spikes # main loop

        next_fire_time, firing_neuron = top_with_handle(fire_time_heap) # get absolute fire time of next spiking neuron

        # time of next arrival (from Heap 2)
        if isempty(arrival_heap)
            next_arrival_time = Inf
        else
            next_arrival_time, _ = top_with_handle(arrival_heap)
        end

        if next_arrival_time <= next_fire_time
            # ── arrival event wins ──
            current_time = next_arrival_time
            # drain all arrivals at this exact time
            while !isempty(arrival_heap)
                arrival_time, heap_handle = top_with_handle(arrival_heap)
                if abs(arrival_time - current_time) < 1e-12
                    pop!(arrival_heap)
                    ptc!(fire_time_heap, pending_targets[heap_handle], current_time, phase_speed, ptc_amplitude, phase_threshold)
                    delete!(pending_targets, heap_handle)
                else
                    break
                end
            end
            continue # arrival events do not consume a spike slot
        else
            # ── spike emission event wins ──
            current_time = next_fire_time # clock advances to exact spike time
            Random.seed!(rng, firing_neuron + seed_topo) # spiking neuron index is seed of rng
            sample!(rng, 1:num_neurons-1, target_neurons; replace = false) # get receiving neuron indices

            @inbounds for i = 1:in_degree # avoid autapses
                target_neurons[i] >= firing_neuron && (target_neurons[i] += 1)
            end

            firing_neuron_delay = neuron_delays[firing_neuron] # outgoing delay of spiking neuron
            if firing_neuron_delay == 0.0
                ptc!(fire_time_heap, target_neurons, current_time, phase_speed, ptc_amplitude, phase_threshold) # immediate coupling
            else
                # schedule arrival in Heap 2
                heap_handle = push!(arrival_heap, current_time + firing_neuron_delay)
                pending_targets[heap_handle] = copy(target_neurons)
            end

            update!(fire_time_heap, firing_neuron, current_time + phase_threshold) # reset: phase 0 fires again after phase_threshold
            push!(spike_times, current_time) # store spike times
            push!(spike_neuron_ids, firing_neuron) # store spiking neuron index
        end
    end
    num_spikes / current_time / num_neurons / mem_time_const * phase_speed, spike_neuron_ids, spike_times * mem_time_const / phase_speed # output: rate, spike times & indices
end

function export_connectivity_csv(
    filename::String,
    num_neurons::Int,
    in_degree::Int,
    seed_topo::Int,
    input_delays_sec::Vector{Float64} = Float64[]
)
    @assert in_degree < num_neurons "in_degree must be < num_neurons"

    # Resolve per-neuron delays in seconds (no phase-unit conversion needed)
    if isempty(input_delays_sec)
        delays_out = zeros(Float64, num_neurons)
    else
        @assert length(input_delays_sec) == num_neurons "input_delays_sec must have length num_neurons"
        delays_out = input_delays_sec
    end

    adj = fill(-1.0, num_neurons, num_neurons)  # -1.0 = no connection
    target_neurons = Vector{Int}(undef, in_degree)
    rng = Xoroshiro128Plus(0)  # initial seed irrelevant; overwritten each iteration

    # Reconstruct topology — exact replica of spike-emission block in lifnet_edited
    for source in 1:num_neurons
        Random.seed!(rng, source + seed_topo)
        sample!(rng, 1:num_neurons-1, target_neurons; replace = false)
        @inbounds for i in 1:in_degree
            target_neurons[i] >= source && (target_neurons[i] += 1)
        end
        delay_s = delays_out[source]
        for t in target_neurons
            adj[source, t] = delay_s
        end
    end

    # Write CSV: header row of column labels, then data rows (row=source, col=target)
    open(filename, "w") do io
        println(io, join(string.(1:num_neurons), ","))
        writedlm(io, adj, ',')
    end
    nothing
end

function save_spikes_csv(filename::String, neuron_ids::Vector{Int64}, spike_times::Vector{Float64})
    open(filename, "w") do io
        println(io, "neuron_id,spike_time_s")
        writedlm(io, hcat(neuron_ids, spike_times), ',')
    end
end

function ptc!(fire_time_heap, target_ids, current_time, phase_speed, ptc_amplitude, phase_threshold) # phase transition curve of LIF
    for i in target_ids
        current_phase = phase_threshold - (fire_time_heap.nodes[fire_time_heap.node_map[i]].value - current_time) # recover absolute phase
        updated_phase = -phase_speed * log(exp(-current_phase / phase_speed) + ptc_amplitude) # apply PTC (Eq. 12)
        update!(fire_time_heap, i, current_time + phase_threshold - updated_phase) # store new absolute fire time
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    # set parameters:
    # num_neurons: # of neurons, in_degree: synapses/neuron, synaptic_weight: syn. strength, mem_time_const: membr. time const.
    num_neurons, num_spikes, in_degree, synaptic_weight, target_rate, mem_time_const, seed_init, seed_topo = 10^5, 10^5, 100, 1, 1.0, 0.01, 1, 1
    # quick run to compile code
    @time lifnet_edited(100, 1, 10, synaptic_weight, target_rate, mem_time_const, seed_init, seed_topo)

    # run & benchmark: no delay (baseline)
    GC.gc()
    println("No delay:")
    @time firing_rate, neuron_ids, spike_times = lifnet_edited(num_neurons, num_spikes, in_degree, synaptic_weight, target_rate, mem_time_const, seed_init, seed_topo)
    println("  rate = $firing_rate Hz")

    # run & benchmark: heterogeneous per-neuron delays (uniform in [1ms, 5ms])
    neuron_delays_sec = 0.001 .+ 0.004 .* rand(num_neurons)
    GC.gc()
    println("Per-neuron delays (1–5 ms uniform):")
    @time firing_rate_d, neuron_ids_d, spike_times_d = lifnet_edited(num_neurons, num_spikes, in_degree, synaptic_weight, target_rate, mem_time_const, seed_init, seed_topo, neuron_delays_sec)
    println("  rate = $firing_rate_d Hz")

    # Export connectivity for a 200-neuron network
    println("Exporting connectivity CSV (200 neurons, with delays)...")
    export_connectivity_csv("connectivity_200.csv", 200, 100, seed_topo,
        0.001 .+ 0.004 .* rand(200))
    println("Written to connectivity_200.csv")

    println("Exporting connectivity CSV (200 neurons, no delays)...")
    export_connectivity_csv("connectivity_200_nodelay.csv", 200, 100, seed_topo)
    println("Written to connectivity_200_nodelay.csv")

    # plot spike raster
    figure(figsize=(12, 5))

    subplot(1, 2, 1)
    plot(spike_times, neuron_ids, ",k", ms = 0.1)
    ylabel("Neuron Index", fontsize = 14)
    xlabel("Time (s)", fontsize = 14)
    title("No delay", fontsize = 14)

    subplot(1, 2, 2)
    plot(spike_times_d, neuron_ids_d, ",k", ms = 0.1)
    ylabel("Neuron Index", fontsize = 14)
    xlabel("Time (s)", fontsize = 14)
    title("Per-neuron delays (1–5 ms)", fontsize = 14)

    tight_layout()
    show()
    readline()  # keep window open until Enter is pressed
end
