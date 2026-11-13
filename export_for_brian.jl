# Export everything Brian2.py needs to rerun the exact same network from the exact same initial state,
# plus the Julia reference results (spikes, and voltages sampled on Brian's time grid).
include("LIF_ver.jl")

function export_brian_comparison(outdir::String; num_neurons=200, in_degree=100, synaptic_weight=1, target_rate=1.0,
                                 mem_time_const=0.01, seed_init=1, seed_topo=1, seed_delay=1,
                                 with_delays=false, delay_range_sec=(0.001, 0.005),
                                 dt=1e-5, duration=1.0, record_dt=1e-3, trace_ids=1:5)
    mkpath(outdir)

    # per-neuron delays as whole multiples of dt (Brian2 rounds delays to its time grid), seeded so reruns repeat
    if with_delays
        rng = Xoroshiro128Plus(seed_delay)
        delay_steps = round(Int, delay_range_sec[1] / dt):round(Int, delay_range_sec[2] / dt)
        delays_sec = dt .* rand(rng, delay_steps, num_neurons)
    else
        delays_sec = Float64[]
    end

    # sample all neurons at multiples of dt so samples line up with Brian's grid; the t=0 column is the initial state
    record_steps = round(Int, record_dt / dt)
    sample_times_sec = dt .* (0:record_steps:round(Int, duration / dt))

    # lifnet_edited stops after a spike count: rerun with more spikes until it passes `duration` (deterministic)
    num_spikes = ceil(Int, 10 * num_neurons * target_rate * duration)
    local neuron_ids, spike_times, voltages, traces
    while true
        _, neuron_ids, spike_times, voltages, traces = lifnet_edited(num_neurons, num_spikes, in_degree, synaptic_weight, target_rate, mem_time_const,
                                                                     seed_init, seed_topo, delays_sec; sample_times_sec, record_ids=1:num_neurons, trace_ids)
        maximum(spike_times) > duration && break
        num_spikes *= 2
    end

    # network: same topology reconstruction as the simulation, with the delays actually used
    export_connectivity_csv(joinpath(outdir, "connectivity.csv"), num_neurons, in_degree, seed_topo, delays_sec)

    open(joinpath(outdir, "initial_voltages.csv"), "w") do io
        println(io, "neuron_id,v0")
        for i in 1:num_neurons
            println(io, i, ",", voltages[i, 1])
        end
    end

    # parameters in the units Brian2 uses (Vt=1, Vreset=0), so Brian2.py reads them instead of re-deriving them
    ext_input = mem_time_const * sqrt(in_degree) * synaptic_weight * target_rate
    params = Pair{String,Any}["num_neurons" => num_neurons, "in_degree" => in_degree, "synaptic_weight" => synaptic_weight,
              "target_rate" => target_rate, "mem_time_const" => mem_time_const,
              "v_threshold" => 1.0, "v_reset" => 0.0, "v_drive" => 1.0 + ext_input, "w" => synaptic_weight / sqrt(in_degree),
              "dt" => dt, "duration" => duration, "record_dt" => record_dt, "with_delays" => Int(with_delays),
              "seed_init" => seed_init, "seed_topo" => seed_topo, "seed_delay" => seed_delay]
    open(joinpath(outdir, "params.csv"), "w") do io
        println(io, "name,value")
        for (name, value) in params
            println(io, name, ",", value)
        end
    end

    # Julia reference results, truncated to the comparison window
    keep = spike_times .<= duration
    save_spikes_csv(joinpath(outdir, "spikes_julia.csv"), neuron_ids[keep], spike_times[keep])
    open(joinpath(outdir, "voltages_julia.csv"), "w") do io
        println(io, "time_s,", join(string.(1:num_neurons), ","))
        writedlm(io, hcat(sample_times_sec, permutedims(voltages)), ',')
    end

    # exact voltage traces: voltage right after every input/reset of the traced neurons (brian2_compare.py records the same neurons)
    open(joinpath(outdir, "traces_julia.csv"), "w") do io
        println(io, "neuron_id,time_s,v")
        for (nid, trace) in zip(trace_ids, traces), (t, v) in eachrow(trace)
            t <= duration && println(io, nid, ",", t, ",", v)
        end
    end

    println("$outdir: $(count(keep)) spikes in $(duration) s, $(length(sample_times_sec)) voltage samples, V range $(extrema(voltages))")
end

if abspath(PROGRAM_FILE) == @__FILE__
    export_brian_comparison("comparison/nodelay")
    export_brian_comparison("comparison/delay"; with_delays=true)
end
