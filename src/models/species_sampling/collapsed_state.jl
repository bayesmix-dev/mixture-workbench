"""Copy positive integer labels to order of first occurrence, starting at one."""
function canonical_labels(z)
    all(h -> h isa Integer && !(h isa Bool) && h > 0, z) ||
        throw(ArgumentError("partition labels must be positive integers"))
    labels = Dict{eltype(z),Int}()
    result = Int[]
    for h in z
        if !haskey(labels, h)
            labels[h] = length(labels)+1
        end
        push!(result, labels[h])
    end
    result
end

function cluster_counts(z)
    labels = canonical_labels(z)
    isempty(labels) && return Int[]
    counts = zeros(Int, maximum(labels))
    for h in labels
        counts[h] += 1
    end
    counts
end

mutable struct State{M<:ComponentModel,P<:PartitionPrior,R<:AbstractRNG}
    model::M
    prior::P
    z::Vector{Int}
    rng::R
    sweeps::Int
end

"""
Initialise a canonical partition. `initial_clusters` uses a round-robin
allocation; an explicit `partition` takes precedence. A supplied RNG is copied
so the chain owns its random stream. Otherwise `seed` must be nonnegative.
"""
function initialize(model::ComponentModel, prior::PartitionPrior, seed;
        initial_clusters=1, partition=nothing)
    rng = if seed isa AbstractRNG
        deepcopy(seed)
    elseif seed isa Integer && !(seed isa Bool) && seed >= 0
        Xoshiro(seed)
    else
        throw(ArgumentError("seed must be a nonnegative integer or an AbstractRNG"))
    end
    n = length(observations(model))
    n > 0 || throw(ArgumentError("the model must contain at least one observation"))
    z = if partition === nothing
        initial_clusters isa Integer && !(initial_clusters isa Bool) && 1 <= initial_clusters <= n ||
            throw(ArgumentError("initial_clusters must be an integer between one and the sample size"))
        [mod1(i, initial_clusters) for i in 1:n]
    else
        length(partition) == n || throw(ArgumentError("partition length must equal the sample size"))
        canonical_labels(partition)
    end
    state = State(model, prior, z, rng, 0)
    check_state(state)
    isfinite(collapsed_logtarget(state)) ||
        throw(ArgumentError("initial partition has a nonfinite collapsed target"))
    state
end

function check_state(state::State)
    length(state.z) == length(observations(state.model)) ||
        throw(ArgumentError("state partition length does not match the observations"))
    state.z == canonical_labels(state.z) || throw(ArgumentError("state partition must use canonical labels"))
    state.sweeps >= 0 || throw(ArgumentError("sweep count must be nonnegative"))
    true
end
