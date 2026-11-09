function sample_logweights(rng::AbstractRNG, logweights)
    isempty(logweights) && throw(ArgumentError("at least one log weight is required"))
    all(x -> isfinite(x) || x == -Inf, logweights) ||
        throw(ArgumentError("log weights must be finite or negative infinity"))
    top = maximum(logweights)
    isfinite(top) || throw(ArgumentError("at least one log weight must be finite"))
    weights = exp.(logweights .- top)
    threshold = rand(rng)*sum(weights)
    cumulative = 0.0
    for h in eachindex(weights)
        cumulative += weights[h]
        threshold < cumulative && return h
    end
    findlast(>(0), weights)
end

"""
Return candidate partitions and their unnormalised log full-conditional weights
for one observation, without changing the state or consuming randomness. The
last candidate creates a new occupied block; all candidates are canonical.
"""
function collapsed_allocation(state::State, i::Integer)
    checkbounds(state.z, i)
    remaining = canonical_labels(state.z[setdiff(eachindex(state.z), (i,))])
    groups = [Int[] for _ in 1:(isempty(remaining) ? 0 : maximum(remaining))]
    offset = 0
    for j in eachindex(state.z)
        j == i && continue
        offset += 1
        push!(groups[remaining[offset]], j)
    end
    logweights = seating_logweights(state.prior, length.(groups))
    for h in eachindex(groups)
        logweights[h] += logpredictive(state.model, groups[h], i)
    end
    logweights[end] += logpredictive(state.model, Int[], i)
    partitions = Vector{Int}[]
    for h in eachindex(logweights)
        labels = copy(remaining)
        insert!(labels, i, h)
        push!(partitions, canonical_labels(labels))
    end
    (partitions=partitions, logweights=logweights)
end

"""One systematic collapsed Gibbs sweep, retaining a canonical partition."""
function collapsed_sweep!(state::State)
    check_state(state)
    for i in eachindex(state.z)
        choices = collapsed_allocation(state, i)
        selected = sample_logweights(state.rng, choices.logweights)
        state.z .= choices.partitions[selected]
    end
    state.sweeps += 1
    state
end
