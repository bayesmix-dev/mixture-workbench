"""
Complete log joint density of a specific canonical partition and the observed
data. The random mixing distribution and all component locations are integrated
out. The posterior normalising constant (the data evidence) is not included.
"""
function collapsed_logtarget(state::State)
    check_state(state)
    counts = cluster_counts(state.z)
    logeppf(state.prior, counts) +
        sum((logmarginal(state.model, findall(==(h), state.z)) for h in eachindex(counts)); init=0.0)
end
