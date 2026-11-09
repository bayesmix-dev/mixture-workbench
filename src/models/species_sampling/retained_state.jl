"""Occupied atoms and allocations; DP weights are integrated out."""
mutable struct State{M<:S.ComponentModel}
    model::M
    prior::S.DirichletProcess
    z::Vector{Int}
    theta::Vector{Float64}
    rng::Xoshiro
    sweeps::Int
    atom_attempts::Int
    atom_accepts::Int
end

_conjugate(::S.ComponentModel) = false
_conjugate(::S.GaussianLocation) = true

function initialize(model::S.ComponentModel, prior::S.PartitionPrior, seed::Integer;
        initial_clusters=1, partition=nothing)
    prior isa S.DirichletProcess ||
        throw(ArgumentError("the named Neal kernels require DirichletProcess; use an explicit DP prior"))
    model isa Union{S.GaussianLocation,PoissonLogRate} ||
        throw(ArgumentError("no atom update is implemented for $(typeof(model))"))
    # The shared collapsed initializer additionally requires logmarginal, which
    # is deliberately unavailable for the nonconjugate model.
    base = if _conjugate(model)
        S.initialize(model, prior, seed; initial_clusters, partition)
    else
        !(seed isa Bool) && seed>=0 || throw(ArgumentError("seed must be a nonnegative integer"))
        n=length(model.y)
        z=if partition===nothing
            initial_clusters isa Integer && !(initial_clusters isa Bool) && 1<=initial_clusters<=n ||
                throw(ArgumentError("initial_clusters must be an integer between one and the sample size"))
            [mod1(i,initial_clusters) for i in 1:n]
        else
            length(partition)==n || throw(ArgumentError("partition length must equal the sample size"))
            S.canonical_labels(partition)
        end
        S.State(model,prior,z,Xoshiro(seed),0)
    end
    theta = if _conjugate(model)
        [S.randposterior(base.rng, model, findall(==(h),base.z)) for h in 1:maximum(base.z)]
    else
        fill(model.base_mean, maximum(base.z))
    end
    state = State(model, prior, base.z, theta, base.rng, 0, 0, 0)
    check_state(state)
    state
end
