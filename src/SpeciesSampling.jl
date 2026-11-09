module SpeciesSampling

using Random, Distributions, SpecialFunctions

export PartitionPrior, DirichletProcess, PitmanYor, strength, discount,
    logeppf, seating_logweights, ComponentModel, GaussianLocation, observations,
    loglikelihood, logbase, randbase, logmarginal, randposterior, logpredictive,
    State, initialize, canonical_labels, cluster_counts, check_state,
    collapsed_allocation, collapsed_sweep!, collapsed_logtarget, sample_logweights

abstract type PartitionPrior end

function finite_real(x, name)
    x isa Real && !(x isa Bool) || throw(ArgumentError("$name must be a finite real number"))
    value = Float64(x)
    isfinite(value) || throw(ArgumentError("$name must be a finite real number"))
    value
end

"""Dirichlet-process partition prior with fixed positive strength `alpha`."""
struct DirichletProcess <: PartitionPrior
    alpha::Float64
    function DirichletProcess(alpha)
        a = finite_real(alpha, "strength")
        a > 0 || throw(ArgumentError("Dirichlet-process strength must be positive"))
        new(a)
    end
end

"""
Pitman–Yor partition prior with fixed strength `alpha` and discount `d`.
The support is `0 <= d < 1` and `alpha > -d`; at `d == 0`, `alpha > 0`.
This includes negative strength when `d > 0`.
"""
struct PitmanYor <: PartitionPrior
    alpha::Float64
    d::Float64
    function PitmanYor(alpha, d)
        a = finite_real(alpha, "strength")
        dd = finite_real(d, "discount")
        0 <= dd < 1 || throw(ArgumentError("discount must satisfy 0 <= discount < 1"))
        a > -dd || throw(ArgumentError("strength must exceed -discount (and be positive at discount zero)"))
        new(a, dd)
    end
end

strength(p::PartitionPrior) = p.alpha
discount(::DirichletProcess) = 0.0
discount(p::PitmanYor) = p.d

function valid_counts(counts)
    all(n -> n isa Integer && !(n isa Bool) && n > 0, counts) ||
        throw(ArgumentError("occupied block counts must be positive integers"))
    counts
end

"""
Log probability of one specific set partition with the supplied block sizes.
It includes all prior normalising factors; it has no combinatorial factor for
the number of partitions with those sizes. The empty partition has probability
one. Source: Pitman (2006), Section 3.2, Theorem 3.2, equation (3.6), with his
discount alpha and strength theta written here as d and alpha respectively:
https://www.stat.berkeley.edu/~aldous/206-Exch/Papers/pitman_CSP.pdf
"""
function logeppf(prior::PartitionPrior, counts)
    valid_counts(counts)
    isempty(counts) && return 0.0
    n = sum(counts)
    a, d = strength(prior), discount(prior)
    # Products start after the deterministic first customer. In particular,
    # this avoids a log(alpha) term for the valid negative-strength PY case.
    value = sum((log(a + j*d) for j in 1:(length(counts)-1)); init=0.0)
    value -= sum((log(a + j) for j in 1:(n-1)); init=0.0)
    for count in counts
        value += sum((log(j-d) for j in 1:(count-1)); init=0.0)
    end
    value
end

"""
Unnormalised log seating weights for occupied blocks followed by one new
block. For an empty partition there is one deterministic choice with log
weight zero, including when a Pitman–Yor strength is zero or negative.
"""
function seating_logweights(prior::PartitionPrior, counts)
    valid_counts(counts)
    isempty(counts) && return [0.0]
    a, d = strength(prior), discount(prior)
    vcat([log(n-d) for n in counts], log(a + length(counts)*d))
end

"""
Source-level component interface. Concrete models provide observations and
the ordinary Julia methods below. A collapsed sweep additionally requires
`logmarginal`; samplers that retain component parameters use the base draws,
likelihood and posterior-update methods. Missing methods fail explicitly.
"""
abstract type ComponentModel end
observations(model::ComponentModel) = model.y
loglikelihood(model::ComponentModel, y, theta) =
    throw(ArgumentError("implement loglikelihood(model, y, theta) for $(typeof(model))"))
logbase(model::ComponentModel, theta) =
    throw(ArgumentError("implement logbase(model, theta) for $(typeof(model))"))
randbase(rng::AbstractRNG, model::ComponentModel) =
    throw(ArgumentError("implement randbase(rng, model) for $(typeof(model))"))
logmarginal(model::ComponentModel, indices) =
    throw(ArgumentError("collapsed sampling requires logmarginal(model, indices) for $(typeof(model))"))
randposterior(rng::AbstractRNG, model::ComponentModel, indices) =
    throw(ArgumentError("implement randposterior(rng, model, indices) for $(typeof(model))"))
logpredictive(model::ComponentModel, indices, i::Integer) =
    logmarginal(model, vcat(collect(indices), i)) - logmarginal(model, indices)

"""Normal observations with known `observation_sd` and a Normal location base."""
struct GaussianLocation <: ComponentModel
    y::Vector{Float64}
    observation_sd::Float64
    base_mean::Float64
    base_sd::Float64
    function GaussianLocation(y; observation_sd=1.0, base_mean=0.0, base_sd=2.0)
        isempty(y) && throw(ArgumentError("GaussianLocation needs at least one observation"))
        yy = [finite_real(x, "observation") for x in y]
        sigma = finite_real(observation_sd, "observation_sd")
        mu = finite_real(base_mean, "base_mean")
        tau = finite_real(base_sd, "base_sd")
        for (name, sd) in (("observation_sd", sigma), ("base_sd", tau))
            sd > 0 || throw(ArgumentError("$name must be positive"))
            isfinite(sd^2) && sd^2 > 0 && isfinite(inv(sd^2)) ||
                throw(ArgumentError("$name is outside the supported Float64 variance range"))
        end
        isfinite(sigma^2 + tau^2) ||
            throw(ArgumentError("observation and base variances overflow Float64 when added"))
        model = new(yy, sigma, mu, tau)
        all(isfinite(logpdf(Normal(mu, hypot(sigma,tau)), x)) for x in yy) &&
            isfinite(logmarginal(model, eachindex(yy))) ||
            throw(ArgumentError("data and scales give nonfinite Gaussian marginal densities in Float64"))
        posterior_parameters(model, eachindex(yy))
        model
    end
end

loglikelihood(model::GaussianLocation, y, theta) =
    logpdf(Normal(theta, model.observation_sd), y)
logbase(model::GaussianLocation, theta) = logpdf(Normal(model.base_mean, model.base_sd), theta)
randbase(rng::AbstractRNG, model::GaussianLocation) = rand(rng, Normal(model.base_mean, model.base_sd))

# Sequential conjugate updates avoid subtracting large sums of squares.
function posterior_parameters(model::GaussianLocation, indices)
    mean, variance = model.base_mean, model.base_sd^2
    obsvariance = model.observation_sd^2
    for i in indices
        denominator = variance + obsvariance
        mean += (variance/denominator)*(model.y[i]-mean)
        variance *= obsvariance/denominator
    end
    isfinite(mean) && isfinite(variance) && variance > 0 ||
        throw(ArgumentError("Gaussian posterior parameters are outside the supported Float64 range"))
    mean, sqrt(variance)
end

function logmarginal(model::GaussianLocation, indices)
    mean, variance = model.base_mean, model.base_sd^2
    obsvariance = model.observation_sd^2
    value = 0.0
    for i in indices
        denominator = variance + obsvariance
        value += logpdf(Normal(mean, sqrt(denominator)), model.y[i])
        mean += (variance/denominator)*(model.y[i]-mean)
        variance *= obsvariance/denominator
    end
    value
end

function randposterior(rng::AbstractRNG, model::GaussianLocation, indices)
    mean, sd = posterior_parameters(model, indices)
    rand(rng, Normal(mean, sd))
end

function logpredictive(model::GaussianLocation, indices, i::Integer)
    mean, sd = posterior_parameters(model, indices)
    logpdf(Normal(mean, hypot(sd, model.observation_sd)), model.y[i])
end

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

end
