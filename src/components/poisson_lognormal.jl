"""
Nonconjugate Poisson observations with a Gaussian base measure on log rates.
Counts and their total must be smaller than 2^53. Calculations use Float64:
full joint densities include factorial terms and can lose absolute precision
for very large counts; MH ratios cancel those terms before evaluation.
"""
struct PoissonLogRate <: S.ComponentModel
    y::Vector{Int}
    base_mean::Float64
    base_sd::Float64
    proposal_sd::Float64
end

function _positive(x, name)
    x isa Real && !(x isa Bool) && isfinite(x) && x > 0 ||
        throw(ArgumentError("$name must be finite and positive"))
    value = Float64(x)
    isfinite(value) && value>0 || throw(ArgumentError("$name is outside the supported Float64 range"))
    value
end

function PoissonLogRate(y::AbstractVector; base_mean=0.0, base_sd=2.0, proposal_sd=0.4)
    isempty(y) && throw(ArgumentError("observations must be nonempty"))
    counts = Int[]
    for value in y
        value isa Real && !(value isa Bool) && isfinite(value) && value >= 0 && isinteger(value) ||
            throw(ArgumentError("Poisson observations must be nonnegative integers"))
        value < 2.0^53 || throw(ArgumentError("Poisson observations must be smaller than 2^53"))
        push!(counts, Int(value))
    end
    sum(BigInt, counts) < big(2)^53 ||
        throw(ArgumentError("the total count must be smaller than 2^53"))
    base_mean isa Real && !(base_mean isa Bool) && isfinite(base_mean) ||
        throw(ArgumentError("base_mean must be finite"))
    isfinite(Float64(base_mean)) || throw(ArgumentError("base_mean is outside the supported Float64 range"))
    PoissonLogRate(counts, Float64(base_mean), _positive(base_sd,"base_sd"),
        _positive(proposal_sd,"proposal_sd"))
end

# Adapted from examples/poisson_lognormal/PoissonLognormal.jl: keep the
# factorial term and perform the symmetric random walk on the log-rate scale.
function loglikelihood(::PoissonLogRate, y, theta)
    isfinite(theta) || return -Inf
    rate = exp(theta)
    isfinite(rate) || return -Inf
    y * theta - rate - loggamma(y + 1.0)
end
logbase(m::PoissonLogRate, theta) =
    -log(m.base_sd) - 0.5log(2pi) - 0.5((theta-m.base_mean)/m.base_sd)^2
randbase(rng::AbstractRNG, m::PoissonLogRate) = m.base_mean + m.base_sd * randn(rng)
