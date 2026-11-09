module PoissonLognormal

using GeneralDPPMixtures, Random, SpecialFunctions, TOML
const G = GeneralDPPMixtures

"""A finite Poisson mixture with fixed M, symmetric Dirichlet weights, and Gaussian log rates."""
struct Model
    y::Vector{Int}
    M::Int
    gamma::Float64
    mu::Float64
    sd::Float64
    proposal_sd::Float64
end

function positive_real(x, name)
    x isa Real && !(x isa Bool) && isfinite(x) && x > 0 ||
        throw(ArgumentError("$name must be finite and positive"))
    Float64(x)
end
function integer_setting(x, name; minimum=1)
    x isa Integer && !(x isa Bool) && minimum <= x <= typemax(Int) ||
        throw(ArgumentError("$name must be an integer >= $minimum"))
    Int(x)
end
function Model(y::AbstractVector; components=3, weight_shape=0.5, base_mean=0.0,
        base_sd=1.5, proposal_sd=0.5)
    isempty(y) && throw(ArgumentError("observations must be nonempty"))
    counts = Int[]
    for value in y
        value isa Real && !(value isa Bool) && isfinite(value) && value >= 0 && isinteger(value) ||
            throw(ArgumentError("Poisson observations must be nonnegative integers"))
        value < 2.0^53 || throw(ArgumentError("Poisson observations must be smaller than 2^53 for the Float64 CSV protocol"))
        try
            push!(counts, Int(value))
        catch error
            error isa InexactError || rethrow()
            throw(ArgumentError("Poisson observation exceeds the supported integer range"))
        end
    end
    sum(BigInt,counts) < big(2)^53 || throw(ArgumentError("the total count must be smaller than 2^53 for exact Float64 sufficient-statistic updates"))
    base_mean isa Real && !(base_mean isa Bool) && isfinite(base_mean) ||
        throw(ArgumentError("base_mean must be finite"))
    Model(counts, integer_setting(components, "components"),
        positive_real(weight_shape, "weight_shape"), Float64(base_mean),
        positive_real(base_sd, "base_sd"), positive_real(proposal_sd, "proposal_sd"))
end

struct Settings
    iterations::Int
    warmup::Int
    thin::Int
    chains::Int
    seed::UInt64
    save_allocations::Bool
end

const CONFIG_KEYS = Set(["method", "components", "weight_shape", "base_mean", "base_sd",
    "proposal_sd", "iterations", "warmup", "thin", "chains", "seed", "save_allocations"])

function configuration(raw::AbstractDict, y)
    unknown = setdiff(Set(keys(raw)), CONFIG_KEYS)
    isempty(unknown) || throw(ArgumentError("unsupported Poisson-lognormal settings: $(join(sort!(collect(unknown)), ", "))"))
    get(raw, "method", nothing) == "poisson_lognormal" ||
        throw(ArgumentError("method must be 'poisson_lognormal' for this entrypoint"))
    m = Model(y; components=get(raw,"components",3), weight_shape=get(raw,"weight_shape",0.5),
        base_mean=get(raw,"base_mean",0.0), base_sd=get(raw,"base_sd",1.5),
        proposal_sd=get(raw,"proposal_sd",0.5))
    iterations = integer_setting(get(raw,"iterations",4000), "iterations")
    warmup = integer_setting(get(raw,"warmup",1000), "warmup"; minimum=0)
    thin = integer_setting(get(raw,"thin",5), "thin")
    warmup < iterations || throw(ArgumentError("warmup must be smaller than iterations"))
    iterations - warmup >= thin || throw(ArgumentError("iterations must retain at least one draw after warmup and thinning"))
    chains = integer_setting(get(raw,"chains",4), "chains")
    seed = integer_setting(get(raw,"seed",123), "seed"; minimum=0)
    save = get(raw,"save_allocations",true)
    save isa Bool || throw(ArgumentError("save_allocations must be a Boolean"))
    m, Settings(iterations,warmup,thin,chains,UInt64(seed),save)
end

# The factorial term is retained so target values have a documented common scale.
function loglikelihood(y::Integer, eta::Real)
    isfinite(eta) || return -Inf
    rate = exp(eta)
    isfinite(rate) || return -Inf
    y * eta - rate - loggamma(y + 1.0)
end
logbase(m::Model, eta) = -log(m.sd) - 0.5log(2pi) - 0.5((eta-m.mu)/m.sd)^2

mutable struct State
    model::Model
    z::Vector{Int}
    eta::Vector{Float64}
    n::Vector{Int}
    sumy::Vector{Float64}
    rng::Xoshiro
    attempted::Int
    accepted::Int
    empty_refreshes::Int
end

function initialize(m::Model, seed::Integer)
    seed >= 0 || throw(ArgumentError("seed must be nonnegative"))
    z = [1+mod(i-1,m.M) for i in eachindex(m.y)]
    n = zeros(Int,m.M)
    sumy = zeros(m.M)
    for i in eachindex(z)
        n[z[i]] += 1
        sumy[z[i]] += m.y[i]
    end
    s = State(m,z,fill(m.mu,m.M),n,sumy,Xoshiro(seed),0,0,0)
    isfinite(logtarget(s)) || throw(ArgumentError("initial target is not finite; choose representable prior parameters"))
    s
end

function logallocation(m::Model, n)
    length(n) == m.M && all(>=(0),n) && sum(n) == length(m.y) ||
        throw(ArgumentError("invalid allocation counts"))
    loggamma(m.M*m.gamma)-loggamma(length(m.y)+m.M*m.gamma)+
        sum(loggamma(nh+m.gamma)-loggamma(m.gamma) for nh in n)
end

"""Full labelled target for z and all M log rates, with mixing weights integrated out."""
function logtarget(s::State)
    m=s.model
    logallocation(m,s.n)+sum(logbase(m,eta) for eta in s.eta)+
        sum(loglikelihood(m.y[i],s.eta[s.z[i]]) for i in eachindex(m.y))
end

"""Gibbs allocation log weights with observation i removed; leaves state intact."""
function allocation_logweights!(scores, s::State, i)
    m=s.model
    for h in 1:m.M
        scores[h]=log(s.n[h]-Int(s.z[i]==h)+m.gamma)+loglikelihood(m.y[i],s.eta[h])
    end
    scores
end

function allocation_sweep!(s::State)
    scores=zeros(s.model.M)
    for i in eachindex(s.z)
        allocation_logweights!(scores,s,i)
        h=G.draw_logweights!(s.rng,scores)
        old=s.z[i]
        s.n[old]-=1; s.sumy[old]-=s.model.y[i]
        s.z[i]=h
        s.n[h]+=1; s.sumy[h]+=s.model.y[i]
    end
    s
end

function log_rate_conditional(s::State, h, eta)
    isfinite(eta) || return -Inf
    # Empty components have exactly the base distribution; avoid 0 * Inf.
    s.n[h] == 0 && return logbase(s.model,eta)
    rate=exp(eta)
    isfinite(rate) || return -Inf
    s.sumy[h]*eta-s.n[h]*rate+logbase(s.model,eta)
end
mh_logratio(s::State,h,proposed) = log_rate_conditional(s,h,proposed)-log_rate_conditional(s,h,s.eta[h])

function rate_sweep!(s::State)
    m=s.model
    for h in 1:m.M
        if s.n[h] == 0
            s.eta[h]=m.mu+m.sd*randn(s.rng)
            s.empty_refreshes+=1
        else
            proposal=s.eta[h]+m.proposal_sd*randn(s.rng)
            ratio=mh_logratio(s,h,proposal)
            s.attempted+=1
            if log(rand(s.rng)) < min(0.0,ratio)
                s.eta[h]=proposal
                s.accepted+=1
            end
        end
    end
    s
end
function sweep!(s::State)
    allocation_sweep!(s)
    rate_sweep!(s)
end

"""Return canonical labels and log rates in matching order, followed by empty slots."""
function canonical_output(s::State)
    labels=G.canonical_labels(s.z)
    occupied=unique(s.z)
    order=vcat(occupied, setdiff(collect(1:s.model.M),occupied))
    labels,s.eta[order]
end

end
