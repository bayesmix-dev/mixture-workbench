module EfficientSlice

using Random, Distributions, SpecialFunctions
using ..SpeciesSampling: ComponentModel, GaussianLocation, PartitionPrior,
    DirichletProcess, strength, loglikelihood, logbase, randbase, logmarginal,
    randposterior, logeppf

export State, initialize, sweep!, ordered_counts, stick_shapes, draw_logstick,
    refresh_components!, draw_auxiliaries!, allocation_logweights,
    allocation_probabilities, occupied_count, materialized_count,
    marginal_logtarget, ordered_logtarget, check_state

# A fresh implementation of the dependent slice-efficient transition of
# Kalli, Griffin and Walker (2011), Section 2. The algorithm and numerical
# precautions were audited against the HSNCP-consistency simulation source;
# see the example's provenance record. This module does not import that code.
#
# z contains ordered GEM indices, including gaps. The transition integrates
# out the previous sticks and slices before drawing the next block. At the
# allocation step only the finite prefix with residual < minimum(slice) is
# needed. There is no statistical truncation parameter.
mutable struct State{M<:ComponentModel,R<:AbstractRNG}
    model::M
    prior::DirichletProcess
    z::Vector{Int}
    theta::Vector{Float64}
    rng::R
    logweights::Vector{Float64}
    logslices::Vector{Float64}
    logresidual::Float64
    iteration::Int
    allocation_updates::Int
    stick_draws::Int
end

function initialize(model::ComponentModel, prior::PartitionPrior, seed;
        initial_labels=nothing)
    model isa GaussianLocation || throw(ArgumentError(
        "EfficientSlice currently requires GaussianLocation with a conjugate Normal base measure"))
    prior isa DirichletProcess || throw(ArgumentError(
        "EfficientSlice currently supports a DirichletProcess prior only"))
    rng=if seed isa AbstractRNG
        deepcopy(seed)
    elseif seed isa Integer && !(seed isa Bool) && seed>=0
        Xoshiro(seed)
    else
        throw(ArgumentError("seed must be a nonnegative integer or an AbstractRNG"))
    end
    labels=initial_labels===nothing ? ones(Int,length(model.y)) : collect(initial_labels)
    length(labels)==length(model.y) || throw(ArgumentError(
        "initial_labels must contain one ordered GEM index per observation"))
    all(j->j isa Integer && !(j isa Bool) && 0<j<typemax(Int),labels) ||
        throw(ArgumentError("initial_labels must be positive representable integer GEM indices"))
    z=Int.(labels)
    theta=[Float64(randbase(rng,model)) for _ in 1:maximum(z)]
    all(isfinite,theta) || error("initial component draw is not finite")
    State(model,prior,z,theta,rng,Float64[],Float64[],NaN,0,0,0)
end

occupied_count(s::State)=length(unique(s.z))
materialized_count(s::State)=length(s.theta)

function ordered_counts(s::State)
    counts=zeros(Int,maximum(s.z))
    for j in s.z
        counts[j]+=1
    end
    counts
end

function stick_shapes(s::State)
    counts=ordered_counts(s)
    tail=sum(counts)
    shapes=Vector{Tuple{Float64,Float64}}(undef,length(counts))
    alpha=strength(s.prior)
    for j in eachindex(counts)
        tail-=counts[j]
        shapes[j]=(1.0+counts[j],alpha+tail)
    end
    shapes
end

log1pexp(x)=max(x,0.0)+log1p(exp(-abs(x)))

"Draw log(V), log(1-V) without forming a Beta draw that can round to one."
function draw_logstick(rng::AbstractRNG,a::Real,b::Real)
    isfinite(a) && isfinite(b) && a>=1 && b>0 || throw(ArgumentError(
        "stick shapes must be finite with a >= 1 and b > 0"))
    if a==1
        lr=-randexp(rng)/b
        lv=log(-expm1(lr))
    else
        gx=rand(rng,Gamma(a,1.0))
        # Gamma(b) = Gamma(b+1) * Uniform(0,1)^(1/b), in log form.
        # This avoids a Gamma draw underflowing to zero when b is small.
        gy=rand(rng,Gamma(b<1 ? b+1 : b,1.0))
        gx>0 && gy>0 && isfinite(gx) && isfinite(gy) || error(
            "stick Gamma variates are not representable; change the parameter scale")
        lx=log(gx)
        ly=log(gy)-(b<1 ? randexp(rng)/b : 0.0)
        lv=-log1pexp(ly-lx)
        lr=-log1pexp(lx-ly)
    end
    isfinite(lv) && isfinite(lr) && lv<=0 && lr<0 || error(
        "stick logarithms are not representable; change the parameter scale")
    Float64(lv),Float64(lr)
end

function refresh_components!(s::State)
    counts=ordered_counts(s)
    resize!(s.theta,length(counts))
    for j in eachindex(counts)
        ids=findall(==(j),s.z)
        s.theta[j]=isempty(ids) ? randbase(s.rng,s.model) :
            randposterior(s.rng,s.model,ids)
        isfinite(s.theta[j]) || error("component draw is not finite")
    end
    s
end

function append_stick!(s::State,residual,a,b)
    length(s.logweights)<typemax(Int)-1 || error("GEM prefix length exceeds integer support")
    lv,lr=draw_logstick(s.rng,a,b)
    weight=residual+lv
    remaining=residual+lr
    isfinite(weight) && isfinite(remaining) && remaining<residual || error(
        "GEM residual cannot decrease at the current floating-point scale; sampling stopped without truncation")
    push!(s.logweights,weight)
    s.stick_draws+=1
    remaining
end

function draw_auxiliaries!(s::State)
    length(s.theta)==maximum(s.z) || throw(ArgumentError(
        "refresh_components! must discard the previous unused prefix before drawing auxiliaries"))
    empty!(s.logweights)
    residual=0.0
    for (a,b) in stick_shapes(s)
        residual=append_stick!(s,residual,a,b)
    end
    resize!(s.logslices,length(s.z))
    for i in eachindex(s.z)
        lw=s.logweights[s.z[i]]
        lu=lw-randexp(s.rng)
        isfinite(lu) && lu<lw || error(
            "slice logarithm cannot be represented strictly below its current weight")
        s.logslices[i]=lu
    end
    smallest=minimum(s.logslices)
    alpha=strength(s.prior)
    while residual>=smallest
        residual=append_stick!(s,residual,1.0,alpha)
        theta=Float64(randbase(s.rng,s.model))
        isfinite(theta) || error("unused component draw is not finite")
        push!(s.theta,theta)
    end
    s.logresidual=residual
    s
end

function allocation_logweights(s::State,i::Integer)
    checkbounds(s.z,i)
    length(s.logslices)==length(s.z) && length(s.logweights)==length(s.theta) ||
        throw(ArgumentError("draw slice auxiliaries before calculating allocation probabilities"))
    # The weight is an eligibility threshold here, not an additional factor.
    [s.logslices[i]<s.logweights[j] ?
        loglikelihood(s.model,s.model.y[i],s.theta[j]) : -Inf for j in eachindex(s.theta)]
end

function allocation_probabilities(s::State,i::Integer)
    lp=allocation_logweights(s,i)
    mx=maximum(lp)
    isfinite(mx) || error("no component with finite likelihood is eligible for observation $i")
    q=exp.(lp.-mx)
    q./sum(q)
end

function sweep!(s::State)
    check_state(s;auxiliaries=false) || throw(ArgumentError("invalid ordered-GEM sampler state"))
    refresh_components!(s)
    draw_auxiliaries!(s)
    for i in eachindex(s.z)
        q=allocation_probabilities(s,i)
        threshold=rand(s.rng)*sum(q)
        chosen=findlast(>(0),q)
        for j in eachindex(q)
            threshold-=q[j]
            if threshold<=0 && q[j]>0
                chosen=j
                break
            end
        end
        s.z[i]=chosen
        s.allocation_updates+=1
    end
    s.iteration+=1
    s
end

"Partition target with all weights and component parameters integrated out."
function marginal_logtarget(s::State)
    counts=filter(>(0),ordered_counts(s))
    value=logeppf(s.prior,counts)
    for j in unique(s.z)
        value+=logmarginal(s.model,findall(==(j),s.z))
    end
    value
end

"Ordered-label target retaining occupied atoms, integrating sticks and slices."
function ordered_logtarget(s::State)
    counts=ordered_counts(s)
    alpha=strength(s.prior)
    tail=sum(counts)
    value=0.0
    for j in eachindex(counts)
        tail-=counts[j]
        value+=log(alpha)+logbeta(1.0+counts[j],alpha+tail)
        if counts[j]>0
            value+=logbase(s.model,s.theta[j])
        end
    end
    value+sum(loglikelihood(s.model,s.model.y[i],s.theta[s.z[i]]) for i in eachindex(s.z))
end

function check_state(s::State;auxiliaries=true)
    length(s.z)==length(s.model.y) && !isempty(s.z) &&
        all(j->0<j<=length(s.theta),s.z) && all(isfinite,s.theta) || return false
    s.iteration>=0 && s.allocation_updates>=0 && s.stick_draws>=0 || return false
    auxiliaries || return true
    length(s.logslices)==length(s.z) && length(s.logweights)==length(s.theta) &&
        all(isfinite,s.logslices) && all(isfinite,s.logweights) &&
        all(<=(0),s.logweights) && isfinite(s.logresidual) &&
        s.logresidual<minimum(s.logslices) || return false
    all(s.logslices[i]<s.logweights[s.z[i]] for i in eachindex(s.z))
end

end
