module NealSamplers

# Neal (2000), "Markov chain sampling methods for Dirichlet process mixture
# models", Algorithms 2, 3 and 8. The author's corresponding preprint gives
# the updates on pages 6 and 13: https://glizen.com/radfordneal/ftp/mixmc.pdf

using Random, SpecialFunctions
import ..SpeciesSampling as S
import ..SpeciesSampling: loglikelihood, logbase, randbase, collapsed_logtarget, check_state
import ..GeneralDPPMixtures: draw_logweights!

export PoissonLogRate, State, initialize, neal2_sweep!, neal3_sweep!, neal8_sweep!,
    neal2_reassign!, neal8_reassign!, refresh_atoms!, joint_logtarget,
    collapsed_logtarget, check_state, auxiliary_logweights, atom_logconditional,
    atom_mh_logratio

# The component methods remain in NealSamplers so public type identities and
# method ownership are unchanged by the source layout.
include("../../../components/poisson_lognormal.jl")
include("../retained_state.jl")
include("../retained_targets.jl")

function _canonicalize!(s::State)
    order = unique(s.z)
    s.theta = s.theta[order]
    s.z = S.canonical_labels(s.z)
    s
end

function _remove!(s::State,i::Integer)
    old = s.z[i]
    singleton = count(==(old),s.z) == 1
    previous = s.theta[old]
    s.z[i] = 0
    if singleton
        deleteat!(s.theta,old)
        for j in eachindex(s.z)
            s.z[j] > old && (s.z[j] -= 1)
        end
    end
    singleton,previous
end

function _insert!(s::State,i::Integer,selected::Integer,newtheta)
    if selected > length(s.theta)
        push!(s.theta,newtheta)
        selected = length(s.theta)
    end
    s.z[i] = selected
    _canonicalize!(s)
end

function _index(s::State,i)
    i isa Integer && !(i isa Bool) && 1<=i<=length(s.z) ||
        throw(ArgumentError("observation index is out of range"))
    Int(i)
end

"""Algorithm 2: existing atoms use likelihoods; a birth uses the exact base predictive."""
function neal2_reassign!(s::State,i)
    _conjugate(s.model) || throw(ArgumentError("Neal 2 requires an implemented exact predictive and posterior draw"))
    i = _index(s,i)
    _remove!(s,i)
    k = length(s.theta)
    scores = [log(count(==(h),s.z)) + loglikelihood(s.model,s.model.y[i],s.theta[h]) for h in 1:k]
    push!(scores,log(S.strength(s.prior)) + S.logmarginal(s.model,[i]))
    selected = draw_logweights!(s.rng,scores)
    theta = selected > k ? S.randposterior(s.rng,s.model,[i]) : 0.0
    _insert!(s,i,selected,theta)
end

"""Log weights conditional on occupied atoms and a supplied set of auxiliary atoms."""
function auxiliary_logweights(model::S.ComponentModel, prior::S.DirichletProcess,
        y, counts, theta, auxiliary)
    length(counts) == length(theta) && all(>(0),counts) && !isempty(auxiliary) ||
        throw(ArgumentError("occupied counts and auxiliary atoms are inconsistent"))
    scores = [log(counts[h]) + loglikelihood(model,y,theta[h]) for h in eachindex(theta)]
    append!(scores,[log(S.strength(prior))-log(length(auxiliary)) + loglikelihood(model,y,t) for t in auxiliary])
    scores
end

function _auxiliary_count(auxiliaries)
    auxiliaries isa Integer && !(auxiliaries isa Bool) && 1<=auxiliaries<=typemax(Int) ||
        throw(ArgumentError("auxiliaries must be a positive integer"))
    Int(auxiliaries)
end

"""Algorithm 8 allocation update, retaining a singleton's old atom among the auxiliaries."""
function neal8_reassign!(s::State,i; auxiliaries=3)
    m = _auxiliary_count(auxiliaries)
    i = _index(s,i)
    singleton,previous = _remove!(s,i)
    auxiliary = singleton ? [previous] : Float64[]
    append!(auxiliary,[randbase(s.rng,s.model) for _ in 1:m-length(auxiliary)])
    counts = [count(==(h),s.z) for h in eachindex(s.theta)]
    scores = auxiliary_logweights(s.model,s.prior,s.model.y[i],counts,s.theta,auxiliary)
    selected = draw_logweights!(s.rng,scores)
    k = length(s.theta)
    newtheta = selected > k ? auxiliary[selected-k] : 0.0
    _insert!(s,i,selected,newtheta)
    # Return the actual temporary representation for independent replay checks.
    (; singleton, auxiliary, counts, selected)
end

"""Conditional atom log density, including base and likelihood normalisers."""
function atom_logconditional(s::State,h::Integer,theta)
    1<=h<=length(s.theta) || throw(ArgumentError("atom index is out of range"))
    logbase(s.model,theta) +
        sum(loglikelihood(s.model,s.model.y[i],theta) for i in eachindex(s.z) if s.z[i]==h)
end

"""
Symmetric random-walk ratio, excluding constants that cancel. For Poisson
counts, a centred expm1 calculation also avoids losing the second-order term
when an atom close to its likelihood mode receives a small proposal.
"""
function atom_mh_logratio(s::State,h::Integer,proposed)
    1<=h<=length(s.theta) || throw(ArgumentError("atom index is out of range"))
    if s.model isa PoissonLogRate
        isfinite(proposed) || return -Inf
        old=s.theta[h]
        delta=proposed-old
        isfinite(delta) || return -Inf
        newrate=exp(proposed)
        isfinite(newrate) || return -Inf
        oldrate=exp(old)
        ids=findall(==(h),s.z)
        total=sum(s.model.y[i] for i in ids)
        expected=length(ids)*oldrate
        likelihood_change=if abs(delta)<0.5
            delta*(total-expected)-expected*(expm1(delta)-delta)
        else
            total*delta-length(ids)*(newrate-oldrate)
        end
        standard_delta=delta/s.model.base_sd
        standard_old=(old-s.model.base_mean)/s.model.base_sd
        return likelihood_change-standard_delta*(standard_old+0.5standard_delta)
    end
    atom_logconditional(s,h,proposed)-atom_logconditional(s,h,s.theta[h])
end

function refresh_atoms!(s::State)
    for h in eachindex(s.theta)
        if _conjugate(s.model)
            s.theta[h] = S.randposterior(s.rng,s.model,findall(==(h),s.z))
        elseif s.model isa PoissonLogRate
            proposed = s.theta[h] + s.model.proposal_sd * randn(s.rng)
            ratio = atom_mh_logratio(s,h,proposed)
            s.atom_attempts += 1
            if log(rand(s.rng)) < min(0.0,ratio)
                s.theta[h] = proposed
                s.atom_accepts += 1
            end
        else
            throw(ArgumentError("no atom update is implemented for $(typeof(s.model))"))
        end
    end
    s
end

function neal2_sweep!(s::State)
    _conjugate(s.model) || throw(ArgumentError("Neal 2 requires an implemented exact predictive and posterior draw"))
    for i in eachindex(s.z); neal2_reassign!(s,i); end
    refresh_atoms!(s)
    s.sweeps += 1
    s
end

"""Algorithm 3 is the shared collapsed DP sweep; component atoms are integrated out."""
function neal3_sweep!(s::S.State)
    s.prior isa S.DirichletProcess ||
        throw(ArgumentError("the named Neal 3 kernel requires DirichletProcess"))
    _conjugate(s.model) || throw(ArgumentError("Neal 3 requires an implemented analytic collapsed likelihood"))
    S.collapsed_sweep!(s)
end

function neal8_sweep!(s::State; auxiliaries=3)
    m = _auxiliary_count(auxiliaries)
    for i in eachindex(s.z); neal8_reassign!(s,i;auxiliaries=m); end
    refresh_atoms!(s)
    s.sweeps += 1
    s
end


end
