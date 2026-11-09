module MFMLikelihoodTempering

using GeneralDPPMixtures, Random, Distributions
import ..MFMBaseline as MF
import ..MFMTarget as MT
const G=GeneralDPPMixtures

export allocation!, hot_sweep!, loglikelihood, logtarget, swap_logratio, attempt_swap!

function checked_beta(beta::Real)
    b=Float64(beta)
    isfinite(b) && b>=0 || throw(ArgumentError("likelihood power must be finite and nonnegative"))
    b
end

loglikelihood(s::MF.State)=MT.likelihood(s)

"""
The NIW-collapsed likelihood is powered AFTER integrating the means and
covariances. The allocation/count/hyperparameter prior is unchanged. At beta=1
this is the original MFM target; the point-process model fields are unused.
"""
function logtarget(s::MF.State,beta::Real)
    b=checked_beta(beta)
    b==1. && return MT.logtarget(s)
    MT.logtarget(s)+(b-1.)*loglikelihood(s)
end

# Called after removing one observation and deleting an empty active cluster.
# The new-cluster NIW predictive is its singleton evidence. All unoccupied
# component labels share this evidence and have combined mass (M-k)*gamma.
function allocation_prior_scores!(scores,s)
    k=length(s.active); canbirth=k<s.M
    resize!(scores,k+Int(canbirth))
    for (j,h) in enumerate(s.active)
        scores[j]=log(s.clusters[h].n+s.gamma)
    end
    empty=canbirth ? MF.empty_slot!(s) : 0
    canbirth && (scores[end]=log(s.M-k)+log(s.gamma))
    empty
end

function allocation_scores!(scores,s,beta,i)
    empty=allocation_prior_scores!(scores,s)
    if beta!=0.
        for (j,h) in enumerate(s.active)
            scores[j]+=beta*MF.logpred(s,h,s.model.y[i])
        end
        empty!=0 && (scores[end]+=beta*MF.logpred(s,empty,s.model.y[i]))
    end
    empty
end

"""A systematic exact allocation Gibbs scan at fixed M, gamma and Lambda."""
function allocation!(s::MF.State,beta::Real)
    b=checked_beta(beta); scores=Float64[]
    for i in eachindex(s.c)
        old=s.c[i]
        G.remove!(s.clusters[old],s.model.y[i]); MF.refresh!(s,old)
        s.clusters[old].n==0 && deleteat!(s.active,findfirst(==(old),s.active))
        k=length(s.active); empty=allocation_scores!(scores,s,b,i)
        j=G.draw_logweights!(s.rng,scores)
        h=j<=k ? s.active[j] : empty
        j>k && push!(s.active,h)
        s.c[i]=h
        G.add!(s.clusters[h],s.model.y[i]); MF.refresh!(s,h)
    end
    s
end

# L(c) does not depend on M, gamma or Lambda conditional on the partition.
# Consequently the original unbounded Beta/Poisson M update, Gamma Lambda
# update and log-random-walk gamma MH update remain exact at every beta.
function hyperparameters!(s)
    rng=s.rng
    logz=G.logbeta_draw(rng,s.M*s.gamma,length(s.c)); k=length(s.active)
    t=s.lambda*exp(s.gamma*logz)
    s.M=k+rand(rng,Poisson(t))+Int(rand(rng)<t/(k+t))
    s.lambda=rand(rng,Gamma(s.M,1/1.1))
    candidate=exp(log(s.gamma)+0.25randn(rng)); s.gamma_attempts+=1
    lr=MF.gamma_logtarget(s,candidate)-MF.gamma_logtarget(s,s.gamma)+log(candidate/s.gamma)
    if log(rand(rng))<lr
        s.gamma=candidate; s.gamma_accepts+=1
    end
    s
end

"""
One exact collapsed likelihood-power sweep. beta=1 calls MF.sweep! directly,
preserving its complete state/RNG trajectory. An optional gamma slice update
may be composed by the caller at any temperature, since its conditional is
unchanged. No block proposal is performed here.
"""
function hot_sweep!(s::MF.State,beta::Real)
    b=checked_beta(beta)
    b==1. && return MF.sweep!(s)
    allocation!(s,b)
    hyperparameters!(s)
end

function compatible(a,b)
    am=a.model; bm=b.model
    (am.y===bm.y || am.y==bm.y) && am.omega==bm.omega && am.tau==bm.tau &&
        a.mu0==b.mu0 && a.kappa==b.kappa
end

"""Log MH ratio for swapping two physical states of the same MFM model."""
function swap_logratio(a::MF.State,b::MF.State,beta_a::Real,beta_b::Real)
    ba=checked_beta(beta_a); bb=checked_beta(beta_b)
    compatible(a,b) || throw(ArgumentError("swap states have different MFM targets"))
    (ba-bb)*(loglikelihood(b)-loglikelihood(a))
end

"""
Attempt a symmetric swap between two fixed temperature slots. Whole State
objects are exchanged, including their independent RNG streams and counters;
there is no RNG reset or sharing. A separate caller-owned RNG makes the swap
decision. Checkpoint the states, temperature vector, and this swap RNG together
for exact resumption. Independent diagnostic chains require independent ladders
and swap RNGs; only the beta=1 slot has the original posterior target.

Returns (accepted, logratio, logu, i, j). Slot/replica provenance tags, if used,
must be exchanged by the caller exactly when accepted is true.
"""
function attempt_swap!(states::AbstractVector,i::Integer,j::Integer,betas::AbstractVector,rng)
    length(states)==length(betas) || throw(ArgumentError("state/temperature counts differ"))
    checkbounds(states,i); checkbounds(states,j)
    i!=j || throw(ArgumentError("swap slots must be distinct"))
    states[i]!==states[j] || throw(ArgumentError("swap slots alias the same state"))
    ratio=swap_logratio(states[i],states[j],betas[i],betas[j])
    logu=log(rand(rng)); accepted=logu<min(0.,ratio)
    accepted && ((states[i],states[j])=(states[j],states[i]))
    (accepted=accepted,logratio=ratio,logu=logu,i=Int(i),j=Int(j))
end

end
