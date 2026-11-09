module MFMConcentrationSplitMerge

using GeneralDPPMixtures, Random, SpecialFunctions
import ..MFMBaseline as MF
import ..MFMTarget as MT
import ..MFMBlockMoves as MB
import ..MFMSequentialSplitMerge as SS
const G=GeneralDPPMixtures

export Workspace,Proposal,propose!,raw_ratio,commit!,attempt!

"""Stateless anchor-only sequential joint count/concentration proposal."""
struct Workspace end
Workspace(::G.Model)=Workspace()

"""A joint-M block proposal plus its deterministic gamma transform."""
struct Proposal{V}
    base::MB.Proposal
    newgamma::Float64
    eta::Float64
    logjacobian::Float64
    original_labels::Vector{Int}
    original_active::Vector{Int}
    original_mu0::V
    original_kappa::Float64
end
function Base.getproperty(p::Proposal,name::Symbol)
    name in fieldnames(typeof(p)) ? getfield(p,name) : getproperty(getfield(p,:base),name)
end
Base.propertynames(p::Proposal,private::Bool=false)=(fieldnames(typeof(p))...,fieldnames(MB.Proposal)...)

function fullgroups(s,p)
    unaffected=[findall(==(h),s.c) for h in s.active if !(h in p.oldids)]
    vcat(unaffected,p.oldgroups),vcat(unaffected,p.newgroups)
end

"""
Uniform ordered observation anchors select a local split or merge. The move
changes M by the same +/-1 as K and maps gamma to M*gamma/newM, preserving the
total Dirichlet concentration eta to floating-point precision. Lambda stays
fixed. A split's partial sequential scan uses destination gamma; a merge's
reverse forced split uses the original gamma. The same random permutation is
the reverse auxiliary. The MH density is with respect to dgamma, so its
Jacobian is M/newM. No additional component-count or anchor-size factor occurs.

Only NIW evidence is powered by beta. Full gamma-dependent size terms for ALL
occupied clusters and the Exp(rate=.1) gamma prior remain in the ratio. The
operator has no artificial count bound; ordinary baseline moves compose with
it to explore total concentration and unbounded M. `pair` and `order` are
proposal-only overrides of the uniform production auxiliary law.
"""
function propose!(::Workspace,s::MF.State,rng;pair=nothing,order=nothing,beta=1.)
    beta=MB.checkbeta(beta);n=length(s.c);n<2 && return nothing
    k=length(s.active);1<=k<=s.M || error("occupied MFM count exceeds M")
    isfinite(s.gamma) && s.gamma>0 && isfinite(s.lambda) && s.lambda>0 || error("invalid MFM hyperparameters")
    eta=s.M*s.gamma;isfinite(eta) && eta>0 || error("invalid total concentration")
    i,j=if pair===nothing
        a=rand(rng,1:n);b=rand(rng,1:n-1);(a,b>=a ? b+1 : b)
    else
        pair
    end
    1<=i<=n && 1<=j<=n && i!=j || throw(ArgumentError("invalid concentration split/merge pair"))
    ids=unique([s.c[i],s.c[j]]);split=length(ids)==1
    D=sort([t for t in eachindex(s.c) if s.c[t] in ids])
    newk=k+(split ? 1 : -1);newM=Base.checked_add(s.M,split ? 1 : -1)
    1<=newk<=newM || error("invalid joint count transform")
    newgamma=eta/newM;isfinite(newgamma) && newgamma>0 || error("invalid transformed gamma")
    oldlabels=split ? [1,1] : [1,2];newlabels=split ? [1,2] : [1,1]
    oo=order===nothing ? shuffle(rng,setdiff(D,[i,j])) : Int.(copy(order))
    kind=split ? :local_joint_split : :local_joint_merge
    base=MB.makeproposal(s,kind,(i,j),ids,[i,j],Int[],oldlabels,newlabels,oo,newk,D;newM,beta)
    if split
        base.newgroups,q=SS.sequential_split!(s,D,[i,j],oo,rng;beta,gamma=newgamma)
        base.logforward=q;base.logreverse=0.
    else
        forced=zeros(Int,n)
        for h in eachindex(base.oldgroups);forced[base.oldgroups[h]].=h;end
        _,q=SS.sequential_split!(s,D,[i,j],oo,rng;forced,beta,gamma=s.gamma)
        base.newgroups=[D];base.logforward=0.;base.logreverse=q
    end
    p=Proposal(base,newgamma,eta,log(Float64(s.M))-log(Float64(newM)),copy(s.c),copy(s.active),s.mu0,s.kappa)
    oldgroups,newgroups=fullgroups(s,p)
    # Counts of unaffected groups also change their gamma factors. Evidence is
    # gamma-independent, so only affected groups need new NIW factorizations.
    base.Gterm=sum(loggamma(length(g)+newgamma)-loggamma(newgamma) for g in newgroups)-
        sum(loggamma(length(g)+s.gamma)-loggamma(s.gamma) for g in oldgroups)
    base.H=sum(MT.log_marginal(s,g) for g in base.newgroups)-sum(MT.log_marginal(s,g) for g in base.oldgroups)
    base.countterm=MT.logfallingfactorial(newM,newk)-MT.logfallingfactorial(s.M,k)+
        loggamma(newM*newgamma)-loggamma(newM*newgamma+n)-loggamma(eta)+loggamma(eta+n)+
        (newM-s.M)*log(s.lambda)-loggamma(newM)+loggamma(s.M)-.1*(newgamma-s.gamma)
    base.selectionterm=0.;base.launch_component=1
    base.component_logforward=[base.logforward];base.component_logreverse=[base.logreverse]
    mapped=split ? [ids[1],0] : [ids[1]]
    base.moved_observations=sum(s.c[t]!=mapped[h] for h in eachindex(base.newgroups) for t in base.newgroups[h])
    base.ratio=base.Gterm+beta*base.H+base.countterm+base.logreverse-base.logforward+p.logjacobian
    isfinite(base.ratio) || error("nonfinite concentration split/merge ratio")
    base.logu=log(rand(rng));base.outcome=base.logu<min(0.,base.ratio) ? :accepted : :rejected
    p
end

"""Independent complete canonical targets, including all gamma size terms."""
function raw_ratio(s::MF.State,p::Proposal)
    p.logforward===nothing && throw(ArgumentError("proposal has no endpoint density"))
    oldgroups,newgroups=fullgroups(s,p)
    oldtarget=MT.logtarget(s;groups=oldgroups,M=p.M,gamma=p.gamma,lambda=p.lambda)
    newtarget=MT.logtarget(s;groups=newgroups,M=p.newM,gamma=p.newgamma,lambda=p.lambda)
    oldL=sum(MT.log_marginal(s,g) for g in oldgroups)
    newL=sum(MT.log_marginal(s,g) for g in newgroups)
    difference=newtarget-oldtarget+(p.beta-1)*(newL-oldL)
    (target_difference=newtarget-oldtarget,tempered_target_difference=difference,
        log_forward=p.logforward,log_reverse=p.logreverse,log_jacobian=p.logjacobian,
        raw=difference+p.logreverse-p.logforward+p.logjacobian)
end

"""Build every cache before assigning transformed gamma; NIW caches ignore gamma."""
function commit!(s::MF.State,p::Proposal)
    p.outcome==:accepted || return s
    s.M==p.M && s.gamma==p.gamma && s.lambda==p.lambda || error("hyperparameters changed between proposal and commit")
    s.c==p.original_labels && s.active==p.original_active && s.mu0==p.original_mu0 && s.kappa==p.original_kappa ||
        error("partition or NIW prior changed between proposal and commit")
    isfinite(p.newgamma) && p.newgamma>0 && isapprox(p.newM*p.newgamma,p.eta;rtol=8eps(Float64),atol=0.) ||
        error("invalid concentration transform")
    # MB.commit! checks counts/complete partition and finishes all fallible
    # cluster and Cholesky work before touching s. Its predictive caches have
    # no gamma dependence. The final Float64 gamma assignment cannot fail.
    MB.commit!(s,p.base)
    s.gamma=p.newgamma
    s
end
function attempt!(w::Workspace,s::MF.State,rng;observer=nothing,kwargs...)
    p=propose!(w,s,rng;kwargs...);p===nothing && return nothing
    observer===nothing || observer(s,p)
    commit!(s,p);p
end

end
