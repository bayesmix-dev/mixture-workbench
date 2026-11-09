module MFMBlockMoves

using GeneralDPPMixtures, LinearAlgebra, Random, SpecialFunctions
import ..MFMBaseline as MF
import ..MFMTarget as MT
import ..DPPMultiReallocation as MR
const G=GeneralDPPMixtures

export Workspace, Proposal, propose!, attempt!, commit!, restricted_scan!,
       mixture_logprob, raw_ratio, likelihood, logtarget, partition_logtarget

"""Frozen reference launches; blocks=2 selects pairs, blocks=0 all clusters."""
struct Workspace
    bank::MR.Workspace
end
Workspace(m::G.Model;references=Vector{Vector{Int}}(),blocks=2,conflict_policy=:fallback)=
    Workspace(MR.Workspace(m;references,blocks,conflict_policy))

likelihood(s::MF.State)=sum(MT.log_marginal(s,s.clusters[h]) for h in s.active)
logtarget(s::MF.State)=MT.logtarget(s)
partition_logtarget(s::MF.State;kwargs...)=MT.partition_logtarget(s;kwargs...)

mutable struct Proposal
    kind::Symbol
    pair::Union{Nothing,Tuple{Int,Int}}
    anchors::Vector{Int}
    otheranchors::Vector{Int}
    oldanchorlabels::Vector{Int}
    newanchorlabels::Vector{Int}
    order::Vector{Int}
    oldids::Vector{Int}
    oldgroups::Vector{Vector{Int}}
    newgroups::Vector{Vector{Int}}
    k::Int
    newk::Int
    M::Int
    newM::Int
    gamma::Float64
    lambda::Float64
    beta::Float64
    union_size::Int
    moved_observations::Int
    launch_component::Int
    component_logforward::Vector{Float64}
    component_logreverse::Vector{Float64}
    Gterm::Union{Nothing,Float64}
    H::Union{Nothing,Float64}
    countterm::Union{Nothing,Float64}
    selectionterm::Union{Nothing,Float64}
    logforward::Union{Nothing,Float64}
    logreverse::Union{Nothing,Float64}
    ratio::Union{Nothing,Float64}
    logu::Union{Nothing,Float64}
    outcome::Symbol
end

function cluster(s,members)
    c=G.Cluster(s.model,zero(s.mu0))
    for i in sort(members); G.add!(c,s.model.y[i]); end
    c
end

# A temporary MF.State supplies the independently implemented NIW predictive
# caches. Its RNG is never used; refresh! mutates only its own predictive arrays.
function cache_state(s,clusters,labels,active)
    temporary=MF.State(s.model,clusters,labels,active,s.mu0,s.kappa,
        [s.model.omegaL for _ in clusters],[s.mu0 for _ in clusters],zeros(length(clusters)),
        s.M,s.gamma,s.lambda,s.rng,s.gamma_attempts,s.gamma_accepts)
    for h in eachindex(clusters); MF.refresh!(temporary,h); end
    temporary
end

"""
One NIW-collapsed restricted Gibbs scan from a deterministic reference launch.
Every output cluster has a locked anchor; global merges lock the primary pair
to the same cluster. Scores are log(n_h+gamma)+log NIW posterior predictive.
The output cluster count is fixed within this proposal, so no birth mass is
included here. The falling-factorial count correction enters the MH ratio.
"""
function checkbeta(beta)
    beta isa Real && isfinite(beta) && 0<=beta<=1 || throw(ArgumentError("beta must be finite and in [0,1]"))
    Float64(beta)
end

function restricted_scan!(s::MF.State,D,anchors,order,rng,initial;
        forced=nothing,anchorlabels=collect(eachindex(anchors)),beta=1.)
    beta=checkbeta(beta)
    r=maximum(anchorlabels); n=length(s.c)
    length(anchorlabels)==length(anchors) && sort(unique(anchorlabels))==collect(1:r) || throw(ArgumentError("invalid anchor labels"))
    length(unique(anchors))==length(anchors) && all(in(D),anchors) || throw(ArgumentError("invalid anchors"))
    sort(order)==sort(setdiff(D,anchors)) || throw(ArgumentError("invalid restricted scan order"))
    length(initial)==n && all(i->initial[i] in 1:r,D) || throw(ArgumentError("invalid launch partition"))
    all(a->initial[anchors[a]]==anchorlabels[a],eachindex(anchors)) || throw(ArgumentError("unoriented launch"))
    if forced!==nothing
        length(forced)==n && all(i->forced[i] in 1:r,D) &&
            all(a->forced[anchors[a]]==anchorlabels[a],eachindex(anchors)) || throw(ArgumentError("invalid forced partition"))
    end
    labels=copy(initial)
    groups=[cluster(s,[i for i in D if labels[i]==h]) for h in 1:r]
    cached=cache_state(s,groups,labels,collect(1:r)); scores=zeros(r); lp=0.
    for i in order
        old=labels[i]; G.remove!(groups[old],s.model.y[i]); MF.refresh!(cached,old)
        for h in 1:r
            scores[h]=log(groups[h].n+s.gamma)+beta*MF.logpred(cached,h,s.model.y[i])
        end
        mx=maximum(scores); lognorm=mx+log(sum(exp(x-mx) for x in scores))
        h=forced===nothing ? G.draw_weights(rng,exp.(scores.-mx)) : forced[i]
        lp+=scores[h]-lognorm
        labels[i]=h; G.add!(groups[h],s.model.y[i]); MF.refresh!(cached,h)
    end
    [[i for i in D if labels[i]==h] for h in 1:r],lp
end

function mixture_logprob(s,D,anchors,order,rng,bank,forced;anchorlabels=collect(eachindex(anchors)),beta=1.)
    beta=checkbeta(beta)
    components=if forced===nothing
        # An unforced scan consumes randomness separately for every component.
        [last(restricted_scan!(s,D,anchors,order,rng,launch;forced,anchorlabels,beta)) for launch in bank]
    else
        # Forced scans are deterministic and read-only. Keep every component
        # in its original position, including duplicate launch multiplicities,
        # so logmeanexp executes exactly the original arithmetic. Full-vector
        # equality (not a hash or only labels on D) identifies per-call reuse.
        values=Vector{Float64}(undef,length(bank));distinct=Int[]
        for (j,launch) in enumerate(bank)
            match=findfirst(i->bank[i]==launch,distinct)
            if match===nothing
                values[j]=last(restricted_scan!(s,D,anchors,order,rng,launch;forced,anchorlabels,beta))
                push!(distinct,j)
            else
                values[j]=values[distinct[match]]
            end
        end
        values
    end
    MR.logmeanexp(components),components
end

function makeproposal(s,kind,pair,ids,anchors,otheranchors,oldlabels,newlabels,order,newk,D;newM=s.M,beta=1.)
    Proposal(kind,pair,anchors,otheranchors,oldlabels,newlabels,order,ids,
        [findall(==(h),s.c) for h in ids],Vector{Int}[],length(s.active),newk,s.M,newM,s.gamma,s.lambda,checkbeta(beta),
        length(D),0,0,Float64[],Float64[],nothing,nothing,nothing,nothing,nothing,nothing,
        nothing,nothing,:constructed)
end

islocal(p::Proposal)=p.kind in (:local_split,:local_merge,:local_joint_split,:local_joint_merge)

function finish_proposal!(w,s,p,D,rng;global_move=false,local_move=false)
    # A local union may intersect any number of reference blocks. Do not use
    # the all-data bank's global occupied-count filter for this proposal.
    bank=local_move && w.bank.blocks==0 ? MR.Workspace(w.bank.references,w.bank.whitened,2,w.bank.conflict_policy) : w.bank
    forward=MR.launches(bank,s.model,D,p.anchors;anchorlabels=p.newanchorlabels)
    reverse=MR.launches(bank,s.model,D,p.anchors;anchorlabels=p.oldanchorlabels)
    p.launch_component=rand(rng,eachindex(forward))
    p.newgroups,_=restricted_scan!(s,D,p.anchors,p.order,rng,forward[p.launch_component];anchorlabels=p.newanchorlabels,beta=p.beta)
    oldlabels=zeros(Int,length(s.c)); newlabels=copy(oldlabels)
    for h in eachindex(p.oldgroups); oldlabels[p.oldgroups[h]].=h; end
    for h in eachindex(p.newgroups); newlabels[p.newgroups[h]].=h; end
    p.logforward,p.component_logforward=mixture_logprob(s,D,p.anchors,p.order,rng,forward,newlabels;anchorlabels=p.newanchorlabels,beta=p.beta)
    p.logreverse,p.component_logreverse=mixture_logprob(s,D,p.anchors,p.order,rng,reverse,oldlabels;anchorlabels=p.oldanchorlabels,beta=p.beta)
    oldclusters=[cluster(s,g) for g in p.oldgroups]
    newclusters=[cluster(s,g) for g in p.newgroups]
    p.Gterm=sum(loggamma(s.gamma+c.n)-loggamma(s.gamma) for c in newclusters)-
        sum(loggamma(s.gamma+c.n)-loggamma(s.gamma) for c in oldclusters)
    p.H=sum(MT.log_marginal(s,c) for c in newclusters)-sum(MT.log_marginal(s,c) for c in oldclusters)
    # Canonical partition target has M!/(M-k)! assignments of component labels.
    if p.newM==p.M
        p.countterm=p.newk>p.k ? log(s.M-p.k) : p.newk<p.k ? -log(s.M-p.k+1) : 0.
    else
        # Joint moves change the full M-dependent allocation normalization and
        # the shifted-Poisson prior as well as the occupied-label multiplicity.
        unaffected=local_move ? [findall(==(h),s.c) for h in s.active if !(h in p.oldids)] : Vector{Int}[]
        fullchange=MT.logtarget(s;groups=vcat(unaffected,p.newgroups),M=p.newM)-
            MT.logtarget(s;groups=vcat(unaffected,p.oldgroups),M=p.M)
        p.countterm=fullchange-p.Gterm-p.H
    end
    if local_move
        # The same uniform ordered observation pair selects this union in the
        # reverse direction, and its scan order is uniform on the same set.
        # No cluster-size selection factor is present.
        oldother=Vector{Int}[]; newother=Vector{Int}[]
        matchedids=p.newk>p.k ? [p.oldids[1],0] : p.oldids[1:1]
    elseif global_move
        split=p.newk>p.k
        oldother=split ? p.oldgroups[2:end] : p.oldgroups[3:end]
        newother=split ? p.newgroups[3:end] : p.newgroups[2:end]
        matchedids=split ? vcat([p.oldids[1],0],p.oldids[2:end]) : vcat(p.oldids[1:1],p.oldids[3:end])
    else
        oldother=p.oldgroups; newother=p.newgroups; matchedids=p.oldids
    end
    p.selectionterm=sum(log(length(g)) for g in oldother;init=0.)-
        sum(log(length(g)) for g in newother;init=0.)
    p.moved_observations=sum(s.c[i]!=matchedids[h] for h in eachindex(p.newgroups) for i in p.newgroups[h])
    p.ratio=p.Gterm+p.beta*p.H+p.countterm+p.selectionterm+p.logreverse-p.logforward
    isfinite(p.ratio) || error("nonfinite NIW-collapsed MFM block acceptance ratio")
    p.logu=log(rand(rng)); p.outcome=p.logu<min(0.,p.ratio) ? :accepted : :mh_rejected
    p
end

function propose_reallocation!(w,s,rng;ids=nothing,anchors=nothing,order=nothing,beta=1.)
    k=length(s.active); k<2 && return nothing
    k<=s.M || error("occupied MFM count exceeds M")
    r=w.bank.blocks==0 ? k : min(w.bank.blocks,k)
    selected=ids===nothing ? sort(shuffle(rng,s.active)[1:r]) : Int.(copy(ids))
    length(selected)==r && length(unique(selected))==r && all(in(s.active),selected) || throw(ArgumentError("invalid selected clusters"))
    oldgroups=[findall(==(h),s.c) for h in selected]
    aa=anchors===nothing ? [rand(rng,g) for g in oldgroups] : Int.(copy(anchors))
    length(aa)==r && all(h->aa[h] in oldgroups[h],1:r) || throw(ArgumentError("anchors do not match selected clusters"))
    D=sort(vcat(oldgroups...)); oo=order===nothing ? shuffle(rng,setdiff(D,aa)) : Int.(copy(order))
    p=makeproposal(s,r==2 ? :reallocate : :multi_reallocate,nothing,selected,aa,Int[],collect(1:r),collect(1:r),oo,k,D;beta)
    finish_proposal!(w,s,p,D,rng)
end

function propose_global!(w,s,rng;pair=nothing,otheranchors=nothing,order=nothing,joint=false,beta=1.)
    n=length(s.c); n<2 && return nothing
    k=length(s.active); k<=s.M || error("occupied MFM count exceeds M")
    i,j=if pair===nothing
        a=rand(rng,1:n); b=rand(rng,1:n-1); (a,b>=a ? b+1 : b)
    else
        pair
    end
    1<=i<=n && 1<=j<=n && i!=j || throw(ArgumentError("invalid global pair"))
    primary=unique([s.c[i],s.c[j]]); split=length(primary)==1
    remaining=setdiff(s.active,primary); D=collect(1:n)
    if split && k==s.M && !joint
        p=makeproposal(s,:global_split,(i,j),vcat(primary,remaining),[i,j],Int[],Int[],Int[],Int[],k+1,D;beta)
        p.outcome=:capacity; return p
    end
    others,aa=if otheranchors===nothing
        chosen=shuffle(rng,remaining)
        chosen,[rand(rng,findall(==(h),s.c)) for h in chosen]
    else
        a=Int.(copy(otheranchors)); chosen=[s.c[t] for t in a if 1<=t<=n]
        length(a)==length(remaining) && length(chosen)==length(a) &&
            length(unique(chosen))==length(chosen) && sort(chosen)==sort(remaining) || throw(ArgumentError("invalid other-cluster anchors"))
        chosen,a
    end
    ids=vcat(primary,others); anchors=vcat([i,j],aa)
    oldlabels=split ? vcat([1,1],collect(2:k)) : collect(1:k)
    newlabels=split ? collect(1:k+1) : vcat([1,1],collect(2:k-1))
    oo=order===nothing ? shuffle(rng,setdiff(D,anchors)) : Int.(copy(order))
    kind=joint ? (split ? :global_joint_split : :global_joint_merge) : (split ? :global_split : :global_merge)
    p=makeproposal(s,kind,(i,j),ids,anchors,aa,oldlabels,newlabels,oo,k+(split ? 1 : -1),D;
        newM=joint ? s.M+(split ? 1 : -1) : s.M,beta)
    finish_proposal!(w,s,p,D,rng;global_move=true)
end

function propose_local!(w,s,rng;pair=nothing,order=nothing,joint=false,beta=1.)
    n=length(s.c); n<2 && return nothing
    k=length(s.active); k<=s.M || error("occupied MFM count exceeds M")
    i,j=if pair===nothing
        a=rand(rng,1:n); b=rand(rng,1:n-1); (a,b>=a ? b+1 : b)
    else
        pair
    end
    1<=i<=n && 1<=j<=n && i!=j || throw(ArgumentError("invalid local pair"))
    ids=unique([s.c[i],s.c[j]]); split=length(ids)==1
    D=sort([t for t in eachindex(s.c) if s.c[t] in ids])
    kind=joint ? (split ? :local_joint_split : :local_joint_merge) : (split ? :local_split : :local_merge)
    oldlabels=split ? [1,1] : [1,2]; newlabels=split ? [1,2] : [1,1]
    newk=k+(split ? 1 : -1); newM=joint ? s.M+(split ? 1 : -1) : s.M
    if split && k==s.M && !joint
        p=makeproposal(s,kind,(i,j),ids,[i,j],Int[],oldlabels,newlabels,Int[],newk,D;newM,beta)
        p.outcome=:capacity; return p
    end
    oo=order===nothing ? shuffle(rng,setdiff(D,[i,j])) : Int.(copy(order))
    p=makeproposal(s,kind,(i,j),ids,[i,j],Int[],oldlabels,newlabels,oo,newk,D;newM,beta)
    finish_proposal!(w,s,p,D,rng;local_move=true)
end

"""
Propose with kind=:reallocate (r-to-r), :global (all-data k-to-k±1 at fixed M),
:global_joint (all-data k-to-k±1 and M-to-M±1 together), :local (one cluster
splits or two merge at fixed M), or :local_joint (the same local partition move
and M-to-M±1 together). Local proposals leave every other cluster's members
unchanged and use all frozen references regardless of their global block count.
Gamma and Lambda remain fixed. Existing unbounded local hyperparameter updates
compose outside this kernel. Reference banks must remain frozen. The optional
beta in [0,1] multiplies only the integrated NIW log likelihood; allocation,
component-count and hyperparameter priors retain their full strength.
"""
function propose!(w::Workspace,s::MF.State,rng;kind=:reallocate,beta=1.,kwargs...)
    beta=checkbeta(beta)
    kind==:reallocate && return propose_reallocation!(w,s,rng;beta,kwargs...)
    kind==:global && return propose_global!(w,s,rng;beta,kwargs...)
    kind==:global_joint && return propose_global!(w,s,rng;joint=true,beta,kwargs...)
    kind==:local && return propose_local!(w,s,rng;beta,kwargs...)
    kind==:local_joint && return propose_local!(w,s,rng;joint=true,beta,kwargs...)
    throw(ArgumentError("MFM block kind must be :reallocate, :global, :global_joint, :local or :local_joint"))
end

"""Atomic cache commit. Unused G.Cluster.x fields are not MFM parameters."""
function commit!(s::MF.State,p::Proposal)
    p.outcome==:accepted || return s
    s.M==p.M && s.gamma==p.gamma && s.lambda==p.lambda || error("hyperparameters changed between proposal and commit")
    1<=p.newk<=p.newM || error("MFM proposal exceeds component capacity")
    count_move=p.pair!==nothing
    if count_move
        unaffected=[findall(==(h),s.c) for h in s.active if !(h in p.oldids)]
        groups=vcat(unaffected,p.newgroups)
        cl=[cluster(s,g) for g in groups]; labels=zeros(Int,length(s.c)); active=collect(eachindex(cl))
        for h in eachindex(groups); labels[groups[h]].=h; end
    else
        cl=copy(s.clusters); labels=copy(s.c); active=copy(s.active)
        for h in eachindex(p.oldids)
            id=p.oldids[h]; cl[id]=cluster(s,p.newgroups[h]); labels[p.newgroups[h]].=id
        end
    end
    all(>(0),labels) && sum(c.n for c in cl)==length(labels) || error("MFM block proposal does not partition the data")
    cached=cache_state(s,cl,labels,active)
    # All Cholesky factorizations and predictive refreshes precede mutation.
    s.clusters=cached.clusters; s.c=cached.c; s.active=cached.active
    s.predL=cached.predL; s.predmu=cached.predmu; s.prednorm=cached.prednorm
    s.M=p.newM
    s
end

function attempt!(w::Workspace,s::MF.State,rng;observer=nothing,kwargs...)
    p=propose!(w,s,rng;kwargs...); p===nothing && return nothing
    observer===nothing || observer(s,p)
    commit!(s,p); p
end

"""Independent full canonical-partition target and auxiliary proposal route."""
function raw_ratio(s::MF.State,p::Proposal)
    p.logforward===nothing && throw(ArgumentError("proposal has no endpoint density"))
    unaffected=[findall(==(h),s.c) for h in s.active if !(h in p.oldids)]
    oldtarget=MT.logtarget(s;groups=vcat(unaffected,p.oldgroups),M=p.M)
    newtarget=MT.logtarget(s;groups=vcat(unaffected,p.newgroups),M=p.newM)
    if p.beta!=1.
        oldtarget+=(p.beta-1)*sum(MT.log_marginal(s,g) for g in vcat(unaffected,p.oldgroups))
        newtarget+=(p.beta-1)*sum(MT.log_marginal(s,g) for g in vcat(unaffected,p.newgroups))
    end
    if islocal(p)
        oldother=Vector{Int}[]; newother=Vector{Int}[]
    elseif p.pair===nothing
        oldother=p.oldgroups; newother=p.newgroups
    else
        split=p.newk>p.k
        oldother=split ? p.oldgroups[2:end] : p.oldgroups[3:end]
        newother=split ? p.newgroups[3:end] : p.newgroups[2:end]
    end
    qforward=p.logforward-sum(log(length(g)) for g in oldother;init=0.)
    qreverse=p.logreverse-sum(log(length(g)) for g in newother;init=0.)
    (target_difference=newtarget-oldtarget,log_forward=qforward,log_reverse=qreverse,
        raw=newtarget-oldtarget+qreverse-qforward)
end

end
