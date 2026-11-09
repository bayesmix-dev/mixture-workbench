module DPPGlobalSplitMerge

using GeneralDPPMixtures, LinearAlgebra, Random, SpecialFunctions
import ..DPPMultiReallocation as MR
import ..DPPSplitMerge as SM
const G=GeneralDPPMixtures

export Workspace, Proposal, propose!, attempt!, commit!, raw_ratio

"""Frozen launch bank for global fixed-spectrum moves changing k by one."""
struct Workspace
    bank::MR.Workspace
end
Workspace(m::G.Model;references=Vector{Vector{Int}}(),conflict_policy=:fallback)=
    Workspace(MR.Workspace(m;references,blocks=0,conflict_policy))

mutable struct Proposal{X}
    kind::Symbol
    pair::Tuple{Int,Int}
    otheranchors::Vector{Int}
    anchors::Vector{Int}
    oldanchorlabels::Vector{Int}
    newanchorlabels::Vector{Int}
    order::Vector{Int}
    oldids::Vector{Int}
    oldgroups::Vector{Vector{Int}}
    newgroups::Vector{Vector{Int}}
    oldxs::Vector{X}
    newxs::Vector{X}
    k::Int
    newk::Int
    M::Int
    union_size::Int
    moved_observations::Int
    launch_component::Int
    component_logforward::Vector{Float64}
    component_logreverse::Vector{Float64}
    Gterm::Union{Nothing,Float64}
    H::Union{Nothing,Float64}
    DK::Union{Nothing,Float64}
    selectionterm::Union{Nothing,Float64}
    logforward::Union{Nothing,Float64}
    logreverse::Union{Nothing,Float64}
    ratio::Union{Nothing,Float64}
    logu::Union{Nothing,Float64}
    outcome::Symbol
end

_geometry(f,l)=l===nothing ? f() : Base.lock(f,l)

function makeproposal(s,pair,ids,otheranchors,oldanchorlabels,newanchorlabels,order,split)
    m=s.model; k=length(s.active)
    Proposal(split ? :global_split : :global_merge,pair,otheranchors,
        vcat(collect(pair),otheranchors),oldanchorlabels,newanchorlabels,order,ids,
        [findall(==(h),s.c) for h in ids],Vector{Int}[],
        [s.clusters[h].x for h in ids],typeof(first(m.y))[],k,k+(split ? 1 : -1),length(s.S),
        length(s.c),0,0,Float64[],Float64[],nothing,nothing,nothing,nothing,nothing,nothing,
        nothing,nothing,:constructed)
end

"""
Global split/merge with exact auxiliary-anchor and mixture-proposal accounting.

A uniform ordered pair of distinct observations selects split (same old cluster)
or merge (different old clusters). One uniform anchor from every OTHER occupied
cluster is selected after randomly ordering those clusters. All anchors stay
fixed during a single full-data scan: the pair is separated for a split and
coincident for a merge. The other anchors identify surviving clusters across
both endpoints, giving the correction prod(n_old_other)/prod(n_new_other).

Forward and reverse endpoint laws use their own appropriate-k frozen launch
banks and explicitly marginalize over every component, including different bank
sizes. Locations are untruncated Student draws and domain failures self-loops.
Optional pair/otheranchors/order arguments support deterministic audits; default
production selection is always the uniform auxiliary law described above.
"""
function propose!(w::Workspace,s::G.MarginalState,rng;
        pair=nothing,otheranchors=nothing,order=nothing,geometry_lock=nothing)
    m=s.model; n=length(s.c); n<2 && return nothing
    i,j=if pair===nothing
        a=rand(rng,1:n); b=rand(rng,1:n-1)
        (a,b>=a ? b+1 : b)
    else
        pair
    end
    1<=i<=n && 1<=j<=n && i!=j || throw(ArgumentError("invalid global split/merge pair"))
    primary=unique([s.c[i],s.c[j]]); split=length(primary)==1
    remaining=setdiff(s.active,primary); k=length(s.active)
    if split && k==length(s.S)
        p=makeproposal(s,(i,j),vcat(primary,remaining),Int[],Int[],Int[],Int[],true)
        p.outcome=:capacity
        return p
    end
    others,aa=if otheranchors===nothing
        ids=shuffle(rng,remaining)
        ids,[rand(rng,findall(==(h),s.c)) for h in ids]
    else
        a=Int.(copy(otheranchors)); ids=[s.c[t] for t in a if 1<=t<=n]
        length(a)==length(remaining) && length(ids)==length(a) &&
            length(unique(ids))==length(ids) && sort(ids)==sort(remaining) ||
            throw(ArgumentError("other anchors must cover exactly the other occupied clusters"))
        ids,a
    end
    ids=vcat(primary,others); anchors=vcat([i,j],aa)
    oldanchorlabels=split ? vcat([1,1],collect(2:k)) : collect(1:k)
    newanchorlabels=split ? collect(1:k+1) : vcat([1,1],collect(2:k-1))
    D=collect(1:n); oo=order===nothing ? shuffle(rng,setdiff(D,anchors)) : Int.(copy(order))
    p=makeproposal(s,(i,j),ids,aa,oldanchorlabels,newanchorlabels,oo,split)
    forwardbank=MR.launches(w.bank,m,D,anchors;anchorlabels=newanchorlabels)
    reversebank=MR.launches(w.bank,m,D,anchors;anchorlabels=oldanchorlabels)
    p.launch_component=rand(rng,eachindex(forwardbank))
    p.newgroups,_=MR.restricted_scan!(m,D,anchors,oo,rng,forwardbank[p.launch_component];anchorlabels=newanchorlabels)
    oldlabels=zeros(Int,n); newlabels=zeros(Int,n)
    for h in eachindex(p.oldgroups); oldlabels[p.oldgroups[h]].=h; end
    for h in eachindex(p.newgroups); newlabels[p.newgroups[h]].=h; end
    p.logforward,p.component_logforward=MR.mixture_logprob(m,D,anchors,oo,rng,forwardbank,newlabels;anchorlabels=newanchorlabels)
    p.logreverse,p.component_logreverse=MR.mixture_logprob(m,D,anchors,oo,rng,reversebank,oldlabels;anchorlabels=oldanchorlabels)
    oldclusters=[SM.cluster(m,g) for g in p.oldgroups]
    newclusters=[SM.cluster(m,g) for g in p.newgroups]
    # The number of clusters changes, so -logGamma(a) must be included per group.
    p.Gterm=sum(loggamma(m.a+c.n)-loggamma(m.a) for c in newclusters)-
        sum(loggamma(m.a+c.n)-loggamma(m.a) for c in oldclusters)
    p.H=sum(SM.logA(m,c) for c in newclusters)-sum(SM.logA(m,c) for c in oldclusters)
    oldother=split ? p.oldgroups[2:end] : p.oldgroups[3:end]
    newother=split ? p.newgroups[3:end] : p.newgroups[2:end]
    p.selectionterm=sum(log(length(g)) for g in oldother;init=0.)-
        sum(log(length(g)) for g in newother;init=0.)
    # Match surviving daughters by their anchors, rather than counting the
    # harmless contiguous storage-ID renumbering performed at commit.
    matchedids=split ? vcat([ids[1],0],ids[2:end]) : vcat(ids[1:1],ids[3:end])
    p.moved_observations=sum(s.c[t]!=matchedids[h] for h in eachindex(p.newgroups) for t in p.newgroups[h])
    for c in newclusters; push!(p.newxs,G.student_draw(rng,SM.location_law(m,c)...)); end
    if !all(x->G.inside(m,x),p.newxs); p.outcome=:outside_domain; return p; end
    p.DK=_geometry(geometry_lock) do
        SM.geometry_delta(m,s.S,typeof(first(m.y))[],p.oldxs,p.newxs)
    end
    if p.DK==-Inf; p.outcome=:zero_determinant; return p; end
    # M stays fixed, so Gamma(aM)/Gamma(n+aM) and spectral terms cancel.
    p.ratio=p.Gterm+p.H+p.DK+p.selectionterm+p.logreverse-p.logforward
    isfinite(p.ratio) || error("nonfinite global DPP split/merge acceptance ratio")
    p.logu=log(rand(rng)); p.outcome=p.logu<min(0.,p.ratio) ? :accepted : :mh_rejected
    p
end

"""Commit all clusters atomically after numerical validation; S remains fixed."""
function commit!(s::G.MarginalState,p::Proposal;geometry_lock=nothing)
    p.outcome==:accepted || return s
    m=s.model; length(p.newgroups)==length(p.newxs)==p.newk || error("invalid global split/merge proposal")
    cl=[SM.cluster(m,g,x) for (g,x) in zip(p.newgroups,p.newxs)]
    labels=zeros(Int,length(s.c))
    for h in eachindex(p.newgroups); labels[p.newgroups[h]].=h; end
    all(>(0),labels) && sum(c.n for c in cl)==length(labels) || error("global proposal does not partition the data")
    active=collect(eachindex(cl)); F=G.feature_matrix(m,s.S,p.newxs)
    Q,ld=G.orthobasis(F,G.Counters())
    physical=_geometry(geometry_lock) do; SM.physical_logdet(m,s.S,p.newxs); end
    isfinite(ld) && isfinite(physical) && abs(ld-length(active)*m.logvolume-physical)<1e-7 ||
        error("accepted global split/merge cache inconsistency")
    G.grow_streams!(s.streams,length(cl))
    s.clusters=cl; s.c=labels; s.active=active; s.F=F; s.Q=Q; s.ld=ld
    s.counters.projector_rebuilds+=1
    s
end

function attempt!(w::Workspace,s::G.MarginalState,rng;observer=nothing,geometry_lock=nothing,kwargs...)
    p=propose!(w,s,rng;geometry_lock,kwargs...); p===nothing && return nothing
    observer===nothing || observer(s,p)
    commit!(s,p;geometry_lock); p
end

"""Unsimplified full target/proposal density route, including auxiliary anchors."""
function raw_ratio(m,S,oldgroups,oldxs,newgroups,newxs,logforward,logreverse)
    function target(groups,xs)
        all(x->G.inside(m,x),xs) || return -Inf
        F=G.feature_matrix(m,S,xs)/sqrt(m.volume)
        ld=2sum(log,diag(cholesky(Symmetric(F'F)).L))
        ld+sum(loggamma(m.a+length(g))-loggamma(m.a)+
            G.log_marginal(SM.cluster(m,g,x),m) for (g,x) in zip(groups,xs))
    end
    loglocations(groups,xs)=sum(G.student_logdensity(x,SM.location_law(m,SM.cluster(m,g))...) for (g,x) in zip(groups,xs))
    split=length(newgroups)>length(oldgroups)
    oldother=split ? oldgroups[2:end] : oldgroups[3:end]
    newother=split ? newgroups[3:end] : newgroups[2:end]
    qforward=logforward+loglocations(newgroups,newxs)-sum(log(length(g)) for g in oldother;init=0.)
    qreverse=logreverse+loglocations(oldgroups,oldxs)-sum(log(length(g)) for g in newother;init=0.)
    delta=target(newgroups,newxs)-target(oldgroups,oldxs)
    (target_difference=delta,log_forward=qforward,log_reverse=qreverse,raw=delta+qreverse-qforward)
end

end
