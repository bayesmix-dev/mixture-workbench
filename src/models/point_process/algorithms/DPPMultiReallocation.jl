module DPPMultiReallocation

using GeneralDPPMixtures, LinearAlgebra, Random, SpecialFunctions
import ..DPPSplitMerge as SM
const G = GeneralDPPMixtures

export Workspace, Proposal, propose!, attempt!, commit!, raw_ratio,
       launches, restricted_scan!, mixture_logprob, selection_logratio

"""
Frozen proposal information for an exact fixed-spectrum r-to-r move.

`blocks=0` redistributes all occupied clusters; otherwise a uniform subset of
`min(blocks,k)` clusters is used. References are copied on construction and must
remain frozen throughout the retained run. They affect proposals, never targets.
Each reference supplies one launch component in addition to anchored r-means.
All proposal distances use the nonzero empirical-covariance eigenspace of the
fixed observations. No regularization is applied to any target determinant.
"""
struct Workspace
    references::Vector{Vector{Int}}
    whitened::Matrix{Float64}
    blocks::Int
    conflict_policy::Symbol
    function Workspace(references::Vector{Vector{Int}},whitened::Matrix{Float64},blocks::Int,conflict_policy::Symbol)
        conflict_policy in (:fallback,:centroid_assignment) || throw(ArgumentError("invalid reference conflict policy"))
        new(references,whitened,blocks,conflict_policy)
    end
end
# Preserve the existing low-level constructor for MFM and conditional ensembles.
Workspace(references::Vector{Vector{Int}},whitened::Matrix{Float64},blocks::Int)=
    Workspace(references,whitened,blocks,:fallback)
function Workspace(m::G.Model; references=Vector{Vector{Int}}(), blocks=0,conflict_policy=:fallback)
    blocks==0 || blocks>=2 || throw(ArgumentError("blocks must be 0 or at least 2"))
    n=length(m.y); d=length(first(m.y))
    refs=[Int.(copy(ref)) for ref in references]
    all(ref->length(ref)==n && all(>(0),ref),refs) || throw(ArgumentError("invalid reference partition"))
    center=sum(m.y)/n
    Y=reduce(hcat,[y-center for y in m.y])
    E=eigen(Symmetric((Y*Y')/max(1,n-1)))
    largest=max(0.,maximum(E.values)); cutoff=eps(Float64)*largest*d
    scales=[v>cutoff ? inv(sqrt(v)) : 0. for v in E.values]
    Workspace(refs,Diagonal(scales)*E.vectors'*Y,Int(blocks),conflict_policy)
end

mutable struct Proposal{X}
    kind::Symbol
    anchors::Vector{Int}
    order::Vector{Int}
    oldids::Vector{Int}
    oldgroups::Vector{Vector{Int}}
    newgroups::Vector{Vector{Int}}
    oldxs::Vector{X}
    newxs::Vector{X}
    k::Int
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

@inline function distance(W,i,center)
    sum(abs2,W[e,i]-center[e] for e in axes(W,1))
end
function default_launch(w,D,anchors;anchorlabels=collect(eachindex(anchors)))
    r=maximum(anchorlabels); W=w.whitened
    labels=zeros(Int,size(W,2)); centers=zeros(size(W,1),r)
    for h in 1:r
        centers[:,h].=vec(sum(W[:,anchors[findall(==(h),anchorlabels)]];dims=2))/count(==(h),anchorlabels)
    end
    for _ in 1:5
        for i in D
            fixed=findfirst(==(i),anchors)
            labels[i]=fixed===nothing ? argmin([distance(W,i,view(centers,:,h)) for h in 1:r]) : anchorlabels[fixed]
        end
        fill!(centers,0.); counts=zeros(Int,r)
        for i in D
            h=labels[i]; counts[h]+=1
            for e in axes(W,1); centers[e,h]+=W[e,i]; end
        end
        for h in 1:r; centers[:,h]./=counts[h]; end
    end
    labels
end

"""Repair a conflicting reference using fixed-data centroid geometry only.
Reference IDs are canonicalized by first occurrence in sorted D. Sorting the
complete distance/row/daughter tuple gives deterministic tie handling. This is
a launch proposal, not an approximation to either target or endpoint density.
"""
function centroid_launch(w,D,anchors,anchorlabels,ref)
    r=maximum(anchorlabels);length(unique(ref))==r || return nothing
    inds=sort(D);ids=unique(ref[inds]);length(ids)==r || return nothing
    W=w.whitened;row=Dict(id=>j for (j,id) in enumerate(ids))
    centers=zeros(size(W,1),r);counts=zeros(Int,r)
    daughters=zeros(size(W,1),r);anchorcounts=zeros(Int,r)
    for i in inds
        j=row[ref[i]];counts[j]+=1
        for e in axes(W,1);centers[e,j]+=W[e,i];end
    end
    for a in sortperm(anchors)
        h=anchorlabels[a];anchorcounts[h]+=1
        for e in axes(W,1);daughters[e,h]+=W[e,anchors[a]];end
    end
    for h in 1:r
        centers[:,h]./=counts[h];daughters[:,h]./=anchorcounts[h]
    end
    edges=[(sum(abs2,view(centers,:,j)-view(daughters,:,h)),j,h) for j in 1:r for h in 1:r]
    sort!(edges) # total order: distance, canonical reference row, daughter label
    usedrows=falses(r);usedlabels=falses(r);mapping=zeros(Int,r)
    for (_,j,h) in edges
        (usedrows[j] || usedlabels[h]) && continue
        mapping[j]=h;usedrows[j]=true;usedlabels[h]=true
    end
    all(>(0),mapping) || error("incomplete centroid reference assignment")
    labels=zeros(Int,size(W,2))
    for i in inds;labels[i]=mapping[row[ref[i]]];end
    for a in eachindex(anchors);labels[anchors[a]]=anchorlabels[a];end
    labels
end
"""Deterministic launch bank at given anchors and union, independent of state."""
function launches(w::Workspace,m,D,anchors;anchorlabels=collect(eachindex(anchors)))
    length(anchorlabels)==length(anchors) && sort(unique(anchorlabels))==collect(1:maximum(anchorlabels)) || throw(ArgumentError("invalid anchor labels"))
    fallback=default_launch(w,D,anchors;anchorlabels); out=[fallback]
    r=maximum(anchorlabels); W=w.whitened
    for ref in w.references
        # For a global update, references with a different occupied count add
        # only poorly matched components. The count is fixed in both directions.
        w.blocks==0 && length(unique(ref))!=r && continue
        mapping=Dict{Int,Int}();conflict=false
        for a in eachindex(anchors)
            refid=ref[anchors[a]];h=anchorlabels[a]
            haskey(mapping,refid) && mapping[refid]!=h && (conflict=true;break)
            mapping[refid]=h
        end
        if conflict
            repaired=w.conflict_policy==:centroid_assignment ? centroid_launch(w,D,anchors,anchorlabels,ref) : nothing
            push!(out,repaired===nothing ? copy(fallback) : repaired);continue
        end
        labels=zeros(Int,length(m.y))
        for i in D
            labels[i]=get(mapping,ref[i],0)
            if labels[i]==0
                nearest=argmin([sum(abs2,W[e,i]-W[e,anchors[a]] for e in axes(W,1)) for a in eachindex(anchors)])
                labels[i]=anchorlabels[nearest]
            end
        end
        for a in eachindex(anchors); labels[anchors[a]]=anchorlabels[a]; end
        push!(out,labels)
    end
    out
end

"""
One restricted scan from an explicit deterministic launch. Forced evaluation
uses the identical arithmetic and scan order and consumes no random numbers.
Every daughter retains its own anchor, so all integrated-location masses exist.
"""
function restricted_scan!(m,D,anchors,order,rng,initial;forced=nothing,anchorlabels=collect(eachindex(anchors)))
    r=maximum(anchorlabels); n=length(m.y)
    length(anchorlabels)==length(anchors) && sort(unique(anchorlabels))==collect(1:r) || throw(ArgumentError("invalid anchor labels"))
    length(unique(anchors))==length(anchors) && all(in(D),anchors) || throw(ArgumentError("invalid anchors"))
    sort(order)==sort(setdiff(D,anchors)) || throw(ArgumentError("invalid scan order"))
    length(initial)==n && all(i->initial[i] in 1:r,D) || throw(ArgumentError("invalid launch"))
    all(a->initial[anchors[a]]==anchorlabels[a],eachindex(anchors)) || throw(ArgumentError("unoriented launch"))
    if forced!==nothing
        length(forced)==n && all(i->forced[i] in 1:r,D) &&
            all(a->forced[anchors[a]]==anchorlabels[a],eachindex(anchors)) || throw(ArgumentError("invalid forced partition"))
    end
    labels=copy(initial)
    groups=[SM.cluster(m,[i for i in D if labels[i]==h]) for h in 1:r]
    scores=zeros(r); lp=0.
    for i in order
        G.remove!(groups[labels[i]],m.y[i])
        for h in 1:r
            c=groups[h]; count,mu,scatter=c.n,c.mean,c.scatter
            old=SM.logA(m,c)
            G.add!(c,m.y[i]); augmented=SM.logA(m,c)
            c.n=count; c.mean=mu; c.scatter=scatter
            scores[h]=log(count+m.a)+augmented-old
        end
        mx=maximum(scores); lognorm=mx+log(sum(exp(x-mx) for x in scores))
        h=forced===nothing ? G.draw_weights(rng,exp.(scores.-mx)) : forced[i]
        lp+=scores[h]-lognorm
        labels[i]=h; G.add!(groups[h],m.y[i])
    end
    [[i for i in D if labels[i]==h] for h in 1:r],lp
end
function logmeanexp(values)
    mx=maximum(values)
    mx+log(sum(exp(x-mx) for x in values))-log(length(values))
end
function mixture_logprob(m,D,anchors,order,rng,bank,forced;anchorlabels=collect(eachindex(anchors)))
    components=[last(restricted_scan!(m,D,anchors,order,rng,initial;forced,anchorlabels)) for initial in bank]
    logmeanexp(components),components
end
selection_logratio(oldgroups,newgroups)=sum(log(length(g)) for g in oldgroups)-sum(log(length(g)) for g in newgroups)
_geometry(f,l)=l===nothing ? f() : Base.lock(f,l)

"""
Propose an r-to-r redistribution. The cluster subset is uniform, followed by one
uniform observation anchor per selected cluster. Its reverse/forward correction
is prod(n_old)/prod(n_new). Both scan endpoint probabilities marginalize over
EVERY launch component; the realized launch component is never used as qmix.
Centers follow untruncated Student laws, with domain failures self-transitions.
"""
function propose!(w::Workspace,s::G.MarginalState,rng;ids=nothing,anchors=nothing,order=nothing,geometry_lock=nothing)
    m=s.model; k=length(s.active); k<2 && return nothing
    r=w.blocks==0 ? k : min(w.blocks,k)
    selected=ids===nothing ? sort(shuffle(rng,s.active)[1:r]) : Int.(copy(ids))
    length(selected)==r && length(unique(selected))==r && all(in(s.active),selected) || throw(ArgumentError("invalid selected clusters"))
    oldgroups=[findall(==(h),s.c) for h in selected]
    aa=anchors===nothing ? [rand(rng,g) for g in oldgroups] : Int.(copy(anchors))
    length(aa)==r && all(h->aa[h] in oldgroups[h],1:r) || throw(ArgumentError("anchors do not match selected clusters"))
    D=sort(vcat(oldgroups...)); oo=order===nothing ? shuffle(rng,setdiff(D,aa)) : Int.(copy(order))
    p=Proposal(:multi_reallocate,aa,oo,selected,oldgroups,Vector{Int}[],
        [s.clusters[h].x for h in selected],typeof(first(m.y))[],k,length(s.S),length(D),0,0,
        Float64[],Float64[],nothing,nothing,nothing,nothing,nothing,nothing,nothing,nothing,:constructed)
    bank=launches(w,m,D,aa); p.launch_component=rand(rng,eachindex(bank))
    p.newgroups,_=restricted_scan!(m,D,aa,oo,rng,bank[p.launch_component])
    oldlabels=zeros(Int,length(s.c));newlabels=copy(oldlabels)
    for h in 1:r;oldlabels[oldgroups[h]].=h;newlabels[p.newgroups[h]].=h;end
    p.logforward,p.component_logforward=mixture_logprob(m,D,aa,oo,rng,bank,newlabels)
    p.logreverse,p.component_logreverse=mixture_logprob(m,D,aa,oo,rng,bank,oldlabels)
    oldclusters=[SM.cluster(m,g) for g in oldgroups]
    newclusters=[SM.cluster(m,g) for g in p.newgroups]
    p.Gterm=sum(loggamma(m.a+c.n) for c in newclusters)-sum(loggamma(m.a+c.n) for c in oldclusters)
    p.H=sum(SM.logA(m,c) for c in newclusters)-sum(SM.logA(m,c) for c in oldclusters)
    p.selectionterm=selection_logratio(oldgroups,p.newgroups)
    p.moved_observations=sum(s.c[i]!=selected[h] for h in 1:r for i in p.newgroups[h])
    for c in newclusters;push!(p.newxs,G.student_draw(rng,SM.location_law(m,c)...));end
    if !all(x->G.inside(m,x),p.newxs);p.outcome=:outside_domain;return p;end
    unaffected=[s.clusters[h].x for h in s.active if !(h in selected)]
    p.DK=_geometry(geometry_lock) do
        SM.geometry_delta(m,s.S,unaffected,p.oldxs,p.newxs)
    end
    if p.DK==-Inf;p.outcome=:zero_determinant;return p;end
    p.ratio=p.Gterm+p.H+p.DK+p.selectionterm+p.logreverse-p.logforward
    isfinite(p.ratio) || error("nonfinite DPP multi-reallocation ratio")
    p.logu=log(rand(rng));p.outcome=p.logu<min(0.,p.ratio) ? :accepted : :mh_rejected
    p
end

function commit!(s::G.MarginalState,p::Proposal;geometry_lock=nothing)
    p.outcome==:accepted || return s
    r=length(p.oldids)
    r==length(p.newgroups)==length(p.newxs) || error("invalid multi-reallocation proposal")
    m=s.model; cl=copy(s.clusters); labels=copy(s.c)
    for h in 1:r
        id=p.oldids[h];cl[id]=SM.cluster(m,p.newgroups[h],p.newxs[h]);labels[p.newgroups[h]].=id
    end
    F=G.feature_matrix(m,s.S,[cl[h].x for h in s.active]);Q,ld=G.orthobasis(F,G.Counters())
    physical=_geometry(geometry_lock) do
        SM.physical_logdet(m,s.S,[cl[h].x for h in s.active])
    end
    isfinite(ld) && isfinite(physical) && abs(ld-length(s.active)*m.logvolume-physical)<1e-7 || error("accepted multi-reallocation cache inconsistency")
    s.clusters=cl;s.c=labels;s.F=F;s.Q=Q;s.ld=ld;s.counters.projector_rebuilds+=1
    s
end
function attempt!(w::Workspace,s::G.MarginalState,rng;observer=nothing,geometry_lock=nothing,kwargs...)
    p=propose!(w,s,rng;geometry_lock,kwargs...);p===nothing && return nothing
    observer===nothing || observer(s,p)
    commit!(s,p;geometry_lock);p
end

"""Independent unsimplified target/proposal route, including anchor selection."""
function raw_ratio(m,S,unaffected,oldgroups,oldxs,newgroups,newxs,logforward,logreverse)
    function target(groups,xs)
        allxs=vcat(unaffected,xs);all(x->G.inside(m,x),allxs) || return -Inf
        F=G.feature_matrix(m,S,allxs)/sqrt(m.volume)
        ld=2sum(log,diag(cholesky(Symmetric(F'F)).L))
        ld+sum(loggamma(m.a+length(g))-loggamma(m.a)+G.log_marginal(SM.cluster(m,g,x),m) for (g,x) in zip(groups,xs))
    end
    loglocations(groups,xs)=sum(G.student_logdensity(x,SM.location_law(m,SM.cluster(m,g))...) for (g,x) in zip(groups,xs))
    delta=target(newgroups,newxs)-target(oldgroups,oldxs)
    qforward=logforward+loglocations(newgroups,newxs)-sum(log(length(g)) for g in oldgroups)
    qreverse=logreverse+loglocations(oldgroups,oldxs)-sum(log(length(g)) for g in newgroups)
    (target_difference=delta,log_forward=qforward,log_reverse=qreverse,raw=delta+qreverse-qforward)
end

end
