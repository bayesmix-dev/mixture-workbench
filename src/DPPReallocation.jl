module DPPReallocation

using GeneralDPPMixtures, LinearAlgebra, Random, SpecialFunctions, Statistics
import ..DPPSplitMerge as SM
const G = GeneralDPPMixtures

export Workspace, Proposal, propose!, attempt!, commit!, raw_ratio

"""Scratch storage for an exact fixed-spectrum, fixed-k two-cluster move."""
struct Workspace{W,Y}
    scan::W
    launch_data::Y
end
function Workspace(m::G.Model;whiten=false)
    data=if whiten
        length(m.y)>1 || throw(ArgumentError("whitened launch requires at least two observations"))
        center=mean(m.y)
        L=cholesky(Symmetric(cov(reduce(hcat,m.y);dims=2))).L
        [typeof(first(m.y))(L\(y-center)) for y in m.y]
    else
        m.y
    end
    Workspace(SM.Workspace(m),data)
end

# Only geometry can enter SM's shared BigFloat precision fallback. Scans and
# ordinary state-local cache construction can run concurrently across chains.
geometry_call(f,geometry_lock)=geometry_lock===nothing ? f() : lock(f,geometry_lock)

mutable struct Proposal{X}
    kind::Symbol
    anchors::Tuple{Int,Int}
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
    Gterm::Union{Nothing,Float64}
    H::Union{Nothing,Float64}
    DK::Union{Nothing,Float64}
    logforward::Union{Nothing,Float64}
    logreverse::Union{Nothing,Float64}
    ratio::Union{Nothing,Float64}
    logu::Union{Nothing,Float64}
    outcome::Symbol
end

"""
Propose a two-to-two redistribution without touching the state.

The default anchors are a uniform ordered pair of distinct observations. Pairs
in the same cluster give a self-transition: conditioning selection on distinct
clusters would instead require a state-dependent selection correction. Distinct
anchors keep both proposed daughters nonempty and identify their orientation.

The proposal law is exactly one restricted scan from the deterministic launch.
Both endpoint probabilities are evaluated from that same launch and scan order;
they are not probabilities of scans started at the current/proposed partition.
Centers use the untruncated location law, with out-of-domain draws rejected.
`Workspace(m;whiten=true)` uses a fixed data-covariance whitening solely for the
deterministic launch geometry; allocation scores and targets use original data.
"""
function propose!(w::Workspace, s::G.MarginalState, rng;
        anchors=nothing,order=nothing,geometry_lock=nothing)
    m=s.model; n=length(s.c)
    n<2 && return nothing
    i,j=if anchors===nothing
        a=rand(rng,1:n); b=rand(rng,1:n-1)
        (a,b>=a ? b+1 : b)
    else
        anchors
    end
    1<=i<=n && 1<=j<=n && i!=j || throw(ArgumentError("invalid reallocation anchors"))
    ids=unique([s.c[i],s.c[j]])
    oldgroups=[findall(==(h),s.c) for h in ids]
    D=sort(vcat(oldgroups...))
    p=Proposal(:reallocate,(i,j),Int[],ids,oldgroups,Vector{Int}[],
        [s.clusters[h].x for h in ids],typeof(first(m.y))[],length(s.active),length(s.S),
        length(D),0,nothing,nothing,nothing,nothing,nothing,nothing,nothing,:constructed)
    if length(ids)==1
        p.outcome=:same_cluster
        return p
    end
    p.order=order===nothing ? shuffle(rng,setdiff(D,[i,j])) : copy(order)
    p.newgroups,p.logforward=SM.restricted_scan!(w.scan,m,D,i,j,p.order,rng;launch_data=w.launch_data)
    forced=zeros(Int,n)
    for h in 1:2; forced[oldgroups[h]].=h; end
    _,p.logreverse=SM.restricted_scan!(w.scan,m,D,i,j,p.order,rng;forced,launch_data=w.launch_data)
    oldclusters=[SM.cluster(m,g) for g in oldgroups]
    newclusters=[SM.cluster(m,g) for g in p.newgroups]
    p.Gterm=sum(loggamma(m.a+c.n) for c in newclusters)-
        sum(loggamma(m.a+c.n) for c in oldclusters)
    p.H=sum(SM.logA(m,c) for c in newclusters)-sum(SM.logA(m,c) for c in oldclusters)
    p.moved_observations=sum(s.c[t]!=ids[h] for h in 1:2 for t in p.newgroups[h])
    for c in newclusters
        push!(p.newxs,G.student_draw(rng,SM.location_law(m,c)...))
    end
    if !all(x->G.inside(m,x),p.newxs)
        p.outcome=:outside_domain
        return p
    end
    unaffected=[s.clusters[h].x for h in s.active if !(h in ids)]
    p.DK=geometry_call(geometry_lock) do
        SM.geometry_delta(m,s.S,unaffected,p.oldxs,p.newxs)
    end
    if p.DK==-Inf
        p.outcome=:zero_determinant
        return p
    end
    # k and S are fixed: all count and spectral terms cancel. The location
    # proposal densities cancel the corresponding factors of log_marginal.
    p.ratio=p.Gterm+p.H+p.DK+p.logreverse-p.logforward
    isfinite(p.ratio) || error("nonfinite DPP reallocation acceptance ratio")
    p.logu=log(rand(rng))
    p.outcome=p.logu<min(0.,p.ratio) ? :accepted : :mh_rejected
    p
end

"""Commit an accepted proposal after all numerical/cache work succeeds."""
function commit!(s::G.MarginalState,p::Proposal;geometry_lock=nothing)
    p.outcome==:accepted || return s
    length(p.oldids)==length(p.newgroups)==length(p.newxs)==2 || error("invalid reallocation proposal")
    m=s.model; cl=copy(s.clusters); c=copy(s.c)
    for h in 1:2
        id=p.oldids[h]
        cl[id]=SM.cluster(m,p.newgroups[h],p.newxs[h])
        c[p.newgroups[h]].=id
    end
    F=G.feature_matrix(m,s.S,[cl[h].x for h in s.active])
    Q,ld=G.orthobasis(F,G.Counters())
    physical=geometry_call(geometry_lock) do
        SM.physical_logdet(m,s.S,[cl[h].x for h in s.active])
    end
    isfinite(ld) && isfinite(physical) &&
        abs(ld-length(s.active)*m.logvolume-physical)<1e-7 ||
        error("accepted reallocation feature cache inconsistency")
    s.clusters=cl; s.c=c; s.F=F; s.Q=Q; s.ld=ld
    s.counters.projector_rebuilds+=1
    s
end

"""
Attempt one exact redistribution. The observer receives (state, proposal) before
commit; rejected/no-op proposals leave the complete sampler state untouched.
`moved_observations` describes the candidate, including when it is rejected.
Pass the same `geometry_lock` used by other DPP kernels to serialize the rare
shared BigFloat precision fallback. Only geometry evaluations take that lock;
restricted scans and ordinary state-local cache construction remain concurrent.
"""
function attempt!(w::Workspace,s::G.MarginalState,rng;
        observer=nothing,geometry_lock=nothing,kwargs...)
    p=propose!(w,s,rng;geometry_lock,kwargs...)
    p===nothing && return nothing
    observer===nothing || observer(s,p)
    commit!(s,p;geometry_lock)
    p
end

"""Independent unsimplified density route for proposal-ratio auditing."""
function raw_ratio(m,S,unaffected,oldgroups,oldxs,newgroups,newxs,logforward,logreverse)
    function target(groups,xs)
        allxs=vcat(unaffected,xs)
        all(x->G.inside(m,x),allxs) || return -Inf
        F=G.feature_matrix(m,S,allxs)/sqrt(m.volume)
        ld=2sum(log,diag(cholesky(Symmetric(F'F)).L))
        ld+sum(loggamma(m.a+length(g))-loggamma(m.a)+
            G.log_marginal(SM.cluster(m,g,x),m) for (g,x) in zip(groups,xs))
    end
    loglocations(groups,xs)=sum(G.student_logdensity(x,SM.location_law(m,SM.cluster(m,g))...)
        for (g,x) in zip(groups,xs))
    delta=target(newgroups,newxs)-target(oldgroups,oldxs)
    qforward=logforward+loglocations(newgroups,newxs)
    qreverse=logreverse+loglocations(oldgroups,oldxs)
    (target_difference=delta,log_forward=qforward,log_reverse=qreverse,raw=delta+qreverse-qforward)
end

end
