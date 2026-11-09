module DPPCollapsedReallocation

using GeneralDPPMixtures,Random,SpecialFunctions,LinearAlgebra
import ..DPPSplitMerge as SM
import ..DPPMultiReallocation as MR
const G=GeneralDPPMixtures
const Proposal=MR.Proposal
export Workspace,Proposal,propose!,attempt!,commit!,raw_ratio,logA_beta,
       location_law_beta,restricted_scan_beta!,proper_beta

"""Current-partition collapsed Gibbs proposal, with fixed occupied count and S.
`blocks=0` selects all clusters; otherwise a uniform min(blocks,k)-subset is used.
No fitted or frozen reference partition is used.
"""
struct Workspace
    blocks::Int
    function Workspace(blocks::Int)
        blocks==0 || blocks>=2 || throw(ArgumentError("blocks must be 0 or at least two"))
        new(blocks)
    end
end
Workspace(m::G.Model;blocks=0)=Workspace(Int(blocks))
geometry_call(f,l)=l===nothing ? f() : Base.lock(f,l)

function checkbeta(beta)
    beta isa Real && isfinite(beta) && 0<=beta<=1 || throw(ArgumentError("beta must be finite and in [0,1]"))
    Float64(beta)
end
"""A fixed model-only gate, valid even for anchor-only scan intermediates."""
function proper_beta(m,beta)
    b=checkbeta(beta)
    b>length(first(m.y))/(m.tau+1) || throw(ArgumentError("powered collapsed reallocation requires beta > d/(tau+1)"))
    b
end

"""Integral over R^d of the fixed-location, covariance-collapsed likelihood^beta."""
function logA_beta(m,c,beta)
    b=checkbeta(beta);b==1 && return SM.logA(m,c)
    n=c.n;d=length(c.mean);df=b*(m.tau+n)-d
    n>0 && df>0 || throw(ArgumentError("powered location integral is improper"))
    L=cholesky(Symmetric(m.omega+c.scatter)).L;ld=2sum(log,diag(L))
    C=-n*d/2*log(pi)+m.tau/2*m.logdetomega+
        G.logmultigamma((m.tau+n)/2,d)-G.logmultigamma(m.tau/2,d)
    b*(C-(m.tau+n)/2*ld)+d/2*log(pi)-d/2*log(n)+ld/2+
        loggamma(df/2)-loggamma(b*(m.tau+n)/2)
end
"""G's factor convention is L*L'=df*scale; only df changes with beta."""
function location_law_beta(m,c,beta)
    b=checkbeta(beta);b==1 && return SM.location_law(m,c)
    df=b*(m.tau+c.n)-length(c.mean)
    c.n>0 && df>0 || throw(ArgumentError("powered location law is improper"))
    mu,L,_=SM.location_law(m,c)
    mu,L,df
end

"""Current-label anchored Gibbs scan for product Gamma(a+n_h)*A_beta(h)."""
function restricted_scan_beta!(m,D,anchors,order,rng,initial;
        forced=nothing,anchorlabels=collect(eachindex(anchors)),beta=1.)
    b=proper_beta(m,beta)
    b==1 && return MR.restricted_scan!(m,D,anchors,order,rng,initial;forced,anchorlabels)
    r=maximum(anchorlabels);n=length(m.y)
    length(anchorlabels)==length(anchors) && sort(unique(anchorlabels))==collect(1:r) || throw(ArgumentError("invalid anchor labels"))
    length(unique(anchors))==length(anchors) && all(in(D),anchors) || throw(ArgumentError("invalid anchors"))
    sort(order)==sort(setdiff(D,anchors)) || throw(ArgumentError("invalid scan order"))
    length(initial)==n && all(i->initial[i] in 1:r,D) || throw(ArgumentError("invalid launch"))
    all(a->initial[anchors[a]]==anchorlabels[a],eachindex(anchors)) || throw(ArgumentError("unoriented launch"))
    if forced!==nothing
        length(forced)==n && all(i->forced[i] in 1:r,D) &&
            all(a->forced[anchors[a]]==anchorlabels[a],eachindex(anchors)) || throw(ArgumentError("invalid forced partition"))
    end
    labels=copy(initial);groups=[SM.cluster(m,[i for i in D if labels[i]==h]) for h in 1:r]
    scores=zeros(r);lp=0.
    for i in order
        G.remove!(groups[labels[i]],m.y[i])
        for h in 1:r
            c=groups[h];count,mu,scatter=c.n,c.mean,c.scatter
            old=logA_beta(m,c,b);G.add!(c,m.y[i]);augmented=logA_beta(m,c,b)
            c.n=count;c.mean=mu;c.scatter=scatter
            scores[h]=log(count+m.a)+augmented-old
        end
        mx=maximum(scores);lognorm=mx+log(sum(exp(x-mx) for x in scores))
        h=forced===nothing ? G.draw_weights(rng,exp.(scores.-mx)) : forced[i]
        lp+=scores[h]-lognorm;labels[i]=h;G.add!(groups[h],m.y[i])
    end
    [[i for i in D if labels[i]==h] for h in 1:r],lp
end

"""
Generate one anchored Gibbs scan from the current oriented allocation. Its
reverse probability starts from the proposed allocation, visits observations
in reverse order, and is forced to restore their old labels. A uniform order
and its reversed order have equal auxiliary probability.

Each coordinate scan is reversible for product_h Gamma(a+n_h)*A_h, conditional
on anchors. Consequently G+H+logreverse-logforward is zero in exact arithmetic.
The MH ratio retains all four computed terms rather than assuming numerical
cancellation. Student center densities cancel the corresponding likelihood
factors; full DPP geometry and the anchor-size correction remain in the ratio.
"""
function cold_propose!(w::Workspace,s::G.MarginalState,rng;
        ids=nothing,anchors=nothing,order=nothing,geometry_lock=nothing)
    m=s.model;k=length(s.active);k<2 && return nothing
    r=w.blocks==0 ? k : min(w.blocks,k)
    selected=ids===nothing ? sort(shuffle(rng,s.active)[1:r]) : Int.(copy(ids))
    length(selected)==r && length(unique(selected))==r && all(in(s.active),selected) ||
        throw(ArgumentError("invalid selected clusters"))
    oldgroups=[findall(==(h),s.c) for h in selected]
    aa=anchors===nothing ? [rand(rng,g) for g in oldgroups] : Int.(copy(anchors))
    length(aa)==r && all(h->aa[h] in oldgroups[h],1:r) || throw(ArgumentError("invalid cluster anchors"))
    D=sort(vcat(oldgroups...))
    oo=order===nothing ? shuffle(rng,setdiff(D,aa)) : Int.(copy(order))
    p=Proposal(:collapsed_reallocate,aa,oo,selected,oldgroups,Vector{Int}[],
        [s.clusters[h].x for h in selected],typeof(first(m.y))[],k,length(s.S),length(D),0,1,
        Float64[],Float64[],nothing,nothing,nothing,nothing,nothing,nothing,nothing,nothing,:constructed)
    oldlabels=zeros(Int,length(s.c))
    for h in 1:r;oldlabels[oldgroups[h]].=h;end
    p.newgroups,p.logforward=MR.restricted_scan!(m,D,aa,oo,rng,oldlabels)
    newlabels=zeros(Int,length(s.c))
    for h in 1:r;newlabels[p.newgroups[h]].=h;end
    _,p.logreverse=MR.restricted_scan!(m,D,aa,reverse(oo),rng,newlabels;forced=oldlabels)
    p.component_logforward=[p.logforward];p.component_logreverse=[p.logreverse]
    oldclusters=[SM.cluster(m,g) for g in oldgroups]
    newclusters=[SM.cluster(m,g) for g in p.newgroups]
    p.Gterm=sum(loggamma(m.a+c.n) for c in newclusters)-sum(loggamma(m.a+c.n) for c in oldclusters)
    p.H=sum(SM.logA(m,c) for c in newclusters)-sum(SM.logA(m,c) for c in oldclusters)
    p.selectionterm=MR.selection_logratio(oldgroups,p.newgroups)
    p.moved_observations=sum(s.c[i]!=selected[h] for h in 1:r for i in p.newgroups[h])
    for c in newclusters;push!(p.newxs,G.student_draw(rng,SM.location_law(m,c)...));end
    if !all(x->G.inside(m,x),p.newxs);p.outcome=:outside_domain;return p;end
    unaffected=[s.clusters[h].x for h in s.active if !(h in selected)]
    p.DK=geometry_call(geometry_lock) do
        SM.geometry_delta(m,s.S,unaffected,p.oldxs,p.newxs)
    end
    if p.DK==-Inf;p.outcome=:zero_determinant;return p;end
    p.ratio=p.Gterm+p.H+p.DK+p.selectionterm+p.logreverse-p.logforward
    isfinite(p.ratio) || error("nonfinite collapsed reallocation ratio")
    p.logu=log(rand(rng));p.outcome=p.logu<min(0.,p.ratio) ? :accepted : :mh_rejected
    p
end

"""Powered extension; beta=1 calls the original arithmetic and RNG path."""
function propose!(w::Workspace,s::G.MarginalState,rng;beta=1.,kwargs...)
    b=proper_beta(s.model,beta)
    b==1 && return cold_propose!(w,s,rng;kwargs...)
    powered_propose!(w,s,rng,b;kwargs...)
end
function powered_propose!(w::Workspace,s::G.MarginalState,rng,beta;
        ids=nothing,anchors=nothing,order=nothing,geometry_lock=nothing)
    m=s.model;k=length(s.active);k<2 && return nothing
    r=w.blocks==0 ? k : min(w.blocks,k)
    selected=ids===nothing ? sort(shuffle(rng,s.active)[1:r]) : Int.(copy(ids))
    length(selected)==r && length(unique(selected))==r && all(in(s.active),selected) || throw(ArgumentError("invalid selected clusters"))
    oldgroups=[findall(==(h),s.c) for h in selected]
    aa=anchors===nothing ? [rand(rng,g) for g in oldgroups] : Int.(copy(anchors))
    length(aa)==r && all(h->aa[h] in oldgroups[h],1:r) || throw(ArgumentError("invalid cluster anchors"))
    D=sort(vcat(oldgroups...));oo=order===nothing ? shuffle(rng,setdiff(D,aa)) : Int.(copy(order))
    p=Proposal(:collapsed_reallocate,aa,oo,selected,oldgroups,Vector{Int}[],
        [s.clusters[h].x for h in selected],typeof(first(m.y))[],k,length(s.S),length(D),0,1,
        Float64[],Float64[],nothing,nothing,nothing,nothing,nothing,nothing,nothing,nothing,:constructed)
    oldlabels=zeros(Int,length(s.c))
    for h in 1:r;oldlabels[oldgroups[h]].=h;end
    p.newgroups,p.logforward=restricted_scan_beta!(m,D,aa,oo,rng,oldlabels;beta)
    newlabels=zeros(Int,length(s.c))
    for h in 1:r;newlabels[p.newgroups[h]].=h;end
    _,p.logreverse=restricted_scan_beta!(m,D,aa,reverse(oo),rng,newlabels;forced=oldlabels,beta)
    p.component_logforward=[p.logforward];p.component_logreverse=[p.logreverse]
    oldclusters=[SM.cluster(m,g) for g in oldgroups];newclusters=[SM.cluster(m,g) for g in p.newgroups]
    p.Gterm=sum(loggamma(m.a+c.n) for c in newclusters)-sum(loggamma(m.a+c.n) for c in oldclusters)
    # H is the integrated powered-likelihood change, not beta times cold H.
    p.H=sum(logA_beta(m,c,beta) for c in newclusters)-sum(logA_beta(m,c,beta) for c in oldclusters)
    p.selectionterm=MR.selection_logratio(oldgroups,p.newgroups)
    p.moved_observations=sum(s.c[i]!=selected[h] for h in 1:r for i in p.newgroups[h])
    for c in newclusters;push!(p.newxs,G.student_draw(rng,location_law_beta(m,c,beta)...));end
    if !all(x->G.inside(m,x),p.newxs);p.outcome=:outside_domain;return p;end
    unaffected=[s.clusters[h].x for h in s.active if !(h in selected)]
    p.DK=geometry_call(geometry_lock) do
        SM.geometry_delta(m,s.S,unaffected,p.oldxs,p.newxs)
    end
    if p.DK==-Inf;p.outcome=:zero_determinant;return p;end
    p.ratio=p.Gterm+p.H+p.DK+p.selectionterm+p.logreverse-p.logforward
    isfinite(p.ratio) || error("nonfinite powered collapsed reallocation ratio")
    p.logu=log(rand(rng));p.outcome=p.logu<min(0.,p.ratio) ? :accepted : :mh_rejected
    p
end

function commit!(s::G.MarginalState,p::Proposal;geometry_lock=nothing)
    p.kind==:collapsed_reallocate || throw(ArgumentError("not a collapsed reallocation proposal"))
    MR.commit!(s,p;geometry_lock)
end
function attempt!(w::Workspace,s::G.MarginalState,rng;observer=nothing,geometry_lock=nothing,kwargs...)
    p=propose!(w,s,rng;geometry_lock,kwargs...)
    p===nothing && return nothing
    observer===nothing || observer(s,p)
    commit!(s,p;geometry_lock)
    p
end
function raw_ratio(m,S,unaffected,oldgroups,oldxs,newgroups,newxs,logforward,logreverse;beta=1.)
    b=proper_beta(m,beta)
    b==1 && return MR.raw_ratio(m,S,unaffected,oldgroups,oldxs,newgroups,newxs,logforward,logreverse)
    function target(groups,xs)
        allxs=vcat(unaffected,xs);all(x->G.inside(m,x),allxs) || return -Inf
        ld=SM.physical_logdet(m,S,allxs)
        ld+sum(loggamma(m.a+length(g))-loggamma(m.a)+b*G.log_marginal(SM.cluster(m,g,x),m) for (g,x) in zip(groups,xs))
    end
    loglocations(groups,xs)=sum(G.student_logdensity(x,location_law_beta(m,SM.cluster(m,g),b)...) for (g,x) in zip(groups,xs))
    delta=target(newgroups,newxs)-target(oldgroups,oldxs)
    qforward=logforward+loglocations(newgroups,newxs)-sum(log(length(g)) for g in oldgroups)
    qreverse=logreverse+loglocations(oldgroups,oldxs)-sum(log(length(g)) for g in newgroups)
    (target_difference=delta,log_forward=qforward,log_reverse=qreverse,raw=delta+qreverse-qforward)
end

end
