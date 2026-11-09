module DPPSequentialSplitMerge

using GeneralDPPMixtures, Random, SpecialFunctions
import ..DPPSplitMerge as SM
import ..DPPCollapsedReallocation as CR
const G=GeneralDPPMixtures
const Proposal=SM.Proposal

export Workspace, Proposal, sequential_split!, propose!, attempt!, commit!, raw_ratio

"""Stateless split/merge workspace; no reference partition is used."""
struct Workspace end
Workspace(::G.Model)=Workspace()
geometry_call(f,l)=l===nothing ? f() : Base.lock(f,l)

"""
Start with the two anchors alone and allocate each remaining union member once.
The score is log(n_h+a)+log A(C_h∪{i})−log A(C_h), where A integrates the
cluster center over R^d and its covariance under the unchanged DPP model.
`forced` evaluates an oriented endpoint without advancing the external RNG.
The returned probability conditions on the ordered anchors and permutation.
"""
function cold_sequential_split!(m::G.Model,D,anchors,order,rng;forced=nothing)
    n=length(m.y)
    length(D)==length(unique(D)) && all(t->1<=t<=n,D) || throw(ArgumentError("invalid split union"))
    length(anchors)==2 && anchors[1]!=anchors[2] && all(in(D),anchors) || throw(ArgumentError("invalid split anchors"))
    sort(order)==sort(setdiff(D,anchors)) || throw(ArgumentError("order must permute every nonanchor once"))
    if forced!==nothing
        length(forced)==n && all(t->forced[t] in (1,2),D) && forced[anchors[1]]==1 && forced[anchors[2]]==2 ||
            throw(ArgumentError("forced split labels must orient the anchors"))
    end
    groups=[[Int(anchors[1])],[Int(anchors[2])]]
    clusters=[SM.cluster(m,g) for g in groups]
    scores=zeros(2);logq=0.
    for t in order
        for h in 1:2
            c=clusters[h];before=SM.logA(m,c)
            oldn,oldmean,oldscatter=c.n,c.mean,c.scatter
            G.add!(c,m.y[t]);after=SM.logA(m,c)
            c.n=oldn;c.mean=oldmean;c.scatter=oldscatter
            scores[h]=log(oldn+m.a)+after-before
        end
        top=max(scores...);normalizer=top+log(exp(scores[1]-top)+exp(scores[2]-top))
        h=forced===nothing ? (rand(rng)<exp(scores[1]-normalizer) ? 1 : 2) : forced[t]
        logq+=scores[h]-normalizer
        push!(groups[h],t);G.add!(clusters[h],m.y[t])
    end
    sort!.(groups),logq
end

"""
Optional likelihood-powered partial split law. The model-only proper-beta gate
also covers the singleton anchor intermediates. At beta=1 the original cold
arithmetic and RNG path are called directly. A_beta is the integral of the
fixed-location covariance-collapsed likelihood raised to beta, not A^beta.
"""
function sequential_split!(m::G.Model,D,anchors,order,rng;forced=nothing,beta=1.)
    b=CR.proper_beta(m,beta)
    b==1 && return cold_sequential_split!(m,D,anchors,order,rng;forced)
    n=length(m.y)
    length(D)==length(unique(D)) && all(t->1<=t<=n,D) || throw(ArgumentError("invalid split union"))
    length(anchors)==2 && anchors[1]!=anchors[2] && all(in(D),anchors) || throw(ArgumentError("invalid split anchors"))
    sort(order)==sort(setdiff(D,anchors)) || throw(ArgumentError("order must permute every nonanchor once"))
    if forced!==nothing
        length(forced)==n && all(t->forced[t] in (1,2),D) && forced[anchors[1]]==1 && forced[anchors[2]]==2 ||
            throw(ArgumentError("forced split labels must orient the anchors"))
    end
    groups=[[Int(anchors[1])],[Int(anchors[2])]]
    clusters=[SM.cluster(m,g) for g in groups]
    scores=zeros(2);logq=0.
    for t in order
        for h in 1:2
            c=clusters[h];before=CR.logA_beta(m,c,b)
            oldn,oldmean,oldscatter=c.n,c.mean,c.scatter
            G.add!(c,m.y[t]);after=CR.logA_beta(m,c,b)
            c.n=oldn;c.mean=oldmean;c.scatter=oldscatter
            scores[h]=log(oldn+m.a)+after-before
        end
        top=max(scores...);normalizer=top+log(exp(scores[1]-top)+exp(scores[2]-top))
        h=forced===nothing ? (rand(rng)<exp(scores[1]-normalizer) ? 1 : 2) : forced[t]
        logq+=scores[h]-normalizer
        push!(groups[h],t);G.add!(clusters[h],m.y[t])
    end
    sort!.(groups),logq
end

"""
Cold fixed-spectrum split/merge with a uniform ordered observation pair. A
split grows two anchor-seeded partial groups; a merge has one deterministic
allocation and replays that same sequential split probability in reverse.
The union and ordered anchors are identical in reverse. Uniform scan orders
are paired with the same order, so neither an anchor-size nor a factor-two
correction occurs. `anchors` and `order` are proposal-only diagnostic overrides.

Centers are independent, untruncated SM.location_law draws. A draw outside the
physical box is a single rejected proposal. The fixed natural rank capacity
K=length(S) is a self-loop; the spectrum and original sampler streams are never
drawn from by this proposal. Only the external RNG advances.
"""
function cold_propose!(::Workspace,s::G.MarginalState,rng;anchors=nothing,order=nothing,geometry_lock=nothing)
    m=s.model;n=length(s.c);n<2 && return nothing
    k=length(s.active);M=length(s.S);k<=M || error("occupied DPP count exceeds the fixed spectral rank")
    i,j=if anchors===nothing
        a=rand(rng,1:n);b=rand(rng,1:n-1);(a,b>=a ? b+1 : b)
    else
        anchors
    end
    1<=i<=n && 1<=j<=n && i!=j || throw(ArgumentError("invalid sequential anchors"))
    ids=unique([s.c[i],s.c[j]]);split=length(ids)==1
    old=[findall(==(h),s.c) for h in ids];D=sort(vcat(old...))
    p=Proposal(split ? :split : :merge,(i,j),Int[],ids,old,Vector{Int}[],
        [s.clusters[h].x for h in ids],typeof(first(m.y))[],k,M,
        nothing,nothing,nothing,nothing,nothing,nothing,:constructed)
    if split && k==M;p.outcome=:capacity;return p;end
    p.order=order===nothing ? shuffle(rng,setdiff(D,[i,j])) : Int.(copy(order))
    if split
        daughters,p.logs=cold_sequential_split!(m,D,[i,j],p.order,rng)
        p.newgroups=daughters
    else
        forced=zeros(Int,n);forced[old[1]].=1;forced[old[2]].=2
        daughters,p.logs=cold_sequential_split!(m,D,[i,j],p.order,rng;forced)
        p.newgroups=[D]
    end
    c1=SM.cluster(m,daughters[1]);c2=SM.cluster(m,daughters[2]);parent=SM.cluster(m,D)
    p.Gterm=loggamma(m.a+c1.n)+loggamma(m.a+c2.n)-loggamma(m.a+parent.n)-loggamma(m.a)
    p.H=SM.logA(m,c1)+SM.logA(m,c2)-SM.logA(m,parent)
    for c in (split ? (c1,c2) : (parent,))
        push!(p.newxs,G.student_draw(rng,SM.location_law(m,c)...))
    end
    if !all(x->G.inside(m,x),p.newxs);p.outcome=:outside_domain;return p;end
    unaffected=[s.clusters[h].x for h in s.active if !(h in ids)]
    p.DK=geometry_call(geometry_lock) do
        SM.geometry_delta(m,s.S,unaffected,p.oldxs,p.newxs)
    end
    if p.DK==-Inf;p.outcome=:zero_determinant;return p;end
    p.ratio=split ? p.Gterm+p.H+p.DK-p.logs : -p.Gterm-p.H+p.DK+p.logs
    isfinite(p.ratio) || error("nonfinite sequential split/merge acceptance ratio")
    p.logu=log(rand(rng));p.outcome=p.logu<min(0.,p.ratio) ? :accepted : :mh_rejected
    p
end

"""
Propose for the fixed-spectrum target prior*likelihood^beta. beta=1 delegates
the unchanged cold implementation. For proper hot beta, the sequential scores,
center proposal laws, and acceptance ratio all use A_beta. The returned ratio
already targets beta: a caller must not add another (beta-1)*Delta loglikelihood.
The auxiliary pair and scan-order probabilities cancel exactly as in the cold
law; no anchor-size or factor-two correction is introduced.
"""
function propose!(w::Workspace,s::G.MarginalState,rng;beta=1.,kwargs...)
    b=CR.proper_beta(s.model,beta)
    b==1 && return cold_propose!(w,s,rng;kwargs...)
    powered_propose!(w,s,rng,b;kwargs...)
end
function powered_propose!(::Workspace,s::G.MarginalState,rng,beta;
        anchors=nothing,order=nothing,geometry_lock=nothing)
    m=s.model;n=length(s.c);n<2 && return nothing
    k=length(s.active);M=length(s.S);k<=M || error("occupied DPP count exceeds the fixed spectral rank")
    i,j=if anchors===nothing
        a=rand(rng,1:n);b=rand(rng,1:n-1);(a,b>=a ? b+1 : b)
    else
        anchors
    end
    1<=i<=n && 1<=j<=n && i!=j || throw(ArgumentError("invalid sequential anchors"))
    ids=unique([s.c[i],s.c[j]]);split=length(ids)==1
    old=[findall(==(h),s.c) for h in ids];D=sort(vcat(old...))
    p=Proposal(split ? :split : :merge,(i,j),Int[],ids,old,Vector{Int}[],
        [s.clusters[h].x for h in ids],typeof(first(m.y))[],k,M,
        nothing,nothing,nothing,nothing,nothing,nothing,:constructed)
    if split && k==M;p.outcome=:capacity;return p;end
    p.order=order===nothing ? shuffle(rng,setdiff(D,[i,j])) : Int.(copy(order))
    if split
        daughters,p.logs=sequential_split!(m,D,[i,j],p.order,rng;beta)
        p.newgroups=daughters
    else
        forced=zeros(Int,n);forced[old[1]].=1;forced[old[2]].=2
        daughters,p.logs=sequential_split!(m,D,[i,j],p.order,rng;forced,beta)
        p.newgroups=[D]
    end
    c1=SM.cluster(m,daughters[1]);c2=SM.cluster(m,daughters[2]);parent=SM.cluster(m,D)
    p.Gterm=loggamma(m.a+c1.n)+loggamma(m.a+c2.n)-loggamma(m.a+parent.n)-loggamma(m.a)
    p.H=CR.logA_beta(m,c1,beta)+CR.logA_beta(m,c2,beta)-CR.logA_beta(m,parent,beta)
    for c in (split ? (c1,c2) : (parent,))
        push!(p.newxs,G.student_draw(rng,CR.location_law_beta(m,c,beta)...))
    end
    if !all(x->G.inside(m,x),p.newxs);p.outcome=:outside_domain;return p;end
    unaffected=[s.clusters[h].x for h in s.active if !(h in ids)]
    p.DK=geometry_call(geometry_lock) do
        SM.geometry_delta(m,s.S,unaffected,p.oldxs,p.newxs)
    end
    if p.DK==-Inf;p.outcome=:zero_determinant;return p;end
    p.ratio=split ? p.Gterm+p.H+p.DK-p.logs : -p.Gterm-p.H+p.DK+p.logs
    isfinite(p.ratio) || error("nonfinite powered sequential split/merge acceptance ratio")
    p.logu=log(rand(rng));p.outcome=p.logu<min(0.,p.ratio) ? :accepted : :mh_rejected
    p
end

function commit!(s::G.MarginalState,p::Proposal;geometry_lock=nothing)
    geometry_call(geometry_lock) do
        SM.commit!(s,p)
    end
end
function raw_ratio(s::G.MarginalState,p::Proposal;beta=1.)
    unaffected=[s.clusters[h].x for h in s.active if !(h in p.oldids)]
    raw_ratio(s.model,s.S,unaffected,p.oldgroups,p.oldxs,p.newgroups,p.newxs,p.logs,p.kind;beta)
end
function raw_ratio(m,S,unaffected,oldgroups,oldxs,newgroups,newxs,logs,kind;beta=1.)
    b=CR.proper_beta(m,beta)
    b==1 && return SM.raw_ratio(m,S,unaffected,oldgroups,oldxs,newgroups,newxs,logs,kind)
    kind in (:split,:merge) || throw(ArgumentError("invalid sequential split/merge kind"))
    function target(groups,xs)
        allxs=vcat(unaffected,xs);all(x->G.inside(m,x),allxs) || return -Inf
        SM.physical_logdet(m,S,allxs)+sum(loggamma(m.a+length(g))-loggamma(m.a)+
            b*G.log_marginal(SM.cluster(m,g,x),m) for (g,x) in zip(groups,xs))
    end
    loglocations(groups,xs)=sum(G.student_logdensity(x,CR.location_law_beta(m,SM.cluster(m,g),b)...)
        for (g,x) in zip(groups,xs))
    delta=target(newgroups,newxs)-target(oldgroups,oldxs)
    qforward=loglocations(newgroups,newxs)+(kind==:split ? logs : 0.)
    qreverse=loglocations(oldgroups,oldxs)+(kind==:merge ? logs : 0.)
    (target_difference=delta,log_forward=qforward,log_reverse=qreverse,raw=delta+qreverse-qforward)
end
function attempt!(w::Workspace,s::G.MarginalState,rng;observer=nothing,geometry_lock=nothing,kwargs...)
    p=propose!(w,s,rng;geometry_lock,kwargs...);p===nothing && return nothing
    observer===nothing || observer(s,p)
    commit!(s,p;geometry_lock);p
end

end
