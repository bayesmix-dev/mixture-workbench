module MFMSequentialSplitMerge

using GeneralDPPMixtures, Random, SpecialFunctions
import ..MFMBaseline as MF
import ..MFMTarget as MT
import ..MFMBlockMoves as MB
const G=GeneralDPPMixtures
const Proposal=MB.Proposal

export Workspace, Proposal, sequential_split!, propose!, attempt!, commit!, raw_ratio

"""Stateless workspace: this proposal uses no reference bank or shared scratch."""
struct Workspace end
Workspace(::G.Model)=Workspace()

"""
Anchor-seeded sequential allocation into two nonempty daughters. Only the two
anchors are present initially; each remaining observation is added once, in
`order`. `forced` replays an endpoint without drawing from `rng`. The returned
log probability conditions on the ordered anchors and permutation.
The optional `gamma` changes only the allocation score, leaving the source
state and its gamma-independent NIW predictive caches unchanged.
"""
function sequential_split!(s::MF.State,D,anchors,order,rng;forced=nothing,beta=1.,gamma=s.gamma)
    beta=MB.checkbeta(beta); n=length(s.c)
    gamma isa Real && isfinite(gamma) && gamma>0 || throw(ArgumentError("gamma must be finite and positive"))
    length(D)==length(unique(D)) && all(t->1<=t<=n,D) || throw(ArgumentError("invalid split union"))
    length(anchors)==2 && anchors[1]!=anchors[2] && all(in(D),anchors) || throw(ArgumentError("invalid split anchors"))
    sort(order)==sort(setdiff(D,anchors)) || throw(ArgumentError("order must permute every nonanchor once"))
    if forced!==nothing
        length(forced)==n && all(t->forced[t] in (1,2),D) && forced[anchors[1]]==1 && forced[anchors[2]]==2 ||
            throw(ArgumentError("forced split labels must orient the anchors"))
    end
    groups=[[Int(anchors[1])],[Int(anchors[2])]]
    clusters=[MB.cluster(s,g) for g in groups]
    cached=MB.cache_state(s,clusters,zeros(Int,n),[1,2])
    scores=zeros(2); weights=zeros(2); logq=0.
    for t in order
        for h in 1:2
            scores[h]=log(clusters[h].n+gamma)+beta*MF.logpred(cached,h,s.model.y[t])
        end
        top=maximum(scores)
        for h in 1:2; weights[h]=exp(scores[h]-top); end
        lognormalizer=top+log(sum(weights))
        h=forced===nothing ? G.draw_weights(rng,weights) : forced[t]
        logq+=scores[h]-lognormalizer
        push!(groups[h],t); G.add!(clusters[h],s.model.y[t]); MF.refresh!(cached,h)
    end
    sort!.(groups),logq
end

"""Count-only target change, including the full-data joint-M normalization."""
function count_change(s,newk,newM)
    n=length(s.c); g=s.gamma; M=s.M; k=length(s.active)
    MT.logfallingfactorial(newM,newk)-MT.logfallingfactorial(M,k)+
        loggamma(newM*g)-loggamma(newM*g+n)-loggamma(M*g)+loggamma(M*g+n)+
        (newM-M)*log(s.lambda)-loggamma(newM)+loggamma(M)
end

"""
Local split/merge with uniform ordered observation anchors. A split grows two
partial daughters from their anchors; a merge is deterministic and evaluates
that same sequential split in reverse. The remaining-observation permutation
is uniform and is its own reverse auxiliary (the same order is replayed).
`joint=true` also changes M by ±1. Only NIW evidence is multiplied by `beta`.
Explicit `pair`/`order` are diagnostic overrides of the uniform production law.
"""
function propose!(::Workspace,s::MF.State,rng;pair=nothing,order=nothing,joint=false,beta=1.)
    beta=MB.checkbeta(beta); n=length(s.c); n<2 && return nothing
    k=length(s.active); k<=s.M || error("occupied MFM count exceeds M")
    i,j=if pair===nothing
        a=rand(rng,1:n); b=rand(rng,1:n-1); (a,b>=a ? b+1 : b)
    else
        pair
    end
    1<=i<=n && 1<=j<=n && i!=j || throw(ArgumentError("invalid sequential pair"))
    ids=unique([s.c[i],s.c[j]]); split=length(ids)==1
    D=sort([t for t in eachindex(s.c) if s.c[t] in ids])
    kind=joint ? (split ? :local_joint_split : :local_joint_merge) : (split ? :local_split : :local_merge)
    oldlabels=split ? [1,1] : [1,2]; newlabels=split ? [1,2] : [1,1]
    newk=k+(split ? 1 : -1); newM=joint ? s.M+(split ? 1 : -1) : s.M
    if split && k==s.M && !joint
        p=MB.makeproposal(s,kind,(i,j),ids,[i,j],Int[],oldlabels,newlabels,Int[],newk,D;newM,beta)
        p.outcome=:capacity; return p
    end
    oo=order===nothing ? shuffle(rng,setdiff(D,[i,j])) : Int.(copy(order))
    p=MB.makeproposal(s,kind,(i,j),ids,[i,j],Int[],oldlabels,newlabels,oo,newk,D;newM,beta)
    if split
        p.newgroups,q=sequential_split!(s,D,[i,j],oo,rng;beta)
        p.logforward=q; p.logreverse=0.
    else
        forced=zeros(Int,n)
        for h in eachindex(p.oldgroups); forced[p.oldgroups[h]].=h; end
        _,q=sequential_split!(s,D,[i,j],oo,rng;forced,beta)
        p.newgroups=[D]; p.logforward=0.; p.logreverse=q
    end
    oldclusters=[MB.cluster(s,g) for g in p.oldgroups]
    newclusters=[MB.cluster(s,g) for g in p.newgroups]
    p.Gterm=sum(loggamma(c.n+s.gamma)-loggamma(s.gamma) for c in newclusters)-
        sum(loggamma(c.n+s.gamma)-loggamma(s.gamma) for c in oldclusters)
    p.H=sum(MT.log_marginal(s,c) for c in newclusters)-sum(MT.log_marginal(s,c) for c in oldclusters)
    p.countterm=joint ? count_change(s,newk,newM) : (split ? log(s.M-k) : -log(s.M-k+1))
    p.selectionterm=0.; p.launch_component=1
    p.component_logforward=[p.logforward]; p.component_logreverse=[p.logreverse]
    mapped=split ? [ids[1],0] : [ids[1]]
    p.moved_observations=sum(s.c[t]!=mapped[h] for h in eachindex(p.newgroups) for t in p.newgroups[h])
    p.ratio=p.Gterm+beta*p.H+p.countterm+p.logreverse-p.logforward
    p.logu=log(rand(rng)); p.outcome=p.logu<min(0.,p.ratio) ? :accepted : :rejected
    p
end

commit!(s::MF.State,p::Proposal)=MB.commit!(s,p)
raw_ratio(s::MF.State,p::Proposal)=MB.raw_ratio(s,p)
function attempt!(w::Workspace,s::MF.State,rng;observer=nothing,kwargs...)
    p=propose!(w,s,rng;kwargs...); p===nothing && return nothing
    observer===nothing || observer(s,p)
    commit!(s,p); p
end

end
