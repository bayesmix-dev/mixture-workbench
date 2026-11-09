module DPPTripleSplitMerge

using GeneralDPPMixtures, LinearAlgebra, Random, SpecialFunctions
import ..DPPSplitMerge as SM
const G=GeneralDPPMixtures

export Workspace, Proposal, draw_triple, anchor_logprob, partial_scan!, reference_scan!, union_guides,
       quantile_overlap, coupled_scan!,
       propose!, commit!, attempt!, raw_ratio

"""Fixed data/reference-only ordered-anchor law. Inputs are copied, never adapted.

A component draws one index uniformly from each of its three disjoint sets.
The remaining weight draws a uniform ordered triple of distinct observations.
The law does not depend on the current partition, centers, or selected modes.
"""
struct Workspace
    n::Int
    guided_sets::Vector{NTuple{3,Vector{Int}}}
    guided_weight::Float64
    allocation_law::Symbol
    reference_logits::Vector{NamedTuple{(:two,:three),Tuple{Matrix{Float64},Matrix{Float64}}}}
    coupling_floor::Float64
    reference_labels::Vector{NamedTuple{(:two,:three),Tuple{Vector{Int},Vector{Int}}}}
end
function Workspace(m::G.Model;guided_sets=Vector{NTuple{3,Vector{Int}}}(),guided_weight=0.,
        allocation_law=:partial,reference_logits=[],coupling_floor=1e-6,reference_labels=[])
    n=length(m.y);weight=Float64(guided_weight)
    isfinite(weight) && 0<=weight<1 || throw(ArgumentError("guided weight must be finite in [0,1)"))
    bank=NTuple{3,Vector{Int}}[]
    for sets in guided_sets
        length(sets)==3 || throw(ArgumentError("every guided component requires three sets"))
        abc=ntuple(h->Int.(collect(sets[h])),3)
        all(v->!isempty(v) && length(v)==length(unique(v)) && all(t->1<=t<=n,v),abc) ||
            throw(ArgumentError("guided sets require unique valid observation indices"))
        length(unique(vcat(abc...)))==sum(length,abc) || throw(ArgumentError("guided component sets must be disjoint"))
        push!(bank,ntuple(h->sort(abc[h]),3))
    end
    weight==0 || !isempty(bank) || throw(ArgumentError("positive guided weight requires a frozen anchor bank"))
    allocation_law in (:partial,:reference,:coupled_reference,:reference_union,:coupled_reference_union) || throw(ArgumentError("unknown triple allocation law"))
    floor=Float64(coupling_floor)
    isfinite(floor) && 0<floor<=1 || throw(ArgumentError("coupling floor must be finite in (0,1]"))
    guides=NamedTuple{(:two,:three),Tuple{Matrix{Float64},Matrix{Float64}}}[]
    for guide in reference_logits
        hasproperty(guide,:two) && hasproperty(guide,:three) || throw(ArgumentError("reference guide needs two and three logits"))
        two=Matrix{Float64}(guide.two);three=Matrix{Float64}(guide.three)
        size(two)==(n,2) && size(three)==(n,3) && all(isfinite,two) && all(isfinite,three) ||
            throw(ArgumentError("reference logits require finite n-by-two and n-by-three matrices"))
        push!(guides,(two=copy(two),three=copy(three)))
    end
    if allocation_law in (:reference,:coupled_reference)
        !isempty(bank) && length(guides)==length(bank) || throw(ArgumentError("one frozen reference guide is required per anchor component"))
    elseif !isempty(guides)
        length(guides)==length(bank) || throw(ArgumentError("reference guide and anchor bank lengths differ"))
    end
    if allocation_law==:coupled_reference
        for guide in guides,t in 1:n
            quantile_overlap(view(guide.three,t,:),view(guide.two,t,:);coupling_floor=floor)
        end
    end
    references=NamedTuple{(:two,:three),Tuple{Vector{Int},Vector{Int}}}[]
    for reference in reference_labels
        hasproperty(reference,:two) && hasproperty(reference,:three) || throw(ArgumentError("union reference needs two and three label vectors"))
        two=Int.(collect(reference.two));three=Int.(collect(reference.three))
        length(two)==n && length(three)==n && all(h->0<=h<=2,two) && all(h->0<=h<=3,three) ||
            throw(ArgumentError("union reference requires oriented n-vectors with zero outside selected groups"))
        push!(references,(two=copy(two),three=copy(three)))
    end
    if allocation_law in (:reference_union,:coupled_reference_union)
        !isempty(bank) && length(references)==length(bank) || throw(ArgumentError("one frozen union reference is required per anchor component"))
    elseif !isempty(references)
        length(references)==length(bank) || throw(ArgumentError("union reference and anchor bank lengths differ"))
    end
    for (j,reference) in enumerate(references),h in 1:3
        all(t->reference.two[t]==(h<=2 ? 1 : 2) && reference.three[t]==h,bank[j][h]) ||
            throw(ArgumentError("union reference labels must orient every guided anchor set"))
    end
    Workspace(n,bank,weight,allocation_law,guides,floor,references)
end

"""Build both fixed-reference categorical guides on one invariant affected union.

Zero reference labels exclude observations outside the selected reference groups.
Each prototype is the corresponding fixed reference group intersected with the
sorted observation set D. Scores add yt as an additional observation, even when
yt is already in that prototype. The matrices are frozen for an entire proposal;
outside-D rows are zero and are never sampled. No current allocation or center
enters these scores.
"""
function union_guides(m::G.Model,D,reference)
    n=length(m.y);members=sort(Int.(collect(D)))
    !isempty(members) && length(members)==length(unique(members)) && all(t->1<=t<=n,members) ||
        throw(ArgumentError("invalid union-reference observation set"))
    hasproperty(reference,:two) && hasproperty(reference,:three) || throw(ArgumentError("union reference needs two and three labels"))
    function logits(labels,r)
        length(labels)==n && all(h->h isa Integer && 0<=h<=r,labels) || throw(ArgumentError("invalid oriented union-reference labels"))
        values=zeros(n,r)
        for h in 1:r
            prototype=[t for t in members if labels[t]==h]
            !isempty(prototype) || throw(ArgumentError("each union-reference prototype must contain its anchor"))
            c=SM.cluster(m,prototype);base=SM.logA(m,c);weight=log(c.n+m.a)
            for t in members
                oldn,oldmean,oldscatter=c.n,c.mean,c.scatter
                G.add!(c,m.y[t]);values[t,h]=weight+SM.logA(m,c)-base
                c.n=oldn;c.mean=oldmean;c.scatter=oldscatter
            end
        end
        all(isfinite,values) || error("nonfinite union-reference logits")
        values
    end
    (two=logits(reference.two,2),three=logits(reference.three,3))
end

function draw_triple(w::Workspace,rng)
    n=w.n;n>=3 || throw(ArgumentError("an ordered distinct triple requires at least three observations"))
    if w.guided_weight>0 && rand(rng)<w.guided_weight
        component=rand(rng,eachindex(w.guided_sets));sets=w.guided_sets[component]
        return ntuple(h->rand(rng,sets[h]),3),component
    end
    i=rand(rng,1:n);j=rand(rng,1:n-1);j>=i && (j+=1)
    lo,hi=minmax(i,j);l=rand(rng,1:n-2);l>=lo && (l+=1);l>=hi && (l+=1)
    (i,j,l),0
end

function anchor_logprob(w::Workspace,anchors)
    length(anchors)==3 && length(unique(anchors))==3 && all(t->1<=t<=w.n,anchors) ||
        throw(ArgumentError("anchors must be three distinct observation indices"))
    probability=(1-w.guided_weight)/(w.n*(w.n-1.)*(w.n-2.))
    if w.guided_weight>0
        for sets in w.guided_sets
            all(h->anchors[h] in sets[h],1:3) || continue
            probability+=w.guided_weight/length(w.guided_sets)/prod(length,sets)
        end
    end
    log(probability)
end

"""Sequential allocation from anchor-only nonempty groups, not a full-data scan.

Repeated anchor labels are allowed. `forced` evaluates a complete oriented
endpoint by replaying the SAME observation order and does not draw from `rng`.
"""
function partial_scan!(m::G.Model,D,anchors,anchorlabels,order,rng;forced=nothing)
    n=length(m.y)
    length(D)==length(unique(D)) && all(t->1<=t<=n,D) || throw(ArgumentError("invalid partial-scan union"))
    length(anchors)==length(anchorlabels)==3 && length(unique(anchors))==3 && all(in(D),anchors) ||
        throw(ArgumentError("invalid triple anchors"))
    r=maximum(anchorlabels)
    r in (2,3) && sort(unique(anchorlabels))==collect(1:r) || throw(ArgumentError("anchor labels must cover two or three groups"))
    sort(order)==sort(setdiff(D,collect(anchors))) || throw(ArgumentError("order must visit every nonanchor once"))
    if forced!==nothing
        length(forced)==n && all(t->1<=forced[t]<=r,D) && all(h->forced[anchors[h]]==anchorlabels[h],1:3) ||
            throw(ArgumentError("forced endpoint must orient all anchors"))
    end
    groups=[[Int(anchors[j]) for j in eachindex(anchors) if anchorlabels[j]==h] for h in 1:r]
    clusters=[SM.cluster(m,g) for g in groups];scores=zeros(r);logq=0.
    for t in order
        for h in 1:r
            c=clusters[h];before=SM.logA(m,c)
            oldn,oldmean,oldscatter=c.n,c.mean,c.scatter
            G.add!(c,m.y[t]);after=SM.logA(m,c)
            c.n=oldn;c.mean=oldmean;c.scatter=oldscatter
            scores[h]=log(oldn+m.a)+after-before
        end
        top=maximum(scores);normalizer=top+log(sum(exp(score-top) for score in scores))
        h=if forced===nothing
            u=rand(rng);cumulative=0.;chosen=r
            for j in 1:r-1
                cumulative+=exp(scores[j]-normalizer)
                if u<cumulative;chosen=j;break;end
            end
            chosen
        else
            forced[t]
        end
        logq+=scores[h]-normalizer
        push!(groups[h],t);G.add!(clusters[h],m.y[t])
    end
    sort!.(groups),logq
end

"""Independent anchored categorical allocations under frozen finite logits.

Only nonanchor observations are sampled or scored. Raw logits are normalized
per observation using log-sum-exp; no current or proposed cluster statistic is
used. The guide belongs to the selected fixed anchor component, which is kept
as an auxiliary variable and paired with the SAME component in reverse.
"""
function reference_scan!(m::G.Model,D,anchors,anchorlabels,order,rng,logits;forced=nothing)
    n=length(m.y)
    length(D)==length(unique(D)) && all(t->1<=t<=n,D) || throw(ArgumentError("invalid reference-scan union"))
    length(anchors)==length(anchorlabels)==3 && length(unique(anchors))==3 && all(in(D),anchors) ||
        throw(ArgumentError("invalid reference anchors"))
    r=maximum(anchorlabels)
    r in (2,3) && sort(unique(anchorlabels))==collect(1:r) || throw(ArgumentError("anchor labels must cover two or three groups"))
    size(logits)==(n,r) && all(isfinite,logits) || throw(ArgumentError("reference scan requires finite n-by-r logits"))
    sort(order)==sort(setdiff(D,collect(anchors))) || throw(ArgumentError("order must visit every nonanchor once"))
    if forced!==nothing
        length(forced)==n && all(t->1<=forced[t]<=r,D) && all(h->forced[anchors[h]]==anchorlabels[h],1:3) ||
            throw(ArgumentError("forced reference endpoint must orient all anchors"))
    end
    groups=[[Int(anchors[j]) for j in eachindex(anchors) if anchorlabels[j]==h] for h in 1:r];logq=0.
    for t in order
        top=maximum(view(logits,t,:));logz=log(sum(exp(logits[t,h]-top) for h in 1:r))
        h=if forced===nothing
            u=rand(rng);cumulative=0.;chosen=r
            for j in 1:r-1
                cumulative+=exp((logits[t,j]-top)-logz)
                if u<cumulative;chosen=j;break;end
            end
            chosen
        else
            forced[t]
        end
        logq+=(logits[t,h]-top)-logz;push!(groups[h],t)
    end
    sort!.(groups),logq
end

"""Canonical overlap table for the two frozen categorical CDF partitions.

Every row receives (1-epsilon)*softmax(logits)+epsilon/r. Stored CDF endpoints
are exactly zero and one. The returned marginals sum the SAME finite overlap
table and are the actual forward/reverse categorical normalizers.
"""
function quantile_overlap(old_logits,new_logits;coupling_floor=1e-6)
    floor=Float64(coupling_floor)
    isfinite(floor) && 0<floor<=1 || throw(ArgumentError("coupling floor must be finite in (0,1]"))
    sort([length(old_logits),length(new_logits)])==[2,3] && all(isfinite,old_logits) && all(isfinite,new_logits) ||
        throw(ArgumentError("coupling requires finite two- and three-category logits"))
    function bounds(logits)
        r=length(logits);top=maximum(logits);weights=exp.(Float64.(logits).-top)
        probabilities=(1-floor).*(weights/sum(weights)).+floor/r
        probabilities./=sum(probabilities)
        cdf=zeros(r+1)
        for h in 1:r-1;cdf[h+1]=cdf[h]+probabilities[h];end
        cdf[end]=1.
        all(>(0),diff(cdf)) || throw(ArgumentError("coupling floor does not yield representable positive CDF widths"))
        cdf
    end
    old=bounds(old_logits);new=bounds(new_logits)
    J=[max(0.,min(old[h+1],new[g+1])-max(old[h],new[g])) for h in 1:length(old_logits),g in 1:length(new_logits)]
    rowmass=vec(sum(J;dims=2));colmass=vec(sum(J;dims=1))
    all(>(0),rowmass) && all(>(0),colmass) || error("invalid coupling overlap marginals")
    (joint=J,oldmass=rowmass,newmass=colmass)
end

"""Couple CURRENT oriented labels through frozen categorical interval overlaps.

Forward probabilities condition on each old label; reverse probabilities
condition on the realized new label using the transposed SAME overlap table.
Anchors contribute probability one. `forced` replays a new endpoint without
using the RNG; an impossible overlap returns both log probabilities as -Inf.
"""
function coupled_scan!(m::G.Model,D,anchors,oldanchorlabels,newanchorlabels,order,rng,
        initial,old_logits,new_logits;forced=nothing,coupling_floor=1e-6)
    n=length(m.y)
    length(D)==length(unique(D)) && all(t->1<=t<=n,D) || throw(ArgumentError("invalid coupled-scan union"))
    length(anchors)==length(oldanchorlabels)==length(newanchorlabels)==3 && length(unique(anchors))==3 && all(in(D),anchors) ||
        throw(ArgumentError("invalid coupled anchors"))
    oldr=maximum(oldanchorlabels);newr=maximum(newanchorlabels)
    sort([oldr,newr])==[2,3] && sort(unique(oldanchorlabels))==collect(1:oldr) && sort(unique(newanchorlabels))==collect(1:newr) ||
        throw(ArgumentError("coupled anchor labels must cover two and three groups"))
    sort(order)==sort(setdiff(D,collect(anchors))) || throw(ArgumentError("order must visit every nonanchor once"))
    length(initial)==n && all(t->1<=initial[t]<=oldr,D) && all(h->initial[anchors[h]]==oldanchorlabels[h],1:3) ||
        throw(ArgumentError("initial coupled labels must orient the old anchors"))
    size(old_logits)==(n,oldr) && size(new_logits)==(n,newr) && all(isfinite,old_logits) && all(isfinite,new_logits) ||
        throw(ArgumentError("invalid coupled reference logit matrices"))
    floor=Float64(coupling_floor)
    isfinite(floor) && 0<floor<=1 || throw(ArgumentError("coupling floor must be finite in (0,1]"))
    if forced!==nothing
        length(forced)==n && all(t->1<=forced[t]<=newr,D) && all(h->forced[anchors[h]]==newanchorlabels[h],1:3) ||
            throw(ArgumentError("forced coupled endpoint must orient the new anchors"))
    end
    groups=[[Int(anchors[j]) for j in eachindex(anchors) if newanchorlabels[j]==h] for h in 1:newr]
    qforward=0.;qreverse=0.
    for t in order
        law=quantile_overlap(view(old_logits,t,:),view(new_logits,t,:);coupling_floor=floor)
        h=initial[t]
        g=if forced===nothing
            u=rand(rng)*law.oldmass[h];cumulative=0.
            positive=findall(>(0),view(law.joint,h,:));chosen=last(positive)
            for j in positive
                cumulative+=law.joint[h,j]
                if u<cumulative;chosen=j;break;end
            end
            chosen
        else
            forced[t]
        end
        overlap=law.joint[h,g]
        if overlap>0
            qforward+=log(overlap)-log(law.oldmass[h])
            qreverse+=log(overlap)-log(law.newmass[g])
        else
            qforward=-Inf;qreverse=-Inf
        end
        push!(groups[g],t)
    end
    sort!.(groups),qforward,qreverse
end

mutable struct Proposal{X}
    kind::Symbol
    anchors::NTuple{3,Int}
    anchor_component::Int
    anchor_logprob::Float64
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
    support::Vector{Int}
    union_size::Int
    Gterm::Union{Nothing,Float64}
    H::Union{Nothing,Float64}
    DK::Union{Nothing,Float64}
    logforward::Union{Nothing,Float64}
    logreverse::Union{Nothing,Float64}
    ratio::Union{Nothing,Float64}
    logu::Union{Nothing,Float64}
    outcome::Symbol
end

geometry_call(f,l)=l===nothing ? f() : Base.lock(f,l)

"""Exact cold fixed-spectrum local three-to-two / two-to-three reallocation.

The fixed ordered triple (i,j,l) selects a merge when its labels are all
different and a split when c_i=c_j!=c_l; all other patterns are self-loops.
All three old groups can redistribute into two groups, or the inverse, using
anchor-only sequential allocation. Every daughter has an anchor, hence the
reverse affected union is identical. The fixed triple law and the SAME uniform
scan-order law cancel. No cluster-size selection factor or factor two occurs.

With `allocation_law=:reference`, guided component j uses its own frozen
two/three-group logits in both directions. Component j is explicitly retained
as an auxiliary; its joint component/anchor density cancels, rather than
marginalizing allocation probabilities across components. With
`:coupled_reference`, current labels are transported through the fixed CDF
overlap channel and both conditional q values are evaluated. Uniform component
zero retains the original partial allocation law. Diagnostic forced anchors
without an explicit `anchor_component` also retain the original partial law.
With `:reference_union`, the same fixed reference groups are first intersected
with the affected union D. Both guide matrices are built once, before sampling.
Every old/new group has an anchor, so the reverse union is exactly D and the
same component's matrices are identical in reverse. No additional selection
factor is required. The fixed target parameters are unchanged by this move.
`:coupled_reference_union` transports current labels through the same CDF
channel as `:coupled_reference`, using those invariant union guide matrices.

Only affected centers are independently drawn from the untruncated Student
location laws. Outside-box draws self-loop. S and its natural capacity are
unchanged. Diagnostic `anchors` / `order` overrides do not redefine production
selection; callers must not select them from the current state in production.
"""
function propose!(w::Workspace,s::G.MarginalState,rng;anchors=nothing,anchor_component=nothing,order=nothing,
        forced_labels=nothing,geometry_lock=nothing)
    m=s.model;n=length(s.c);w.n==n || throw(ArgumentError("anchor bank has different data length"))
    anchors===nothing && anchor_component!==nothing && throw(ArgumentError("an explicit anchor component requires explicit anchors"))
    forced_labels!==nothing && !(anchors!==nothing && anchor_component isa Integer && anchor_component>0 && w.allocation_law in (:coupled_reference,:coupled_reference_union)) &&
        throw(ArgumentError("forced new labels require explicit guided coupled-reference anchors"))
    n<3 && return nothing
    aa,component=anchors===nothing ? draw_triple(w,rng) : (Tuple(Int.(collect(anchors))),-1)
    anchorlp=anchor_logprob(w,aa);i,j,l=aa
    if anchor_component!==nothing
        anchor_component isa Integer && 0<=anchor_component<=length(w.guided_sets) || throw(ArgumentError("invalid forced anchor component"))
        component=Int(anchor_component)
        component==0 || all(h->aa[h] in w.guided_sets[component][h],1:3) || throw(ArgumentError("forced anchors are outside their component sets"))
    end
    ids=unique([s.c[i],s.c[j],s.c[l]])
    split=s.c[i]==s.c[j] && s.c[i]!=s.c[l];merge=length(ids)==3
    kind=split ? :triple_split : merge ? :triple_merge : :ineligible
    k=length(s.active);M=length(s.S);k<=M || error("occupied DPP count exceeds fixed spectral rank")
    oldgroups=[findall(==(h),s.c) for h in ids];D=sort(vcat(oldgroups...))
    oldlabels=split ? [1,1,2] : merge ? [1,2,3] : Int[]
    newlabels=split ? [1,2,3] : merge ? [1,1,2] : Int[]
    p=Proposal(kind,aa,component,anchorlp,oldlabels,newlabels,Int[],ids,oldgroups,Vector{Int}[],
        [s.clusters[h].x for h in ids],typeof(first(m.y))[],k,k+(split ? 1 : merge ? -1 : 0),M,copy(s.S),length(D),
        nothing,nothing,nothing,nothing,nothing,nothing,nothing,:constructed)
    if !(split||merge);p.outcome=:ineligible_pattern;return p;end
    if split && k==M;p.outcome=:capacity;return p;end
    p.order=order===nothing ? shuffle(rng,setdiff(D,collect(aa))) : Int.(copy(order))
    unionreference=w.allocation_law in (:reference_union,:coupled_reference_union) && component>0
    reference=w.allocation_law in (:reference,:reference_union) && component>0
    coupled=w.allocation_law in (:coupled_reference,:coupled_reference_union) && component>0
    guide=unionreference ? union_guides(m,D,w.reference_labels[component]) : (reference || coupled) ? w.reference_logits[component] : nothing
    if coupled
        initial=zeros(Int,n)
        for h in eachindex(oldgroups);initial[oldgroups[h]].=h;end
        oldlogits=maximum(oldlabels)==2 ? guide.two : guide.three
        newlogits=maximum(newlabels)==2 ? guide.two : guide.three
        p.newgroups,p.logforward,p.logreverse=coupled_scan!(m,D,aa,oldlabels,newlabels,p.order,rng,
            initial,oldlogits,newlogits;forced=forced_labels,coupling_floor=w.coupling_floor)
        if p.logforward==-Inf;p.outcome=:zero_proposal_density;return p;end
    elseif reference
        logits=maximum(newlabels)==2 ? guide.two : guide.three
        p.newgroups,p.logforward=reference_scan!(m,D,aa,newlabels,p.order,rng,logits)
    else
        p.newgroups,p.logforward=partial_scan!(m,D,aa,newlabels,p.order,rng)
    end
    if !coupled
        forced=zeros(Int,n)
        for h in eachindex(oldgroups);forced[oldgroups[h]].=h;end
        if reference
            logits=maximum(oldlabels)==2 ? guide.two : guide.three
            _,p.logreverse=reference_scan!(m,D,aa,oldlabels,p.order,rng,logits;forced)
        else
            _,p.logreverse=partial_scan!(m,D,aa,oldlabels,p.order,rng;forced)
        end
    end
    oldclusters=[SM.cluster(m,g) for g in oldgroups];newclusters=[SM.cluster(m,g) for g in p.newgroups]
    p.Gterm=sum(loggamma(m.a+c.n)-loggamma(m.a) for c in newclusters)-
        sum(loggamma(m.a+c.n)-loggamma(m.a) for c in oldclusters)
    p.H=sum(SM.logA(m,c) for c in newclusters)-sum(SM.logA(m,c) for c in oldclusters)
    for c in newclusters;push!(p.newxs,G.student_draw(rng,SM.location_law(m,c)...));end
    if !all(x->G.inside(m,x),p.newxs);p.outcome=:outside_domain;return p;end
    unaffected=[s.clusters[h].x for h in s.active if !(h in ids)]
    p.DK=geometry_call(geometry_lock) do;SM.geometry_delta(m,s.S,unaffected,p.oldxs,p.newxs);end
    if p.DK==-Inf;p.outcome=:zero_determinant;return p;end
    p.ratio=p.Gterm+p.H+p.DK+p.logreverse-p.logforward
    isfinite(p.ratio) || error("nonfinite triple split/merge acceptance ratio")
    p.logu=log(rand(rng));p.outcome=p.logu<min(0.,p.ratio) ? :accepted : :mh_rejected
    p
end

"""Commit only after complete membership, numerical and cache validation."""
function commit!(s::G.MarginalState,p::Proposal;geometry_lock=nothing)
    p.outcome==:accepted || return s
    m=s.model;oldr=length(p.oldids);newr=length(p.newgroups)
    (p.kind==:triple_split && oldr==2 && newr==3 || p.kind==:triple_merge && oldr==3 && newr==2) ||
        error("invalid triple proposal dimensions")
    length(p.newxs)==newr && p.k==length(s.active) && p.M==length(s.S) && p.support==s.S && p.newk<=p.M || error("stale/invalid triple proposal")
    all(h->findall(==(p.oldids[h]),s.c)==p.oldgroups[h] && s.clusters[p.oldids[h]].x==p.oldxs[h],1:oldr) || error("stale triple proposal endpoint")
    D=sort(vcat(p.oldgroups...));sort(vcat(p.newgroups...))==D && all(g->!isempty(g),p.newgroups) || error("new groups must partition the affected union")
    all(x->G.inside(m,x),p.newxs) || error("accepted triple centers outside domain")
    ids=copy(p.oldids)
    if newr>oldr
        free=findfirst(t->s.clusters[t].n==0,eachindex(s.clusters))
        push!(ids,free===nothing ? length(s.clusters)+1 : free)
    end
    cl=copy(s.clusters);labels=copy(s.c)
    for h in 1:newr
        id=ids[h];cc=SM.cluster(m,p.newgroups[h],p.newxs[h])
        id>length(cl) ? push!(cl,cc) : (cl[id]=cc)
        labels[p.newgroups[h]].=id
    end
    for id in ids[newr+1:end];cl[id]=G.Cluster(m,zero(first(m.y)));end
    active=vcat([h for h in s.active if !(h in p.oldids)],ids[1:newr])
    sort!(active;by=h->findfirst(==(h),labels))
    length(active)==p.newk && sort(unique(labels))==sort(active) || error("invalid triple cache membership")
    xs=[cl[h].x for h in active];F=G.feature_matrix(m,s.S,xs);Q,ld=G.orthobasis(F,G.Counters())
    physical=geometry_call(geometry_lock) do;SM.physical_logdet(m,s.S,xs);end
    isfinite(ld) && isfinite(physical) && abs(ld-length(active)*m.logvolume-physical)<1e-7 || error("accepted triple cache inconsistency")
    G.grow_streams!(s.streams,length(cl))
    s.clusters=cl;s.c=labels;s.active=active;s.F=F;s.Q=Q;s.ld=ld
    s.counters.projector_rebuilds+=1
    s
end

function attempt!(w::Workspace,s::G.MarginalState,rng;observer=nothing,geometry_lock=nothing,kwargs...)
    p=propose!(w,s,rng;geometry_lock,kwargs...);p===nothing && return nothing
    observer===nothing || observer(s,p)
    commit!(s,p;geometry_lock);p
end

"""Independent unsimplified full Gram target and Student proposal densities."""
function raw_ratio(m,S,unaffected,oldgroups,oldxs,newgroups,newxs,logforward,logreverse)
    function target(groups,xs)
        allxs=vcat(unaffected,xs);all(x->G.inside(m,x),allxs) || return -Inf
        F=G.feature_matrix(m,S,allxs)/sqrt(m.volume)
        ld=2sum(log,diag(cholesky(Symmetric(F'F)).L))
        ld+sum(loggamma(m.a+length(g))-loggamma(m.a)+G.log_marginal(SM.cluster(m,g,x),m) for (g,x) in zip(groups,xs))
    end
    loglocations(groups,xs)=sum(G.student_logdensity(x,SM.location_law(m,SM.cluster(m,g))...) for (g,x) in zip(groups,xs))
    delta=target(newgroups,newxs)-target(oldgroups,oldxs)
    qforward=logforward+loglocations(newgroups,newxs)
    qreverse=logreverse+loglocations(oldgroups,oldxs)
    (target_difference=delta,log_forward=qforward,log_reverse=qreverse,raw=delta+qreverse-qforward)
end
function raw_ratio(s::G.MarginalState,p::Proposal)
    unaffected=[s.clusters[h].x for h in s.active if !(h in p.oldids)]
    raw_ratio(s.model,s.S,unaffected,p.oldgroups,p.oldxs,p.newgroups,p.newxs,p.logforward,p.logreverse)
end

end
