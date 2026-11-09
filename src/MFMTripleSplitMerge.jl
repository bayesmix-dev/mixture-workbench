module MFMTripleSplitMerge

using GeneralDPPMixtures, Random, SpecialFunctions
import ..MFMBaseline as MF
import ..MFMTarget as MT
import ..MFMBlockMoves as MB
const G=GeneralDPPMixtures

export Workspace, Proposal, draw_triple, anchor_logprob, partial_scan!, reference_scan!, union_guides,
       quantile_overlap, coupled_scan!,
       propose!, commit!, attempt!, raw_ratio

"""Copied, frozen data/reference-only law for ordered distinct anchor triples.

Each guided component draws independently from three disjoint observation sets.
Its mixture with uniform ordered triples is independent of the current state.
"""
struct Workspace
    n::Int
    guided_sets::Vector{NTuple{3,Vector{Int}}}
    guided_weight::Float64
    allocation_law::Symbol
    reference_logits::Vector{NamedTuple{(:two,:three),Tuple{Matrix{Float64},Matrix{Float64}}}}
    reference_labels::Vector{NamedTuple{(:two,:three),Tuple{Vector{Int},Vector{Int}}}}
    coupling_floor::Float64
end
function Workspace(m::G.Model;guided_sets=Vector{NTuple{3,Vector{Int}}}(),guided_weight=0.,
        allocation_law=:partial,reference_logits=[],reference_labels=[],coupling_floor=1e-6)
    n=length(m.y); weight=Float64(guided_weight)
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
    weight==0 || !isempty(bank) || throw(ArgumentError("positive guided weight requires a frozen bank"))
    allocation_law in (:partial,:reference,:reference_union,:coupled_reference_union) || throw(ArgumentError("unknown triple allocation law"))
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
    if allocation_law==:reference
        !isempty(bank) && length(guides)==length(bank) || throw(ArgumentError("one frozen reference guide is required per anchor component"))
    elseif !isempty(guides)
        length(guides)==length(bank) || throw(ArgumentError("reference guide and anchor bank lengths differ"))
    end
    labels=NamedTuple{(:two,:three),Tuple{Vector{Int},Vector{Int}}}[]
    for reference in reference_labels
        hasproperty(reference,:two) && hasproperty(reference,:three) && reference.two isa AbstractVector && reference.three isa AbstractVector ||
            throw(ArgumentError("union reference requires two and three label vectors"))
        two=Int.(collect(reference.two));three=Int.(collect(reference.three))
        length(two)==length(three)==n && all(x->0<=x<=2,two) && all(x->0<=x<=3,three) ||
            throw(ArgumentError("union reference labels require n entries in0:2 and0:3"))
        push!(labels,(two=copy(two),three=copy(three)))
    end
    if allocation_law in (:reference_union,:coupled_reference_union) || !isempty(labels)
        !isempty(bank) && length(labels)==length(bank) || throw(ArgumentError("one union reference pair is required per anchor component"))
        for (j,reference) in enumerate(labels),h in 1:3
            all(t->reference.two[t]==(h==3 ? 2 : 1) && reference.three[t]==h,bank[j][h]) ||
                throw(ArgumentError("union references must orient anchor sets as112 and123"))
        end
    end
    Workspace(n,bank,weight,allocation_law,guides,labels,floor)
end
function draw_triple(w::Workspace,rng)
    n=w.n; n>=3 || throw(ArgumentError("three distinct observations are required"))
    if w.guided_weight>0 && rand(rng)<w.guided_weight
        component=rand(rng,eachindex(w.guided_sets)); sets=w.guided_sets[component]
        return ntuple(h->rand(rng,sets[h]),3),component
    end
    i=rand(rng,1:n); j=rand(rng,1:n-1); j>=i && (j+=1)
    lo,hi=minmax(i,j); l=rand(rng,1:n-2); l>=lo && (l+=1); l>=hi && (l+=1)
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

"""Anchor-only partial sequential NIW scan into two or three nonempty groups.

The state-specific mean prior (mu0,kappa), gamma, and covariance prior are used.
There is no unused-component birth mass inside this fixed-output-count scan.
`forced` evaluates an oriented endpoint in the SAME order without RNG draws.
"""
function partial_scan!(s::MF.State,D,anchors,anchorlabels,order,rng;forced=nothing,beta=1.)
    beta=MB.checkbeta(beta); n=length(s.c)
    length(D)==length(unique(D)) && all(t->1<=t<=n,D) || throw(ArgumentError("invalid partial-scan union"))
    length(anchors)==length(anchorlabels)==3 && length(unique(anchors))==3 && all(in(D),anchors) ||
        throw(ArgumentError("invalid triple anchors"))
    r=maximum(anchorlabels)
    r in (2,3) && sort(unique(anchorlabels))==collect(1:r) || throw(ArgumentError("anchor labels must cover two or three groups"))
    sort(order)==sort(setdiff(D,collect(anchors))) || throw(ArgumentError("order must visit every nonanchor once"))
    if forced!==nothing
        length(forced)==n && all(t->1<=forced[t]<=r,D) && all(h->forced[anchors[h]]==anchorlabels[h],1:3) ||
            throw(ArgumentError("forced endpoint must orient every anchor"))
    end
    groups=[[Int(anchors[j]) for j in eachindex(anchors) if anchorlabels[j]==h] for h in 1:r]
    clusters=[MB.cluster(s,g) for g in groups]
    cached=MB.cache_state(s,clusters,zeros(Int,n),collect(1:r)); scores=zeros(r); weights=zeros(r); logq=0.
    for t in order
        for h in 1:r
            scores[h]=log(clusters[h].n+s.gamma)+beta*MF.logpred(cached,h,s.model.y[t])
        end
        top=maximum(scores)
        for h in 1:r; weights[h]=exp(scores[h]-top); end
        lognormalizer=top+log(sum(weights))
        h=forced===nothing ? G.draw_weights(rng,weights) : forced[t]
        logq+=scores[h]-lognormalizer
        push!(groups[h],t); G.add!(clusters[h],s.model.y[t]); MF.refresh!(cached,h)
    end
    sort!.(groups),logq
end

"""Independent nonanchor categoricals under the realized component's fixed guide.

Each row is normalized in log space. Anchor allocations are fixed, not sampled
or scored. The same component and its two/three-column matrices must be retained
in the reverse auxiliary route; probabilities are not mixed across components.
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
            throw(ArgumentError("forced reference endpoint must orient every anchor"))
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

"""NIW guides from fixed reference groups intersected with the affected union.

The sorted union D is identical in reverse because every affected old/new block
has an anchor. These guides depend only on D, fixed reference labels and fixed
within-move gamma/private NIW parameters, never the current labels within D.
Rows outside D are deterministic zeros and are never scored. Reference label0
marks points outside selected fixed groups; those points can still be allocated.
"""
function union_guides(s::MF.State,D,reference)
    n=length(s.c);indices=sort(Int.(collect(D)))
    length(indices)==length(unique(indices)) && all(t->1<=t<=n,indices) || throw(ArgumentError("invalid union guide indices"))
    length(reference.two)==length(reference.three)==n && all(x->0<=x<=2,reference.two) && all(x->0<=x<=3,reference.three) ||
        throw(ArgumentError("invalid oriented union reference labels"))
    function logits(labels,r)
        groups=[[t for t in indices if labels[t]==h] for h in 1:r]
        all(!isempty,groups) || throw(ArgumentError("every restricted reference prototype requires an anchor"))
        clusters=[MB.cluster(s,g) for g in groups];cached=MB.cache_state(s,clusters,zeros(Int,n),collect(1:r));L=zeros(n,r)
        for h in 1:r
            weight=log(clusters[h].n+s.gamma)
            for t in indices;L[t,h]=weight+MF.logpred(cached,h,s.model.y[t]);end
        end
        all(isfinite,L) || error("nonfinite restricted NIW reference guide")
        L
    end
    (two=logits(reference.two,2),three=logits(reference.three,3))
end

"""Finite overlap table of two/three categorical CDF partitions.

The floor mixes each softmax with a uniform law. Forward and reverse
normalizers are marginals of the SAME finite table; swapping input matrices
returns its transpose. No target density is changed by this proposal floor.
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

"""Transport current oriented labels through a fixed two/three overlap table.

Each nonanchor uses qforward=J[old,new]/row[old] and
qreverse=J[old,new]/column[new]. Anchors are deterministic. A forced endpoint
only evaluates the law and consumes no RNG. Zero-overlap paths have zero
probability in both directions, including their explicit diagnostic replay.
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

mutable struct Proposal{M,X}
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
    oldallocations::Vector{Int}
    oldactive::Vector{Int}
    model::M
    mu0::X
    kappa::Float64
    k::Int
    newk::Int
    M::Int
    gamma::Float64
    lambda::Float64
    beta::Float64
    union_size::Int
    Gterm::Union{Nothing,Float64}
    H::Union{Nothing,Float64}
    countterm::Union{Nothing,Float64}
    logforward::Union{Nothing,Float64}
    logreverse::Union{Nothing,Float64}
    ratio::Union{Nothing,Float64}
    logu::Union{Nothing,Float64}
    outcome::Symbol
    newM::Int
    joint_M::Bool
end
# Keep the frozen fixed-M diagnostic constructor usable. The additional fields
# describe the optional count map; they do not change the old proposal law.
Proposal(args::Vararg{Any,30})=Proposal(args...,args[18],false)

"""Exact local 3-to-2 / 2-to-3 partition proposal at fixed M,gamma,Lambda.

For the fixed ordered triple (i,j,l), three distinct blocks become two oriented
blocks i=j versus l; the inverse requires c_i=c_j!=c_l. Other patterns self-loop.
All old and new affected blocks have anchors, so the reverse union and uniform
remaining-index permutation agree. The frozen triple law cancels. The only count
factor is (M)_(Knew)/(M)_(Kold); no cluster-size or factorial selection term occurs.
NIW means are integrated using the state's private mean prior. Only NIW evidence
is multiplied by optional beta. Explicit anchors/order are diagnostic overrides.
With allocation_law=:reference each guided component retains its own frozen
matrix pair in both directions. Component0 uses the old partial law; forced
anchors without an explicit component also retain the old partial law. Matrices
may depend on fixed parameters (gamma,mu0,kappa,Omega,tau), which this move leaves
unchanged. To use current-parameter guides, rebuild them when those parameters
change between calls; a guide frozen at other parameters is also a valid proposal.
With :reference_union, prototype data are fixed reference groups intersected
with the same reverse-invariant D. Both guide matrices are computed once per
proposal and reused in forward/reverse; the MH accounting is unchanged.
The optional joint_M=true map sets Mnew=M+(Knew-K), preserving M-K. Its count
factor also includes the full-data Dirichlet normalizer and shifted-Poisson
prior ratio. Gamma and Lambda stay fixed. There is no integer-map Jacobian.
With :coupled_reference_union, current oriented labels pass through the
two/three CDF overlap channel built from the same invariant union guides.
Forward and reverse probabilities condition on the old and new labels,
respectively. The same component, D and guide matrices are retained in reverse.
"""
function propose!(w::Workspace,s::MF.State,rng;anchors=nothing,anchor_component=nothing,order=nothing,beta=1.,joint_M::Bool=false,forced_labels=nothing)
    beta=MB.checkbeta(beta); n=length(s.c); w.n==n || throw(ArgumentError("anchor bank data length differs"))
    anchors===nothing && anchor_component!==nothing && throw(ArgumentError("an explicit anchor component requires explicit anchors"))
    forced_labels!==nothing && !(anchors!==nothing && anchor_component isa Integer && anchor_component>0 && w.allocation_law==:coupled_reference_union) &&
        throw(ArgumentError("forced new labels require explicit guided coupled-reference anchors"))
    n<3 && return nothing
    aa,component=anchors===nothing ? draw_triple(w,rng) : (Tuple(Int.(collect(anchors))),-1)
    anchorlp=anchor_logprob(w,aa); i,j,l=aa
    if anchor_component!==nothing
        anchor_component isa Integer && 0<=anchor_component<=length(w.guided_sets) || throw(ArgumentError("invalid forced anchor component"))
        component=Int(anchor_component)
        component==0 || all(h->aa[h] in w.guided_sets[component][h],1:3) || throw(ArgumentError("forced anchors are outside their component sets"))
    end
    ids=unique([s.c[i],s.c[j],s.c[l]])
    split=s.c[i]==s.c[j] && s.c[i]!=s.c[l]; merge=length(ids)==3
    kind=split ? :triple_split : merge ? :triple_merge : :ineligible
    k=length(s.active); k<=s.M || error("occupied MFM count exceeds M")
    oldgroups=[findall(==(h),s.c) for h in ids]; D=sort(vcat(oldgroups...))
    oldlabels=split ? [1,1,2] : merge ? [1,2,3] : Int[]
    newlabels=split ? [1,2,3] : merge ? [1,1,2] : Int[]
    p=Proposal(kind,aa,component,anchorlp,oldlabels,newlabels,Int[],ids,oldgroups,Vector{Int}[],
        copy(s.c),copy(s.active),s.model,s.mu0,s.kappa,k,k+(split ? 1 : merge ? -1 : 0),s.M,s.gamma,s.lambda,beta,length(D),
        nothing,nothing,nothing,nothing,nothing,nothing,nothing,:constructed)
    p.joint_M=joint_M
    p.newM=joint_M ? s.M+p.newk-k : s.M
    if !(split||merge); p.outcome=:ineligible_pattern; return p; end
    if split && k==s.M && !joint_M; p.outcome=:capacity; return p; end
    1<=p.newk<=p.newM || error("invalid joint MFM triple count map")
    p.order=order===nothing ? shuffle(rng,setdiff(D,collect(aa))) : Int.(copy(order))
    reference=w.allocation_law in (:reference,:reference_union) && component>0
    coupled=w.allocation_law==:coupled_reference_union && component>0
    guide=reference || coupled ? (w.allocation_law==:reference ? w.reference_logits[component] : union_guides(s,D,w.reference_labels[component])) : nothing
    if coupled
        initial=zeros(Int,n)
        for h in eachindex(oldgroups);initial[oldgroups[h]].=h;end
        oldlogits=maximum(oldlabels)==2 ? guide.two : guide.three
        newlogits=maximum(newlabels)==2 ? guide.two : guide.three
        p.newgroups,p.logforward,p.logreverse=coupled_scan!(s.model,D,aa,oldlabels,newlabels,p.order,rng,
            initial,oldlogits,newlogits;forced=forced_labels,coupling_floor=w.coupling_floor)
        if p.logforward==-Inf;p.outcome=:zero_proposal_density;return p;end
    elseif reference
        logits=maximum(newlabels)==2 ? guide.two : guide.three
        p.newgroups,p.logforward=reference_scan!(s.model,D,aa,newlabels,p.order,rng,logits)
    else
        p.newgroups,p.logforward=partial_scan!(s,D,aa,newlabels,p.order,rng;beta)
    end
    if !coupled
        forced=zeros(Int,n)
        for h in eachindex(oldgroups); forced[oldgroups[h]].=h; end
        if reference
            logits=maximum(oldlabels)==2 ? guide.two : guide.three
            _,p.logreverse=reference_scan!(s.model,D,aa,oldlabels,p.order,rng,logits;forced)
        else
            _,p.logreverse=partial_scan!(s,D,aa,oldlabels,p.order,rng;forced,beta)
        end
    end
    oldclusters=[MB.cluster(s,g) for g in oldgroups]; newclusters=[MB.cluster(s,g) for g in p.newgroups]
    p.Gterm=sum(loggamma(s.gamma+c.n)-loggamma(s.gamma) for c in newclusters)-
        sum(loggamma(s.gamma+c.n)-loggamma(s.gamma) for c in oldclusters)
    p.H=sum(MT.log_marginal(s,c) for c in newclusters)-sum(MT.log_marginal(s,c) for c in oldclusters)
    p.countterm=if joint_M
        # Full-data Dirichlet normalizer and shifted-Poisson M prior. Gamma,
        # Lambda and all NIW parameters stay fixed; the integer count map has
        # no Jacobian. In particular, n here is not the affected union size.
        MT.logfallingfactorial(p.newM,p.newk)-MT.logfallingfactorial(s.M,k)+
            loggamma(p.newM*s.gamma)-loggamma(p.newM*s.gamma+n)-
            loggamma(s.M*s.gamma)+loggamma(s.M*s.gamma+n)+
            (p.newM-s.M)*log(s.lambda)-loggamma(p.newM)+loggamma(s.M)
    else
        split ? log(s.M-k) : -log(s.M-k+1)
    end
    p.ratio=p.Gterm+beta*p.H+p.countterm+p.logreverse-p.logforward
    isfinite(p.ratio) || error("nonfinite MFM triple acceptance ratio")
    p.logu=log(rand(rng)); p.outcome=p.logu<min(0.,p.ratio) ? :accepted : :mh_rejected
    p
end

"""Validate the original endpoint and build every canonical NIW cache before commit."""
function commit!(s::MF.State,p::Proposal)
    p.outcome==:accepted || return s
    oldr=length(p.oldids); newr=length(p.newgroups)
    (p.kind==:triple_split && oldr==2 && newr==3 || p.kind==:triple_merge && oldr==3 && newr==2) ||
        error("invalid MFM triple proposal dimensions")
    s.M==p.M && s.gamma==p.gamma && s.lambda==p.lambda && s.kappa==p.kappa && s.mu0==p.mu0 && s.model===p.model ||
        error("MFM target changed between proposal and commit")
    s.c==p.oldallocations && s.active==p.oldactive && length(s.active)==p.k || error("stale MFM triple partition")
    p.newM==(p.joint_M ? p.M+p.newk-p.k : p.M) &&
        1<=p.newk<=p.newM && p.newk==p.k+newr-oldr || error("MFM triple capacity/count mismatch")
    all(h->findall(==(p.oldids[h]),s.c)==p.oldgroups[h],1:oldr) || error("stale MFM triple affected blocks")
    D=sort(vcat(p.oldgroups...))
    all(!isempty,p.newgroups) && sort(vcat(p.newgroups...))==D || error("new blocks must partition the affected union")
    groups=vcat([findall(==(h),s.c) for h in s.active if !(h in p.oldids)],p.newgroups)
    all(!isempty,groups) && sort(vcat(groups...))==collect(eachindex(s.c)) && length(groups)==p.newk ||
        error("MFM triple proposal does not partition the full data")
    sort!(groups;by=minimum)
    labels=zeros(Int,length(s.c))
    for h in eachindex(groups); labels[groups[h]].=h; end
    clusters=[MB.cluster(s,g) for g in groups]
    cached=MB.cache_state(s,clusters,labels,collect(eachindex(groups)))
    # No RNG or hyperparameter update is performed. Every factorization above
    # succeeds before the first assignment to the original state.
    s.clusters=cached.clusters; s.c=cached.c; s.active=cached.active
    s.predL=cached.predL; s.predmu=cached.predmu; s.prednorm=cached.prednorm
    s.M=p.newM
    s
end

"""Unsimplified full canonical MT target, retaining all unaffected blocks."""
function raw_ratio(s::MF.State,p::Proposal)
    p.logforward===nothing && throw(ArgumentError("proposal has no endpoint density"))
    unaffected=[findall(==(h),p.oldallocations) for h in p.oldactive if !(h in p.oldids)]
    oldgroups=vcat(unaffected,p.oldgroups); newgroups=vcat(unaffected,p.newgroups)
    oldtarget=MT.logtarget(s;groups=oldgroups,M=p.M,gamma=p.gamma,lambda=p.lambda)
    newtarget=MT.logtarget(s;groups=newgroups,M=p.newM,gamma=p.gamma,lambda=p.lambda)
    if p.beta!=1.
        oldtarget+=(p.beta-1)*sum(MT.log_marginal(s,g) for g in oldgroups)
        newtarget+=(p.beta-1)*sum(MT.log_marginal(s,g) for g in newgroups)
    end
    (target_difference=newtarget-oldtarget,log_forward=p.logforward,log_reverse=p.logreverse,
        raw=newtarget-oldtarget+p.logreverse-p.logforward)
end
function attempt!(w::Workspace,s::MF.State,rng;observer=nothing,kwargs...)
    p=propose!(w,s,rng;kwargs...); p===nothing && return nothing
    observer===nothing || observer(s,p)
    commit!(s,p); p
end

end
