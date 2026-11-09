# Unselected research prototype: whole-partition proposal screens gave
# negligible acceptance. This module is excluded from Benchmark and drivers.
module MFMReferenceInvolution

using GeneralDPPMixtures, Random, LinearAlgebra, Statistics
import ..MFMBaseline as MF
import ..MFMTarget as MT
const G=GeneralDPPMixtures

export Workspace, ReferencePair, Proposal, alignment, transform, propose!, attempt!, commit!

struct ReferencePair
    lower_reference::Int
    upper_reference::Int
    upper_to_common::Vector{Int}
    anchors::Vector{Int}
    agreement::Int
end

"""Static reference transport. No reference or pair selection depends on the
evolving state. The L slots bound this proposal's augmentation, not the target.
References, labels, alignments and anchor observations are frozen at creation.
"""
struct Workspace
    references::Vector{Vector{Int}}
    pairs::Vector{ReferencePair}
    lower::Int
    slots::Int
    mask_probability::Float64
    y::Any
end

"""Maximum-overlap injection of k lower labels into k+1 upper labels.
Returns the upper-label -> aligned-common-label map and agreement count.
The subset dynamic program breaks ties deterministically.
"""
function alignment(lower,upper)
    n=length(lower);length(upper)==n || throw(ArgumentError("reference lengths differ"))
    k=maximum(lower);maximum(upper)==k+1 || throw(ArgumentError("reference counts must differ by one"))
    sort(unique(lower))==collect(1:k) && sort(unique(upper))==collect(1:k+1) || throw(ArgumentError("references must be canonical"))
    k<=20 || throw(ArgumentError("alignment dimension exceeds implementation limit"))
    counts=zeros(Int,k,k+1)
    for i in eachindex(lower);counts[lower[i],upper[i]]+=1;end
    full=(1<<(k+1))-1;dp=fill(typemin(Int),full+1);previous=zeros(Int,full+1);column=zeros(Int,full+1);dp[1]=0
    for mask in 0:full
        row=count_ones(mask)+1;row>k && continue
        dp[mask+1]==typemin(Int) && continue
        for h in 1:k+1
            bit=1<<(h-1);mask&bit==0 || continue
            next=mask|bit;score=dp[mask+1]+counts[row,h]
            if score>dp[next+1]
                dp[next+1]=score;previous[next+1]=mask;column[next+1]=h
            end
        end
    end
    eligible=[mask for mask in 0:full if count_ones(mask)==k]
    mask=eligible[argmax([dp[mask+1] for mask in eligible])];score=dp[mask+1]
    mapping=fill(k+1,k+1)
    for row in k:-1:1
        mapping[column[mask+1]]=row;mask=previous[mask+1]
    end
    mapping,score
end

function Workspace(m;references,lower=6,slots=20,mask_probability=1.,pair_indices=nothing)
    k=Int(lower);L=Int(slots);p=Float64(mask_probability)
    1<=k< L && k<=20 || throw(ArgumentError("invalid common-label/slot count"))
    isfinite(p) && 0<=p<=1 || throw(ArgumentError("invalid mask probability"))
    isempty(references) && throw(ArgumentError("no frozen references"))
    all(c->length(c)==length(m.y) && all(>(0),c),references) || throw(ArgumentError("invalid reference labels"))
    refs=[G.canonical_labels(Int.(copy(c))) for c in references]
    lo=findall(c->maximum(c)==k,refs);hi=findall(c->maximum(c)==k+1,refs)
    isempty(lo) || isempty(hi) ? throw(ArgumentError("missing adjacent count reference class")) : nothing
    candidates=pair_indices===nothing ? [(a,b) for a in lo for b in hi] : collect(pair_indices)
    isempty(candidates) && throw(ArgumentError("empty static reference-pair bank"))
    all(ab->ab[1] in lo && ab[2] in hi,candidates) || throw(ArgumentError("invalid reference-pair indices"))
    white=[collect(LowerTriangular(m.omegaL)\y) for y in m.y]
    pairs=ReferencePair[]
    for (a,b) in candidates
        mapping,agreement=alignment(refs[a],refs[b]);anchors=Int[]
        for h in 1:k
            common=[i for i in eachindex(m.y) if refs[a][i]==h && mapping[refs[b][i]]==h]
            isempty(common) && break
            center=mean(white[common])
            push!(anchors,common[argmin([sum(abs2,white[i]-center) for i in common])])
        end
        # Exclusion is static and merely removes unusable reference pairs.
        length(anchors)==k && push!(pairs,ReferencePair(a,b,mapping,anchors,agreement))
    end
    isempty(pairs) && throw(ArgumentError("no reference pair has all common anchors"))
    Workspace(refs,pairs,k,L,p,m.y)
end
Workspace(s::MF.State;kwargs...)=Workspace(s.model;kwargs...)

"""Observation-wise transpositions, an involution conditional on its inputs.
The k common labels are read only at fixed anchor observations; these anchors
have equal reference labels and therefore remain fixed by the transform.
"""
function transform(w::Workspace,pair::ReferencePair,labels,extra,mask=nothing)
    length(labels)==length(w.y) || throw(ArgumentError("allocation length differs"))
    mask===nothing || length(mask)==length(labels) || throw(ArgumentError("mask length differs"))
    values=labels[pair.anchors]
    length(unique(values))==w.lower || throw(ArgumentError("common anchors are not distinct"))
    1<=extra<=w.slots && !(extra in values) || throw(ArgumentError("invalid extra slot"))
    slots=vcat(values,extra);lo=w.references[pair.lower_reference];hi=w.references[pair.upper_reference]
    output=copy(labels)
    for i in eachindex(labels)
        mask===nothing || mask[i] || continue
        a=slots[lo[i]];b=slots[pair.upper_to_common[hi[i]]]
        output[i]=labels[i]==a ? b : labels[i]==b ? a : labels[i]
    end
    output
end

mutable struct Proposal
    kind::Symbol
    original_c::Vector{Int}
    original_M::Int
    original_gamma::Float64
    original_lambda::Float64
    pair_index::Int
    injection::Vector{Int}
    extra_slot::Int
    mask::Union{Nothing,Vector{Bool}}
    labelled_old::Vector{Int}
    labelled_new::Vector{Int}
    labels::Vector{Int}
    oldK::Int
    newK::Int
    newM::Int
    old_logtarget::Float64
    new_logtarget::Float64
    selection_correction::Float64
    ratio::Float64
    logu::Float64
    moved_observations::Int
    outcome::Symbol
end

function nullproposal(s,pair,logu,outcome)
    c=G.canonical_labels(s.c);K=maximum(c)
    Proposal(:reference_transport,copy(s.c),s.M,s.gamma,s.lambda,pair,Int[],0,nothing,
        Int[],Int[],c,K,K,s.M,NaN,NaN,0.,-Inf,logu,0,outcome)
end

"""Augment with a uniform K->L injection, then apply a fixed-reference
involution. Gamma, Lambda and M-K remain unchanged. Only external RNG is used.
The MH slot correction is log(L)_Kold - log(L)_Knew.
"""
function propose!(w::Workspace,s::MF.State,rng)
    rng===s.rng && throw(ArgumentError("reference transport requires an external RNG"))
    w.y===s.model.y || w.y==s.model.y || throw(ArgumentError("reference data differ"))
    old=G.canonical_labels(s.c);K=maximum(old);K<=s.M || error("invalid occupied count")
    j=rand(rng,eachindex(w.pairs));logu=log(rand(rng))
    K>w.slots && return nullproposal(s,j,logu,:outside_proposal_slots)
    pair=w.pairs[j];injection=shuffle(rng,collect(1:w.slots))[1:K]
    labelled=injection[old];values=labelled[pair.anchors]
    length(unique(values))==w.lower || return nullproposal(s,j,logu,:anchors_not_distinct)
    extra=rand(rng,setdiff(1:w.slots,values))
    mask=w.mask_probability==1. ? nothing : [rand(rng)<w.mask_probability for _ in old]
    newlabelled=transform(w,pair,labelled,extra,mask);labels=G.canonical_labels(newlabelled)
    newK=maximum(labels);newM=s.M+newK-K
    oldtarget=MT.logtarget(s)
    groups=[findall(==(h),labels) for h in 1:newK]
    newtarget=MT.logtarget(s;groups,M=newM)
    correction=MT.logfallingfactorial(w.slots,K)-MT.logfallingfactorial(w.slots,newK)
    ratio=newtarget-oldtarget+correction
    isfinite(ratio) || error("nonfinite reference-transport ratio")
    outcome=logu<min(0.,ratio) ? :accepted : :mh_rejected
    # Label-permutation-invariant moved count: maximum-overlap matching is
    # unnecessary for decisions, so report only the changed coallocation set's
    # directly transported observations, in the augmented labelling.
    moved=count(a!=b for (a,b) in zip(labelled,newlabelled))
    Proposal(:reference_transport,copy(s.c),s.M,s.gamma,s.lambda,j,injection,extra,mask,
        labelled,newlabelled,labels,K,newK,newM,oldtarget,newtarget,correction,ratio,logu,moved,outcome)
end

function commit!(s::MF.State,p::Proposal)
    p.outcome==:accepted || return s
    s.c==p.original_c && s.M==p.original_M && s.gamma==p.original_gamma && s.lambda==p.original_lambda || error("state changed before commit")
    labels=p.labels;K=p.newK
    length(labels)==length(s.c) && labels==G.canonical_labels(labels) && maximum(labels)==K && p.newM>=K || throw(ArgumentError("invalid transport endpoint"))
    clusters=[G.Cluster(s.model,zero(s.mu0)) for _ in 1:K]
    for i in eachindex(labels);G.add!(clusters[labels[i]],s.model.y[i]);end
    cached=MF.State(s.model,clusters,copy(labels),collect(1:K),s.mu0,s.kappa,
        [s.model.omegaL for _ in clusters],[s.mu0 for _ in clusters],zeros(K),p.newM,s.gamma,s.lambda,
        s.rng,s.gamma_attempts,s.gamma_accepts)
    for h in 1:K;MF.refresh!(cached,h);end
    s.clusters=cached.clusters;s.c=cached.c;s.active=cached.active
    s.predL=cached.predL;s.predmu=cached.predmu;s.prednorm=cached.prednorm;s.M=p.newM
    s
end
function attempt!(w::Workspace,s::MF.State,rng;observer=nothing)
    p=propose!(w,s,rng);observer===nothing || observer(s,p);commit!(s,p);p
end

end
