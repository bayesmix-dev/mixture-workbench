module MFMReferenceIndependence

using GeneralDPPMixtures, Random, LinearAlgebra, Statistics, Distributions, SpecialFunctions
import ..MFMBaseline as MF
import ..MFMTarget as MT
const G=GeneralDPPMixtures

export Workspace, NIWParameters, Component, Auxiliary, Proposal, prepare,
       niw_parameters, draw_component, logniw, logpermanent, augment,
       draw_proposal, logjoint, collapsed_logjoint, logproposal, logimportance,
       propose!, attempt!, commit!, supportmetadata

struct NIWParameters
    mu::Vector{Float64}
    kappa::Float64
    nu::Float64
    omega::Matrix{Float64}
end

struct Component
    mu::Vector{Float64}
    sigma::Matrix{Float64}
    L::Matrix{Float64}
    lognormalizer::Float64
end
function Component(mu,sigma)
    v=Float64.(collect(mu)); S=Matrix{Float64}(Symmetric(sigma))
    size(S)==(length(v),length(v)) || throw(ArgumentError("component dimensions differ"))
    L=Matrix(cholesky(Symmetric(S)).L)
    Component(v,S,L,-length(v)/2*log(2pi)-sum(log,diag(L)))
end
function lognormal(c::Component,y)
    z=LowerTriangular(c.L)\(y-c.mu)
    c.lognormalizer-sum(abs2,z)/2
end

struct Auxiliary
    M::Int
    gamma::Float64
    lambda::Float64
    logw::Vector{Float64}
    theta::Vector{Component}
    c::Vector{Int}
end

struct ReferenceComponent
    reference::Int
    M::Int
    logweight::Float64
    groups::Vector{Vector{Int}}
    counts::Vector{Int}
    parameters::Union{Nothing,Vector{NIWParameters}}
end

"""
Frozen reference mixture. The finite M support limits this proposal only; the
MFM target retains unbounded M and must be explored by the other exact kernels.
Use Workspace(s;references,...) to cache immutable NIW parameters once. The
Workspace(references,discount) constructor prepares a temporary immutable cache
on each call. No workspace cache is mutated, so a prepared workspace is safe to
share among independent replica updates for the same fixed MFM prior/data.
"""
struct Workspace
    references::Vector{Vector{Int}}
    discount::Float64
    extra_empty::Bool
    gamma_shape::Float64
    gamma_scale::Float64
    max_components::Int
    components::Vector{ReferenceComponent}
    prior::Union{Nothing,NIWParameters}
    y::Any
end
function Workspace(references::AbstractVector,discount::Real=.25;
        extra_empty=true,gamma_shape=2.,gamma_scale=1.,max_components=20,weights=nothing)
    z=Float64(discount)
    isfinite(z) && z>=0 || throw(ArgumentError("discount must be finite and nonnegative"))
    isfinite(gamma_shape) && gamma_shape>0 && isfinite(gamma_scale) && gamma_scale>0 || throw(ArgumentError("invalid gamma proposal"))
    1<=max_components<=24 || throw(ArgumentError("permanent implementation supports at most 24 components"))
    isempty(references) && throw(ArgumentError("at least one frozen reference is required"))
    refs=[G.canonical_labels(Int.(copy(c))) for c in references]
    n=length(first(refs));n>0 && all(c->length(c)==n && all(>(0),c),references) || throw(ArgumentError("invalid references"))
    ww=weights===nothing ? ones(length(refs)) : Float64.(copy(weights))
    length(ww)==length(refs) && all(x->isfinite(x)&&x>0,ww) || throw(ArgumentError("invalid reference weights"))
    logww=log.(ww);logww.-=logsum(logww)
    components=ReferenceComponent[]
    for (j,c) in enumerate(refs)
        k=maximum(c);groups=[findall(==(h),c) for h in 1:k]
        for empty in (extra_empty ? (0,1) : (0,))
            M=k+empty;M<=max_components || throw(ArgumentError("reference M=$M exceeds this proposal's permanent limit"))
            gg=empty==0 ? deepcopy(groups) : vcat(deepcopy(groups),[Int[]])
            push!(components,ReferenceComponent(j,M,logww[j]-log(extra_empty ? 2. : 1.),gg,length.(gg),nothing))
        end
    end
    Workspace(refs,z,Bool(extra_empty),Float64(gamma_shape),Float64(gamma_scale),Int(max_components),components,nothing,nothing)
end
Workspace(s::MF.State;references,discount=.25,kwargs...)=prepare(Workspace(references,discount;kwargs...),s)

function niw_parameters(s::MF.State,members;discount=1.)
    z=Float64(discount)
    isfinite(z) && z>=0 || throw(ArgumentError("invalid NIW discount"))
    prior=NIWParameters(collect(s.mu0),s.kappa,s.model.tau,Matrix(s.model.omega))
    isempty(members) && return prior
    c=G.Cluster(s.model,zero(s.mu0))
    for i in sort(members);G.add!(c,s.model.y[i]);end
    zn=z*c.n;k=s.kappa+zn;delta=c.mean-s.mu0
    omega=s.model.omega+z*c.scatter+(s.kappa*zn/k)*(delta*delta')
    NIWParameters(collect((s.kappa*s.mu0+zn*c.mean)/k),k,s.model.tau+zn,Matrix(Symmetric(omega)))
end
function compatible(w,s)
    p=w.prior
    (w.y===s.model.y || w.y==s.model.y) && p.mu==collect(s.mu0) && p.kappa==s.kappa &&
        p.nu==s.model.tau && p.omega==s.model.omega
end
function prepare(w::Workspace,s::MF.State)
    length(first(w.references))==length(s.c) || throw(ArgumentError("reference observation count differs"))
    if w.prior!==nothing
        compatible(w,s) || throw(ArgumentError("prepared workspace has different MFM prior/data"))
        return w
    end
    prior=niw_parameters(s,Int[])
    cc=[ReferenceComponent(c.reference,c.M,c.logweight,c.groups,c.counts,
        [niw_parameters(s,g;discount=w.discount) for g in c.groups]) for c in w.components]
    Workspace(w.references,w.discount,w.extra_empty,w.gamma_shape,w.gamma_scale,w.max_components,cc,prior,s.model.y)
end

function draw_component(rng,p::NIWParameters)
    sigma=rand(rng,InverseWishart(p.nu,p.omega))
    mu=rand(rng,MvNormal(p.mu,Symmetric(sigma/p.kappa)))
    Component(mu,sigma)
end
logniw(p::NIWParameters,c::Component)=logpdf(InverseWishart(p.nu,p.omega),c.sigma)+
    logpdf(MvNormal(p.mu,Symmetric(c.sigma/p.kappa)),c.mu)

@inline function logadd(a,b)
    a==-Inf && return b
    b==-Inf && return a
    a>b ? a+log1p(exp(b-a)) : b+log1p(exp(a-b))
end
function logsum(values)
    isempty(values) && return -Inf
    m=maximum(values);m==-Inf && return m
    m+log(sum(exp(v-m) for v in values))
end

"""Log permanent by subset DP, including repeated identical component rows."""
function logpermanent(A::AbstractMatrix)
    n,m=size(A);n==m || throw(ArgumentError("permanent requires a square matrix"))
    n<=24 || throw(ArgumentError("permanent dimension exceeds implementation limit"))
    all(x->isfinite(x)||x==-Inf,A) || throw(ArgumentError("invalid log-permanent entry"))
    n==0 && return 0.
    full=(1<<n)-1;dp=fill(-Inf,full+1);dp[1]=0.
    for mask in 0:full-1
        value=dp[mask+1];value==-Inf && continue
        row=count_ones(mask)+1
        for h in 1:n
            bit=1<<(h-1);mask&bit==0 || continue
            next=mask|bit
            dp[next+1]=logadd(dp[next+1],value+A[row,h])
        end
    end
    last(dp)
end

function logdirichlet(logw,alpha)
    length(logw)==length(alpha) || throw(ArgumentError("Dirichlet dimensions differ"))
    loggamma(sum(alpha))-sum(loggamma,alpha)+sum((alpha[h]-1)*logw[h] for h in eachindex(alpha))
end
function draw_logdirichlet(rng,alpha)
    lw=[G.loggamma_draw(rng,a) for a in alpha]
    lw.-logsum(lw)
end
function validate(s,a::Auxiliary)
    a.M>=1 && isfinite(a.gamma) && a.gamma>0 && isfinite(a.lambda) && a.lambda>0 || throw(ArgumentError("invalid auxiliary hyperparameters"))
    length(a.logw)==length(a.theta)==a.M && all(isfinite,a.logw) && abs(logsum(a.logw))<1e-9 || throw(ArgumentError("invalid log weights/components"))
    length(a.c)==length(s.c) && all(h->1<=h<=a.M,a.c) || throw(ArgumentError("invalid labelled allocations"))
    all(c->length(c.mu)==length(s.mu0),a.theta) || throw(ArgumentError("auxiliary dimensions differ"))
    a
end

"""Exact old-state augmentation, using ONLY the external RNG."""
function augment(s::MF.State,rng)
    rng===s.rng && throw(ArgumentError("augmentation requires an external proposal RNG"))
    canonical=G.canonical_labels(s.c);K=maximum(canonical);K<=s.M || error("invalid occupied count")
    injection=shuffle(rng,collect(1:s.M))[1:K]
    labels=injection[canonical];groups=[findall(==(h),labels) for h in 1:s.M]
    logw=draw_logdirichlet(rng,s.gamma.+length.(groups))
    theta=[draw_component(rng,niw_parameters(s,g)) for g in groups]
    Auxiliary(s.M,s.gamma,s.lambda,logw,theta,labels)
end

function allocation_terms(s,a::Auxiliary)
    ll=0.;lp=0.;scores=zeros(a.M)
    for i in eachindex(a.c)
        for h in 1:a.M;scores[h]=a.logw[h]+lognormal(a.theta[h],s.model.y[i]);end
        normalizer=logsum(scores)
        ll+=normalizer;lp+=scores[a.c[i]]-normalizer
    end
    (mixture=ll,allocation=lp)
end
function sample_allocations(rng,s,logw,theta)
    M=length(logw);labels=Vector{Int}(undef,length(s.c));scores=zeros(M)
    for i in eachindex(labels)
        for h in 1:M;scores[h]=logw[h]+lognormal(theta[h],s.model.y[i]);end
        mx=maximum(scores);labels[i]=G.draw_weights(rng,exp.(scores.-mx))
    end
    labels
end
function draw_proposal(w::Workspace,s::MF.State,rng)
    rng===s.rng && throw(ArgumentError("independence proposal requires an external proposal RNG"))
    w=prepare(w,s)
    j=G.draw_weights(rng,exp.([c.logweight for c in w.components]));component=w.components[j]
    gamma=rand(rng,Gamma(w.gamma_shape,w.gamma_scale));lambda=rand(rng,Gamma(component.M,1/1.1))
    # p maps each ordered reference component to its labelled destination.
    p=shuffle(rng,collect(1:component.M));theta=Vector{Component}(undef,component.M)
    counts=zeros(Int,component.M)
    for h in 1:component.M
        theta[p[h]]=draw_component(rng,component.parameters[h]);counts[p[h]]=component.counts[h]
    end
    logw=draw_logdirichlet(rng,gamma.+w.discount.*counts)
    labels=sample_allocations(rng,s,logw,theta)
    Auxiliary(component.M,gamma,lambda,logw,theta,labels),j
end

function reference_logmixture(w,s,a)
    terms=Float64[]
    for c in w.components
        c.M==a.M || continue
        alpha=a.gamma.+w.discount.*c.counts
        A=[logniw(c.parameters[j],a.theta[h])+(alpha[j]-1)*a.logw[h] for j in 1:a.M,h in 1:a.M]
        normalizer=loggamma(sum(alpha))-sum(loggamma,alpha)
        push!(terms,c.logweight+normalizer+logpermanent(A)-loggamma(a.M+1.))
    end
    logsum(terms)
end
function proposal_terms(w::Workspace,s,a::Auxiliary)
    w=prepare(w,s);validate(s,a)
    ref=reference_logmixture(w,s,a)
    alloc=allocation_terms(s,a)
    gamma=logpdf(Gamma(w.gamma_shape,w.gamma_scale),a.gamma)
    lambda=logpdf(Gamma(a.M,1/1.1),a.lambda)
    (reference=ref,allocation=alloc.allocation,mixture=alloc.mixture,gamma=gamma,lambda=lambda,
        total=ref+alloc.allocation+gamma+lambda)
end
logproposal(w::Workspace,s::MF.State,a::Auxiliary)=proposal_terms(w,s,a).total

"""The complete labelled target, with every NIW/Dirichlet/hyperprior constant."""
function logjoint(s::MF.State,a::Auxiliary)
    validate(s,a);prior=niw_parameters(s,Int[])
    hyper=logpdf(Poisson(a.lambda),a.M-1)+logpdf(Exponential(10.),a.gamma)+logpdf(Exponential(10.),a.lambda)
    hyper+logdirichlet(a.logw,fill(a.gamma,a.M))+sum(logniw(prior,c) for c in a.theta)+
        sum(a.logw[a.c[i]]+lognormal(a.theta[a.c[i]],s.model.y[i]) for i in eachindex(a.c))
end

"""Independent collapsed-target times exact old-augmentation density identity."""
function collapsed_logjoint(s::MF.State,a::Auxiliary)
    validate(s,a);groups=[findall(==(h),a.c) for h in 1:a.M];occupied=filter(!isempty,groups)
    MT.logtarget(s;groups=occupied,M=a.M,gamma=a.gamma,lambda=a.lambda)-MT.logfallingfactorial(a.M,length(occupied))+
        logdirichlet(a.logw,a.gamma.+length.(groups))+
        sum(logniw(niw_parameters(s,groups[h]),a.theta[h]) for h in 1:a.M)
end

# Allocation categorical probabilities cancel exactly. Lambda also cancels
# against its exact conditional proposal, leaving the shifted-geometric prior
# p(M)=.1/1.1^M obtained by integrating Lambda. This avoids subtracting two
# enormous allocation log densities in the acceptance ratio.
function importance(w,s,a,terms)
    w=prepare(w,s)
    terms.reference==-Inf && return Inf
    prior=w.prior
    log(.1)-a.M*log(1.1)+logpdf(Exponential(10.),a.gamma)-terms.gamma+
        logdirichlet(a.logw,fill(a.gamma,a.M))+sum(logniw(prior,c) for c in a.theta)+
        terms.mixture-terms.reference
end
logimportance(w::Workspace,s::MF.State,a::Auxiliary)=importance(w,s,a,proposal_terms(w,s,a))

mutable struct Proposal
    kind::Symbol
    original_c::Vector{Int}
    original_M::Int
    original_gamma::Float64
    original_lambda::Float64
    component::Union{Nothing,Int}
    old::Union{Nothing,Auxiliary}
    new::Union{Nothing,Auxiliary}
    old_logtarget::Float64
    new_logtarget::Float64
    old_logq::Float64
    new_logq::Float64
    old_logweight::Float64
    new_logweight::Float64
    ratio::Float64
    logu::Union{Nothing,Float64}
    outcome::Symbol
end
function propose!(w::Workspace,s::MF.State,rng)
    rng===s.rng && throw(ArgumentError("independence kernel requires an external proposal RNG"))
    support=unique(c.M for c in w.components)
    if !(s.M in support)
        logu=log(rand(rng))
        return Proposal(:reference_independence,copy(s.c),s.M,s.gamma,s.lambda,nothing,nothing,nothing,
            NaN,NaN,-Inf,NaN,Inf,NaN,-Inf,logu,:unsupported_old_M)
    end
    w=prepare(w,s);old=augment(s,rng);new,j=draw_proposal(w,s,rng)
    oldterms=proposal_terms(w,s,old);newterms=proposal_terms(w,s,new)
    oldweight=importance(w,s,old,oldterms);newweight=importance(w,s,new,newterms)
    ratio=newweight-oldweight
    isfinite(ratio) || error("nonfinite auxiliary independence ratio")
    logu=log(rand(rng));outcome=logu<min(0.,ratio) ? :accepted : :mh_rejected
    Proposal(:reference_independence,copy(s.c),s.M,s.gamma,s.lambda,j,old,new,
        logjoint(s,old),logjoint(s,new),oldterms.total,newterms.total,oldweight,newweight,ratio,logu,outcome)
end

"""Rebuild all collapsed caches before atomically changing the original state."""
function commit!(s::MF.State,p::Proposal)
    p.outcome==:accepted || return s
    s.c==p.original_c && s.M==p.original_M && s.gamma==p.original_gamma && s.lambda==p.original_lambda || error("state changed before commit")
    a=p.new;validate(s,a);labels=G.canonical_labels(a.c);K=maximum(labels)
    clusters=[G.Cluster(s.model,zero(s.mu0)) for _ in 1:K]
    for i in eachindex(labels);G.add!(clusters[labels[i]],s.model.y[i]);end
    cached=MF.State(s.model,clusters,labels,collect(1:K),s.mu0,s.kappa,
        [s.model.omegaL for _ in clusters],[s.mu0 for _ in clusters],zeros(K),a.M,a.gamma,a.lambda,
        s.rng,s.gamma_attempts,s.gamma_accepts)
    for h in 1:K;MF.refresh!(cached,h);end
    s.clusters=cached.clusters;s.c=cached.c;s.active=cached.active
    s.predL=cached.predL;s.predmu=cached.predmu;s.prednorm=cached.prednorm
    s.M=a.M;s.gamma=a.gamma;s.lambda=a.lambda
    s
end
function attempt!(w::Workspace,s::MF.State,rng;observer=nothing)
    p=propose!(w,s,rng)
    observer===nothing || observer(s,p)
    commit!(s,p);p
end

function supportmetadata(w::Workspace)
    Dict("proposal_support_M"=>sort(unique(c.M for c in w.components)),"reference_partitions"=>length(w.references),
        "discount"=>w.discount,"extra_empty_component"=>w.extra_empty,
        "gamma_proposal_shape"=>w.gamma_shape,"gamma_proposal_scale"=>w.gamma_scale,
        "Lambda_proposal"=>"Gamma(M, scale=1/1.1)","permanent_dimension_limit"=>w.max_components,
        "target_support"=>"M remains unbounded; compose with the original MFM kernels",
        "target"=>"original beta=1 collapsed MFM via exact labelled NIW/weight augmentation")
end

end
