module PointProcessSampler
using GeneralDPPMixtures, LinearAlgebra, Random, Distributions, StaticArrays
const G=GeneralDPPMixtures

include("../../priors/point_process/configurations.jl")

struct Geometry{X}
    xs::Vector{X}
    F::Matrix{Float64}          # dimensionless weighted Fourier features
    K::Matrix{Float64}
    lp::Float64                # log density wrt unit Poisson, up to a constant
end
function logdetK(K,F,xs)
    isempty(K) && return 0.0
    length(xs)>size(F,1) && return -Inf
    for i in eachindex(xs), j in 1:i-1
        xs[i]==xs[j] && return -Inf
    end
    C=cholesky(Symmetric(K);check=false)
    if issuccess(C) && minimum(abs2,diag(C.L))>1e-10*maximum(diag(K))
        return 2sum(log,diag(C.L))
    end
    # Recompute from features when Gram formation loses accuracy. No jitter.
    G.logdet_features(F,G.Counters())
end
function geometry(p::DPPPrior,m,xs)
    F=G.feature_matrix(m,p.modes,xs).*p.roots
    K=F'F
    Geometry(xs,F,K,logdetK(K,F,xs)-length(xs)*m.logvolume)
end
function geometry(p::StraussPrior,m,xs)
    pairs=sum((sum(abs2,xs[i]-xs[j])<=p.radius^2 for i in eachindex(xs) for j in 1:i-1);init=0)
    Geometry(xs,zeros(0,0),zeros(0,0),length(xs)*(log(p.beta)-m.logvolume)+pairs*p.logalpha)
end
function edit(p::DPPPrior,m,g,x,j)
    # j=0: birth; x=nothing: deletion; otherwise a location replacement.
    if x===nothing
        keep=[i for i in eachindex(g.xs) if i!=j]
        K=g.K[keep,keep]; xs=g.xs[keep]
        F=g.F[:,keep]
        return Geometry(xs,F,K,logdetK(K,F,xs)-length(xs)*m.logvolume)
    end
    f=zeros(length(p.modes)); G.features!(f,m,p.modes,x); f .*= p.roots
    xs=copy(g.xs)
    if j==0
        push!(xs,x); F=hcat(g.F,f); M=length(xs); K=zeros(M,M)
        K[1:M-1,1:M-1]=g.K; v=g.F'f
        K[1:M-1,M]=v; K[M,1:M-1]=v; K[M,M]=dot(f,f)
    else
        xs[j]=x; F=copy(g.F); F[:,j]=f; K=copy(g.K)
        v=F'f; K[:,j]=v; K[j,:]=v
    end
    Geometry(xs,F,K,logdetK(K,F,xs)-length(xs)*m.logvolume)
end
function edit(p::StraussPrior,m,g,x,j)
    xs=copy(g.xs)
    x===nothing ? deleteat!(xs,j) : j==0 ? push!(xs,x) : (xs[j]=x)
    geometry(p,m,xs)
end

mutable struct State{M,P,X,L,C}
    model::M
    prior::P
    geom::Geometry{X}
    c::Vector{Int}
    clusters::Vector{C}
    covs::Vector{L}
    logjumps::Vector{Float64}
    logu::Float64
    rng::Xoshiro
    attempts::Vector{Int}       # birth, death, location
    accepts::Vector{Int}
end
function initialize(m,init,p,seed)
    rng=Xoshiro(seed); cl=[G.Cluster(m,x) for x in init.x]
    for i in eachindex(init.c); G.add!(cl[init.c[i]],m.y[i]); end
    for c in cl; G.refresh!(c,m); end
    covs=[G.iw_factor(rng,m.tau+c.n,c.scaleL) for c in cl]
    State(m,p,geometry(p,m,copy(init.x)),copy(init.c),cl,covs,
        zeros(length(cl)),log(length(m.y)/(length(cl)*m.a)),rng,zeros(Int,3),zeros(Int,3))
end
log1pexp(x)=max(x,0)+log1p(exp(-abs(x)))
function logsumexp(x)
    a=maximum(x); a+log(sum(v->exp(v-a),x))
end
"A(i), Eq. (19): invariant birth/death MH steps on the unused configuration."
function unused!(s;steps=10)
    m=s.model; rng=s.rng; logpsi=-m.a*log1pexp(s.logu)
    for _ in 1:steps
        unused=findall(c->c.n==0,s.clusters); ell=length(unused)
        if rand(rng)<0.5
            s.attempts[1]+=1; x=G.uniform_point(rng,m)
            g=edit(s.prior,m,s.geom,x,0)
            lr=g.lp-s.geom.lp+logpsi+m.logvolume-log(ell+1)
            if log(rand(rng))<lr
                s.geom=g; push!(s.clusters,G.Cluster(m,x)); s.accepts[1]+=1
            end
        elseif ell>0
            s.attempts[2]+=1; j=rand(rng,unused)
            g=edit(s.prior,m,s.geom,nothing,j)
            lr=g.lp-s.geom.lp-logpsi-m.logvolume+log(ell)
            if log(rand(rng))<lr
                s.geom=g; deleteat!(s.clusters,j); s.c .-= (s.c .> j); s.accepts[2]+=1
            end
        end # a death at ell=0 is a null move, retaining 1/2 proposal probabilities
    end
end
function sweep!(s;bd_steps=10)
    m=s.model; rng=s.rng; unused!(s;steps=bd_steps)
    M=length(s.clusters); resize!(s.covs,M); resize!(s.logjumps,M)
    rate=log1pexp(s.logu)
    # A(ii-iii), B: marks are reconstructed after the collapsed dimension update.
    # Covariances are refreshed conditional on the old location; then location MH.
    for j in 1:M
        c=s.clusters[j]; G.refresh!(c,m)
        L=G.iw_factor(rng,m.tau+c.n,c.scaleL); s.covs[j]=L
        if c.n>0
            x=G.mvnormal_draw(rng,c.mean,L/sqrt(c.n)); s.attempts[3]+=1
            if G.inside(m,x)
                g=edit(s.prior,m,s.geom,x,j)
                # Independence proposal equals the normalized Gaussian likelihood.
                if log(rand(rng))<g.lp-s.geom.lp
                    s.geom=g; c.x=x; s.accepts[3]+=1
                end
            end
        end
        s.logjumps[j]=G.loggamma_draw(rng,m.a+c.n)-rate
    end
    # C: simultaneous conditionally independent allocations; rebuild stable stats.
    scores=zeros(M); D=length(first(m.y))
    norms=[s.logjumps[j]-D/2*log(2pi)-sum(log,diag(s.covs[j])) for j in 1:M]
    for c in s.clusters; c.n=0; c.mean=zero(c.mean); c.scatter=zero(c.scatter); end
    for i in eachindex(s.c)
        for j in 1:M
            z=LowerTriangular(s.covs[j])\(m.y[i]-s.geom.xs[j])
            scores[j]=norms[j]-sum(abs2,z)/2
        end
        j=G.draw_logweights!(rng,scores); s.c[i]=j; G.add!(s.clusters[j],m.y[i])
    end
    # D is absent for fixed point-process parameters; E is exact Gamma(n,total jumps).
    s.logu=G.loggamma_draw(rng,length(s.c))-logsumexp(s.logjumps)
    s
end
end
