module MFMBaseline
using GeneralDPPMixtures, LinearAlgebra, Random, Distributions, StaticArrays, SpecialFunctions
const G=GeneralDPPMixtures
mutable struct State{M,C,X,L}
    model::M
    clusters::Vector{C}
    c::Vector{Int}
    active::Vector{Int}
    mu0::X
    kappa::Float64
    predL::Vector{L}
    predmu::Vector{X}
    prednorm::Vector{Float64}
    M::Int
    gamma::Float64
    lambda::Float64
    rng::Xoshiro
    gamma_attempts::Int
    gamma_accepts::Int
end
function initialize(m,init,seed)
    cl=[G.Cluster(m,x) for x in init.x]
    for i in eachindex(init.c); G.add!(cl[init.c[i]],m.y[i]); end
    mu0=sum(m.y)/length(m.y)
    s=State(m,cl,copy(init.c),collect(eachindex(cl)),mu0,0.1,
        [m.omegaL for _ in cl],[mu0 for _ in cl],zeros(length(cl)),length(cl),0.1,1.0,Xoshiro(seed),0,0)
    for h in eachindex(cl); refresh!(s,h); end
    s
end
function refresh!(s,h)
    c=s.clusters[h]; m=s.model; d=length(s.mu0); n=c.n
    kap=s.kappa+n; df=m.tau+n-d+1
    delta=c.mean-s.mu0
    scale=m.omega+c.scatter+(s.kappa*n/kap)*(delta*delta')
    # LL' = df * Student scale, as in G.student_logdensity.
    L=cholesky(Symmetric(scale*((kap+1)/kap))).L
    s.predL[h]=L; s.predmu[h]=(s.kappa*s.mu0+n*c.mean)/kap
    s.prednorm[h]=loggamma((df+d)/2)-loggamma(df/2)-d/2*log(pi)-sum(log,diag(L))
end
function logpred(s,h,y)
    c=s.clusters[h]; z=LowerTriangular(s.predL[h])\(y-s.predmu[h])
    s.prednorm[h]-(s.model.tau+c.n+1)/2*log1p(sum(abs2,z))
end
function empty_slot!(s)
    h=findfirst(c->c.n==0,s.clusters)
    if h===nothing
        push!(s.clusters,G.Cluster(s.model,zero(s.mu0)))
        push!(s.predL,s.model.omegaL); push!(s.predmu,s.mu0); push!(s.prednorm,0.)
        h=length(s.clusters)
    end
    refresh!(s,h); h
end
function gamma_logtarget(s,g)
    g>0 || return -Inf
    # Dirichlet allocation density and Gamma(1, rate=.1) prior.
    loggamma(s.M*g)-loggamma(s.M*g+length(s.c))+
        sum(loggamma(s.clusters[h].n+g)-loggamma(g) for h in s.active)-0.1g
end
function sweep!(s)
    rng=s.rng; scores=Float64[]
    for i in eachindex(s.c)
        old=s.c[i]; G.remove!(s.clusters[old],s.model.y[i]); refresh!(s,old)
        s.clusters[old].n==0 && deleteat!(s.active,findfirst(==(old),s.active))
        k=length(s.active); canbirth=k<s.M; resize!(scores,k+Int(canbirth))
        for (j,h) in enumerate(s.active)
            scores[j]=log(s.clusters[h].n+s.gamma)+logpred(s,h,s.model.y[i])
        end
        empty=canbirth ? empty_slot!(s) : 0
        canbirth && (scores[end]=log(s.M-k)+log(s.gamma)+logpred(s,empty,s.model.y[i]))
        j=G.draw_logweights!(rng,scores)
        h=j<=k ? s.active[j] : empty
        j>k && push!(s.active,h)
        s.c[i]=h; G.add!(s.clusters[h],s.model.y[i]); refresh!(s,h)
    end
    # M~1+Poisson(Lambda). Beta augmentation gives an exact, unbounded
    # mixture of Poisson(t) and 1+Poisson(t) for M-k, t=Lambda*z^gamma.
    logz=G.logbeta_draw(rng,s.M*s.gamma,length(s.c)); k=length(s.active)
    t=s.lambda*exp(s.gamma*logz)
    s.M=k+rand(rng,Poisson(t))+Int(rand(rng)<t/(k+t))
    s.lambda=rand(rng,Gamma(s.M,1/1.1))
    candidate=exp(log(s.gamma)+0.25randn(rng)); s.gamma_attempts+=1
    lr=gamma_logtarget(s,candidate)-gamma_logtarget(s,s.gamma)+log(candidate/s.gamma)
    if log(rand(rng))<lr; s.gamma=candidate; s.gamma_accepts+=1; end
    s
end
end
