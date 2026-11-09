using Test, GeneralDPPMixtures, Random, Distributions, Statistics, LinearAlgebra, StaticArrays, SpecialFunctions, QuadGK, Serialization
include("../src/Benchmark.jl")
const B=Benchmark
const G=GeneralDPPMixtures
const PP=B.PP
const MF=B.MF
BLAS.set_num_threads(1)

function fixture()
    sp=G.build_spectrum(Val(1),1.,.6,2.,.02)
    G.Model([SVector(-.4),SVector(.2),SVector(.7)],ones(Int,3),[-1.],[1.],sp;
        a=.5,tau=4.,omega=reshape([.7],1,1),seed=UInt64(12))
end
@testset "Same finite DPP L ensemble" begin
    m=fixture(); p=PP.DPPPrior(m); xs=[SVector(-.5),SVector(.35)]
    g=PP.geometry(p,m,xs)
    # Cauchy-Binet: explicitly sum all fixed-size spectral subsets.
    N=length(p.modes); mass=0.
    for i in 1:N, j in i+1:N
        F=G.feature_matrix(m,[i,j],xs)
        mass+=exp(m.spectrum.logodds[m.spectrum.shell[i]]+m.spectrum.logodds[m.spectrum.shell[j]])*det(F)^2/m.volume^2
    end
    @test exp(g.lp)≈mass rtol=1e-11
    for (x,j) in ((SVector(.8),0),(SVector(.1),1),(nothing,2))
        gp=PP.edit(p,m,g,x,j); exact=PP.geometry(p,m,gp.xs)
        @test gp.lp≈exact.lp atol=1e-10
        @test gp.K≈exact.K
    end
    @test PP.geometry(p,m,[xs[1],xs[1]]).lp==-Inf
end

@testset "Birth/death stationary count laws" begin
    m=fixture(); init=G.starting_configuration(m,1,1)
    for prior in (PP.DPPPrior(m), PP.StraussPrior(3.,.2,0.))
        s=PP.initialize(m,init,prior,UInt64(134))
        empty!(s.clusters); empty!(s.c); s.geom=PP.geometry(prior,m,typeof(m.y)())
        s.logu=log(2.); psi=3.0^(-m.a)
        expected=prior isa PP.DPPPrior ? sum(begin
            l=m.spectrum.lambda[h]; psi*l/(1-l+psi*l)
        end for h in m.spectrum.shell) : prior.beta*psi
        batches=Float64[]
        for b in 1:101
            total=0.
            for _ in 1:400; PP.unused!(s;steps=3); total+=length(s.clusters); end
            b>1 && push!(batches,total/400)
        end
        @test abs(mean(batches)-expected)<6std(batches)/sqrt(length(batches))+.005
    end
end

@testset "NIW predictive against package density" begin
    m=fixture(); init=G.starting_configuration(m,1,1); s=MF.initialize(m,init,UInt64(2))
    c=s.clusters[1]; kap=s.kappa+c.n; df=m.tau+c.n
    delta=c.mean-s.mu0
    scale=m.omega+c.scatter+s.kappa*c.n/kap*(delta*delta')
    pred=TDist(df)
    sigma=sqrt(scale[1,1]*(kap+1)/(kap*df)); mu=(s.kappa*s.mu0+c.n*c.mean)[1]/kap
    @test MF.logpred(s,1,SVector(.25))≈logpdf(pred,(.25-mu)/sigma)-log(sigma) atol=1e-12
end

@testset "General DPP posterior versus independent quadrature" begin
    # Integrate the full random-rank model, including both one-mode supports.
    sp=G.Spectrum{1}(reshape([0,1],1,2),UInt8[0,1],[1,2],[1:1,2:2],[0,1],
        [.5,.5],[0.,0.],1.,2.,.5,1e-8,1.,4/3,0.,log(.25))
    m=G.Model([SVector(-.82),SVector(.49),SVector(1.23)],ones(Int,3),[-1.1],[1.6],sp;
        a=.37,tau=3.,omega=reshape([.6],1,1))
    init=G.Initialization(ones(Int,3),[SVector(.2)],[1,2])
    s=PP.initialize(m,init,PP.DPPPrior(m),UInt64(64302))
    partitions=[[1,2,1],[1,2,2],[1,1,2],[1,1,1]]
    a=m.a; n=length(m.y); lo=m.lower[1]; hi=m.upper[1]; V=hi-lo
    f(x)=sqrt(2)*cospi(2*(x-lo)/V)
    loglik(ids,x)=-length(ids)/2*log(pi)+m.tau/2*log(.6)-
        (m.tau+length(ids))/2*log(.6+sum((m.y[i][1]-x)^2 for i in ids))+
        loggamma((m.tau+length(ids))/2)-loggamma(m.tau/2)
    c1=exp(loggamma(a)-loggamma(a+n)); c2=exp(loggamma(2a)-loggamma(2a+n))
    exact=[begin
        groups=[findall(==(h),c) for h in 1:maximum(c)]
        w=exp(sum(loggamma(a+length(g))-loggamma(a) for g in groups))
        if length(groups)==1
            w*(c1+c2)*quadgk(x->exp(loglik(groups[1],x))*(1+f(x)^2)/V,lo,hi;rtol=1e-9)[1]
        else
            w*c2*quadgk(x->quadgk(y->exp(loglik(groups[1],x)+loglik(groups[2],y))*(f(x)-f(y))^2/V^2,
                lo,hi;rtol=1e-8)[1],lo,hi;rtol=1e-8)[1]
        end
    end for c in partitions]
    exact./=sum(exact)
    batches=zeros(100,4)
    for batch in 1:120
        counts=zeros(4)
        for _ in 1:300
            PP.sweep!(s); c=G.canonical_labels(s.c)
            j=findfirst(==(c),partitions); @assert j!==nothing; counts[j]+=1
        end
        batch>20 && (batches[batch-20,:]=counts/300)
    end
    for j in 1:4
        @test abs(mean(batches[:,j])-exact[j])<6std(batches[:,j])/10+.003
    end
end

@testset "Unbounded shifted Poisson count update" begin
    rng=Xoshiro(176); k=3; n=20; gamma=.7; lambda=7.
    ms=collect(k:100)
    logs=[logpdf(Poisson(lambda),M-1)+loggamma(M+1)-loggamma(M-k+1)+
        loggamma(M*gamma)-loggamma(M*gamma+n) for M in ms]
    weights=exp.(logs.-maximum(logs)); weights./=sum(weights); exact=sum(ms.*weights)
    M=3; batches=Float64[]
    for b in 1:101
        total=0.
        for _ in 1:500
            z=G.logbeta_draw(rng,M*gamma,n); t=lambda*exp(gamma*z)
            M=k+rand(rng,Poisson(t))+Int(rand(rng)<t/(k+t)); total+=M
        end
        b>1 && push!(batches,total/500)
    end
    @test abs(mean(batches)-exact)<6std(batches)/sqrt(length(batches))+.005
end

@testset "All sampler invariants" begin
    y=[SVector(-1.2+.1i) for i in 0:9]
    append!(y,[SVector(.4+.1i) for i in 0:9])
    m=B.model(y,UInt64(18)); init=G.starting_configuration(m,4,1)
    for method in B.METHODS
        s=B.setup(m,init,method,UInt64(43),30.)
        w=method==:dpp_marginal_sm ? B.SM.Workspace(m) : nothing
        for _ in 1:30
            B.step!(s,method,w,Xoshiro(672))
            b=B.base(s); @test sum(c.n for c in b.clusters)==length(m.y)
            @test all(h->count(==(h),b.c)==b.clusters[h].n,eachindex(b.clusters))
            vals=B.scalars(s,method); @test all(isfinite,vals[1:6]); @test vals[2]>=vals[1]
            if method in (:dpp_general,:strauss_general)
                @test s.geom.lp≈PP.geometry(s.prior,m,s.geom.xs).lp atol=1e-8
            end
        end
        before=IOBuffer(); serialize(before,s)
        lp=B.predictive(s,method,[1,10,20],Xoshiro(4))
        after=IOBuffer(); serialize(after,s)
        @test all(isfinite,lp)
        @test take!(before)==take!(after)
    end
end

@testset "Split-merge acceptance versus full target" begin
    m=fixture(); init=G.Initialization([1,1,2],[SVector(-.3),SVector(.7)],collect(eachindex(m.spectrum.kind)))
    s=G.initialize(m,init,:marginal_collapsed,UInt64(32)); w=B.SM.Workspace(m); checked=0
    for seed in 1:50
        p=B.SM.propose!(w,s,Xoshiro(seed);anchors=isodd(seed) ? (1,2) : (1,3))
        p.ratio===nothing && continue
        unaffected=[s.clusters[h].x for h in s.active if !(h in p.oldids)]
        raw=B.SM.raw_ratio(m,s.S,unaffected,p.oldgroups,p.oldxs,p.newgroups,p.newxs,p.logs,p.kind)
        @test p.ratio≈raw.raw atol=1e-10; checked+=1
    end
    @test checked>10
end


# This portable suite uses synthetic observations only. It checks target
# calculations, proposal/reverse densities, cache commits, and tempering.
# Application-driver scheduling and research-cohort tests stay outside it.
include("mfm_target.jl")
include("mfm_hyperparameters.jl")
include("mfm_block_moves.jl")
include("mfm_sequential_splitmerge.jl")
include("mfm_sequential_balance.jl")
include("mfm_likelihood_tempering.jl")
include("dpp_reallocation.jl")
include("dpp_multi_reallocation.jl")
include("dpp_collapsed_reallocation.jl")
include("dpp_collapsed_balance.jl")
include("dpp_powered_collapsed.jl")
include("dpp_powered_collapsed_balance.jl")
include("dpp_sequential_splitmerge.jl")
include("dpp_sequential_balance.jl")
include("dpp_likelihood_tempering.jl")
include("mahalanobis_repulsion.jl")
include("mahalanobis_strauss.jl")
include("cli.jl")
include("poisson_lognormal.jl")

include("species_sampling.jl")
include("species_sampling_cli.jl")

include("neal_samplers.jl")
include("neal_cli.jl")
