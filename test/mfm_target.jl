if !isdefined(Main,:Benchmark);include("../src/Benchmark.jl");end
if !isdefined(Main.Benchmark,:MFMTarget)
    Base.include(Main.Benchmark,joinpath(@__DIR__,"../src/MFMTarget.jl"))
end
module MFMTargetTests
using Test,GeneralDPPMixtures,LinearAlgebra,Random,StaticArrays,SpecialFunctions,Distributions,Statistics
const G=GeneralDPPMixtures
const B=Main.Benchmark
const MF=B.MFMBaseline
const T=B.MFMTarget
function fixture(d=2;n=8)
    rng=Xoshiro(834+d);y=[SVector{d,Float64}(randn(rng,d)) for _ in 1:n]
    m=G.Model(y,ones(Int,n),fill(-4.,d),fill(4.,d),G.make_spectrum(d);
        a=.37,tau=d+2.,omega=.8Matrix{Float64}(I,d,d))
    labels=n==8 ? [1,1,1,2,2,3,4,4] : ones(Int,n)
    xs=[mean(y[findall(==(h),labels)]) for h in 1:maximum(labels)]
    s=MF.initialize(m,G.Initialization(labels,xs,collect(1:12)),UInt64(722))
    s.M=max(6,maximum(labels));s.gamma=.8;s.lambda=1.7
    s
end
@testset "MFM collapsed NIW and joint target" begin
    @testset "NIW evidence against independent Bayes densities and predictives" begin
        for d in (1,2,3)
            s=fixture(d);m=s.model
            for ids in ([1],[1,3,4],[1,2,3,4,6,7,8])
                c=T.member_cluster(s,ids);n=c.n;k=s.kappa+n
                mu=Vector(s.mu0).+.2;Sigma=Matrix(m.omega)*1.31
                delta=c.mean-s.mu0
                scale=Matrix(m.omega+c.scatter+s.kappa*n/k*(delta*delta'))
                mun=Vector((s.kappa*s.mu0+n*c.mean)/k)
                independent=sum(logpdf(MvNormal(mu,Sigma),Vector(m.y[i])) for i in ids)+
                    logpdf(MvNormal(Vector(s.mu0),Sigma/s.kappa),mu)+
                    logpdf(InverseWishart(m.tau,Matrix(m.omega)),Sigma)-
                    logpdf(MvNormal(mun,Sigma/k),mu)-
                    logpdf(InverseWishart(m.tau+n,scale),Sigma)
                @test T.log_marginal(s,c)≈independent atol=3e-11
                @test T.log_marginal(s,ids)≈independent atol=3e-11
                # Independent sequential predictive route, with fixed prior.
                work=deepcopy(s);cc=work.clusters[1]
                cc.n=0;cc.mean=zero(cc.mean);cc.scatter=zero(cc.scatter);MF.refresh!(work,1)
                total=0.
                for i in reverse(ids)
                    total+=MF.logpred(work,1,m.y[i]);G.add!(cc,m.y[i]);MF.refresh!(work,1)
                end
                @test total≈independent atol=3e-11
                changed=deepcopy(c);changed.x=100ones(typeof(c.x));changed.logdet=876.
                @test T.log_marginal(s,changed)==T.log_marginal(s,c)
            end
            @test T.log_marginal(s,Int[])==0.
            @test T.likelihood(s)≈sum(T.log_marginal(s,s.clusters[h]) for h in s.active)
        end
    end
    @testset "canonical partition multiplicity by labelled allocation enumeration" begin
        s=fixture(1;n=4);n=length(s.c);g=s.gamma
        for M in (2,3,4)
            mass=Dict{Tuple,Float64}()
            for labels in Iterators.product(ntuple(_->1:M,n)...)
                counts=[count(==(h),labels) for h in 1:M]
                multinomial=loggamma(n+1)-sum(loggamma(v+1) for v in counts)
                prob=exp(logpdf(DirichletMultinomial(n,fill(g,M)),counts)-multinomial)
                canonical=Tuple(G.canonical_labels(collect(labels)))
                mass[canonical]=get(mass,canonical,0.)+prob
            end
            @test sum(values(mass))≈1. atol=2e-13
            for (labels,prob) in mass
                groups=[findall(==(h),collect(labels)) for h in 1:maximum(labels)]
                evidence=sum(T.log_marginal(s,ids) for ids in groups)
                @test T.partition_logtarget(s;groups,M,gamma=g)-evidence≈log(prob) atol=4e-12
            end
        end
        @test T.logfallingfactorial(5,3)≈log(60.) atol=1e-14
        @test T.logfallingfactorial(5,0)==0.
        @test T.logfallingfactorial(2,3)==-Inf
    end
    @testset "existing and new allocation odds, including singleton deletion" begin
        s=fixture()
        for i in (1,6)
            base=deepcopy(s);old=base.c[i];G.remove!(base.clusters[old],base.model.y[i]);MF.refresh!(base,old)
            base.clusters[old].n==0 && deleteat!(base.active,findfirst(==(old),base.active))
            k=length(base.active);empty=MF.empty_slot!(base)
            ids=vcat(base.active,[empty]);scores=vcat([log(base.clusters[h].n+base.gamma)+MF.logpred(base,h,base.model.y[i]) for h in base.active],
                log(base.M-k)+log(base.gamma)+MF.logpred(base,empty,base.model.y[i]))
            targets=Float64[]
            for h in ids
                candidate=deepcopy(base);candidate.c[i]=h
                h in candidate.active || push!(candidate.active,h)
                G.add!(candidate.clusters[h],candidate.model.y[i]);MF.refresh!(candidate,h)
                push!(targets,T.partition_logtarget(candidate))
            end
            @test targets.-targets[1]≈scores.-scores[1] atol=4e-12
        end
    end
    @testset "M, gamma and Lambda hyperconditional identities" begin
        s=fixture(2);base=T.logtarget(s)
        for g in (.001,.1,.8,3.,30.)
            @test T.logtarget(s;gamma=g)-base≈MF.gamma_logtarget(s,g)-MF.gamma_logtarget(s,s.gamma) atol=4e-11
        end
        for lambda in (.01,.4,2.,20.)
            exact=logpdf(Gamma(s.M,1/1.1),lambda)-logpdf(Gamma(s.M,1/1.1),s.lambda)
            @test T.logtarget(s;lambda)-base≈exact atol=3e-11
        end
        for M in length(s.active):14
            exact=logpdf(Poisson(s.lambda),M-1)+T.logfallingfactorial(M,length(s.active))+
                loggamma(M*s.gamma)-loggamma(M*s.gamma+length(s.c))
            old=logpdf(Poisson(s.lambda),s.M-1)+T.logfallingfactorial(s.M,length(s.active))+
                loggamma(s.M*s.gamma)-loggamma(s.M*s.gamma+length(s.c))
            @test T.logtarget(s;M)-base≈exact-old atol=3e-11
        end
        @test T.logtarget(s;M=1)==-Inf
        @test T.logtarget(s;gamma=0.)==-Inf
        @test T.logtarget(s;lambda=0.)==-Inf
    end
    @testset "joint occupied/total count split and reverse factors" begin
        s=fixture(2)
        old=[findall(==(h),s.c) for h in s.active]
        split=vcat([[old[1][1]],old[1][2:end]],old[2:end])
        k=length(old);n=length(s.c);g=s.gamma;lambda=s.lambda
        clusterterm(gs)=sum(loggamma(length(ids)+g)-loggamma(g)+T.log_marginal(s,ids) for ids in gs)
        B(M)=loggamma(M*g)-loggamma(M*g+n)
        for M in k:k+4
            target=T.logtarget(s;groups=split,M=M+1)-T.logtarget(s;groups=old,M)
            expected=clusterterm(split)-clusterterm(old)+log(lambda)+log(M+1)-log(M)+B(M+1)-B(M)
            @test target≈expected atol=4e-11
            reverse=T.logtarget(s;groups=old,M)-T.logtarget(s;groups=split,M=M+1)
            reversecount=log(M)-log(M+1)-log(lambda)+B(M)-B(M+1)
            @test reverse≈clusterterm(old)-clusterterm(split)+reversecount atol=4e-11
        end
    end
end
end
