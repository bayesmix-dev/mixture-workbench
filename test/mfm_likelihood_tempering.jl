# Standalone: julia --project=. --threads=2 test/mfm_likelihood_tempering.jl
if !isdefined(Main,:Benchmark); include("../src/execution/Benchmark.jl"); end
if !isdefined(Main.Benchmark,:MFMTarget)
    Base.include(Main.Benchmark,joinpath(@__DIR__,"../src/models/mfm/MFMTarget.jl"))
end
if !isdefined(Main.Benchmark,:MFMLikelihoodTempering)
    Base.include(Main.Benchmark,joinpath(@__DIR__,"../src/models/mfm/algorithms/MFMLikelihoodTempering.jl"))
end

module MFMLikelihoodTemperingTests
using Test, GeneralDPPMixtures, Random, LinearAlgebra, StaticArrays, Statistics,
      Serialization, Distributions, SpecialFunctions, QuadGK
const G=GeneralDPPMixtures
const MF=Main.Benchmark.MFMBaseline
const MT=Main.Benchmark.MFMTarget
const T=Main.Benchmark.MFMLikelihoodTempering

function statebytes(s)
    io=IOBuffer(); serialize(io,s); take!(io)
end
function partitions(n,M=n)
    out=Vector{Vector{Int}}()
    function rec(c)
        length(c)==n && (push!(out,c); return)
        for h in 1:min(M,maximum(c)+1); rec(vcat(c,h)); end
    end
    rec([1]);out
end
function fixture(d=1;n=4)
    rng=Xoshiro(16273+d+n)
    y=d==1 && n==4 ? [SVector(-.8),SVector(-.2),SVector(.4),SVector(1.1)] :
        [SVector{d,Float64}(randn(rng,d)) for _ in 1:n]
    G.Model(y,ones(Int,n),fill(-4.,d),fill(4.,d),G.make_spectrum(d);
        tau=d+2.,omega=.7Matrix{Float64}(I,d,d))
end
function state(m,c;M=3,gamma=.43,lambda=1.7,seed=186)
    s=MF.initialize(m,G.Initialization(c,[zero(first(m.y)) for _ in 1:maximum(c)],[1]),seed)
    s.M=M;s.gamma=gamma;s.lambda=lambda;s
end
function audit(s)
    @test sort(unique(s.c))==sort(s.active)
    @test length(s.active)<=s.M && sum(c.n for c in s.clusters)==length(s.c)
    for h in eachindex(s.clusters)
        ids=findall(==(h),s.c);c=s.clusters[h]
        @test c.n==length(ids)
        isempty(ids) && continue
        center=mean(s.model.y[ids]);scatter=sum((s.model.y[i]-center)*(s.model.y[i]-center)' for i in ids)
        @test c.mean≈center atol=2e-11
        @test c.scatter≈scatter atol=3e-10
    end
    fresh=deepcopy(s)
    for h in eachindex(fresh.clusters);MF.refresh!(fresh,h);end
    @test all(a≈b for (a,b) in zip(s.predL,fresh.predL))
    @test s.predmu≈fresh.predmu && s.prednorm≈fresh.prednorm
end

# Independent D=1 evidence: integrate the inverse-gamma variance numerically,
# after integrating the normal mean by its multivariate-normal covariance.
# No NIW marginal or sampler predictive is used in this reference calculation.
function quadrature_evidence(s,ids)
    isempty(ids) && return 0.
    n=length(ids);ys=[s.model.y[i][1] for i in ids]
    varianceprior=InverseGamma(s.model.tau/2,s.model.omega[1,1]/2)
    R=Matrix{Float64}(I,n,n)+ones(n,n)/s.kappa
    mu=fill(s.mu0[1],n)
    integral,error=quadgk(v->pdf(varianceprior,v)*pdf(MvNormal(mu,v*R),ys),0.,Inf;rtol=2e-11,atol=1e-13)
    @test error<max(1e-11,1e-8integral)
    log(integral)
end
function allocation_prior(m,gamma,M)
    n=length(m.y);mass=Dict{Tuple,Float64}()
    for labels in Iterators.product(ntuple(_->1:M,n)...)
        counts=[count(==(h),labels) for h in 1:M]
        multinomial=loggamma(n+1)-sum(loggamma(v+1) for v in counts)
        probability=exp(logpdf(DirichletMultinomial(n,fill(gamma,M)),counts)-multinomial)
        key=Tuple(G.canonical_labels(collect(labels)))
        mass[key]=get(mass,key,0.)+probability
    end
    mass
end

@testset "Exact NIW-collapsed MFM likelihood tempering" begin
    @testset "quadrature targets and exact conditional allocation matrices" begin
        m=fixture();M=3;gamma=.43;pp=partitions(4,M)
        states=[state(m,c;M,gamma) for c in pp]
        prior=allocation_prior(m,gamma,M);@test sum(values(prior))≈1. atol=2e-13
        evidence=Dict{Tuple,Float64}()
        for c in pp,h in unique(c)
            ids=findall(==(h),c);key=Tuple(ids)
            haskey(evidence,key) || (evidence[key]=quadrature_evidence(first(states),ids))
        end
        ll=[sum(evidence[Tuple(findall(==(h),c))] for h in unique(c)) for c in pp]
        hyperprior=logpdf(Poisson(first(states).lambda),M-1)+
            logpdf(Exponential(10.),gamma)+logpdf(Exponential(10.),first(states).lambda)
        for beta in (0.,.37,1.)
            logs=[log(prior[Tuple(c)])+beta*ll[a] for (a,c) in enumerate(pp)]
            probabilities=exp.(logs.-maximum(logs));probabilities./=sum(probabilities)
            @test [T.logtarget(s,beta) for s in states]≈logs.+hyperprior atol=5e-10
            full=Matrix{Float64}(I,length(pp),length(pp))
            for i in 1:4
                transition=zeros(length(pp),length(pp))
                for (a,s) in enumerate(states)
                    removed=deepcopy(s);old=removed.c[i]
                    G.remove!(removed.clusters[old],m.y[i]);MF.refresh!(removed,old)
                    removed.clusters[old].n==0 && deleteat!(removed.active,findfirst(==(old),removed.active))
                    scores=Float64[];empty=T.allocation_scores!(scores,removed,beta,i)
                    ids=empty==0 ? copy(removed.active) : vcat(removed.active,empty)
                    endpointlogs=Float64[];endpoints=Int[]
                    for h in ids
                        candidate=deepcopy(removed);candidate.c[i]=h
                        h in candidate.active || push!(candidate.active,h)
                        G.add!(candidate.clusters[h],m.y[i]);MF.refresh!(candidate,h)
                        b=findfirst(==(G.canonical_labels(candidate.c)),pp)
                        push!(endpoints,b);push!(endpointlogs,logs[b])
                    end
                    p=exp.(scores.-maximum(scores));p./=sum(p)
                    independent=exp.(endpointlogs.-maximum(endpointlogs));independent./=sum(independent)
                    @test p≈independent atol=2e-11
                    for (q,b) in zip(p,endpoints);transition[a,b]+=q;end
                end
                @test all(abs.(vec(sum(transition;dims=2)).-1).<2e-13)
                flux=Diagonal(probabilities)*transition
                @test flux≈flux' atol=2e-12
                @test vec(probabilities'transition)≈probabilities atol=2e-12
                full=full*transition
            end
            @test vec(probabilities'full)≈probabilities atol=2e-12
        end
    end

    @testset "hyperconditional powers cancel and cold sweep is byte compatible" begin
        m=fixture(2;n=8);s=state(m,[1,1,1,2,2,3,3,3];M=5,gamma=.71)
        for beta in (0.,.37,1.)
            base=T.logtarget(s,beta)
            for gamma in (.01,.3,2.,13.)
                other=deepcopy(s);other.gamma=gamma
                @test T.logtarget(other,beta)-base≈MF.gamma_logtarget(s,gamma)-MF.gamma_logtarget(s,s.gamma) atol=2e-11
            end
            for lambda in (.2,3.,20.)
                other=deepcopy(s);other.lambda=lambda
                @test T.logtarget(other,beta)-base≈logpdf(Gamma(s.M,1/1.1),lambda)-logpdf(Gamma(s.M,1/1.1),s.lambda) atol=2e-11
            end
            for M in length(s.active):12
                other=deepcopy(s);other.M=M
                ref(v)=logpdf(Poisson(s.lambda),v-1)+MT.logfallingfactorial(v,length(s.active))+
                    loggamma(v*s.gamma)-loggamma(v*s.gamma+length(s.c))
                @test T.logtarget(other,beta)-base≈ref(M)-ref(s.M) atol=2e-11
            end
        end
        for d in (1,3)
            m=fixture(d;n=10);a=state(m,[1,1,2,2,2,3,3,3,4,4];M=6)
            b=deepcopy(a)
            for _ in 1:80
                MF.sweep!(a);T.hot_sweep!(b,1.)
                @test statebytes(a)==statebytes(b)
                @test rand(deepcopy(a.rng))==rand(deepcopy(b.rng))
            end
        end
        before=statebytes(s)
        for beta in (-.1,Inf,NaN)
            @test_throws ArgumentError T.hot_sweep!(s,beta)
            @test_throws ArgumentError T.logtarget(s,beta)
            @test statebytes(s)==before
        end
    end

    @testset "hot caches and independent product-target swaps" begin
        m=fixture();pp=partitions(4,3)
        states=[state(m,c;M=maximum(c)+1,gamma=.2+.1a,lambda=.7+.2a,seed=209+a) for (a,c) in enumerate(pp)]
        for beta in (0.,.37,1.)
            s=deepcopy(states[3])
            for _ in 1:50;T.hot_sweep!(s,beta);audit(s);end
        end
        for beta_a in (0.,.37,1.),beta_b in (0.,.37,1.),a in states,b in states
            ratio=T.swap_logratio(a,b,beta_a,beta_b)
            independent=T.logtarget(b,beta_a)+T.logtarget(a,beta_b)-T.logtarget(a,beta_a)-T.logtarget(b,beta_b)
            @test ratio≈independent atol=4e-11
            @test ratio==-T.swap_logratio(b,a,beta_a,beta_b)
            # The two-state orbit consists of (a,b) and (b,a).
            logs=[T.logtarget(a,beta_a)+T.logtarget(b,beta_b),T.logtarget(b,beta_a)+T.logtarget(a,beta_b)]
            weights=exp.(logs.-maximum(logs));weights./=sum(weights)
            @test weights[1]*min(1.,exp(ratio))≈weights[2]*min(1.,exp(-ratio)) atol=2e-12
        end
        a,b=deepcopy.(states[[2,5]])
        ensemble=[a,b];rng=Xoshiro(38);ra=deepcopy(a.rng);rb=deepcopy(b.rng)
        move=T.attempt_swap!(ensemble,1,2,[1.,1.],rng)
        @test move.accepted && move.logratio==0.
        @test ensemble[1]===b && ensemble[2]===a
        @test rand(deepcopy(ensemble[1].rng))==rand(rb)
        @test rand(deepcopy(ensemble[2].rng))==rand(ra)
        # A rejecting swap mutates neither physical state and consumes only
        # the independent swap RNG, never a physical state's RNG.
        cold,hot=deepcopy.(states[[1,end]])
        Lcold=T.loglikelihood(cold);Lhot=T.loglikelihood(hot)
        betas=Lhot<Lcold ? [1000.,0.] : [0.,1000.]
        before=statebytes([cold,hot]);ensemble=[cold,hot];rejected=false
        for _ in 1:10
            move=T.attempt_swap!(ensemble,1,2,betas,Xoshiro(912))
            if !move.accepted;rejected=true;break;end
        end
        @test rejected && statebytes(ensemble)==before
        @test_throws ArgumentError T.attempt_swap!([a,a],1,2,[1.,0.],Xoshiro(1))
        @test_throws ArgumentError T.attempt_swap!([a,b],1,1,[1.,0.],Xoshiro(1))
        @test_throws ArgumentError T.attempt_swap!([a,b],1,2,[1.],Xoshiro(1))
        other=state(fixture(1;n=5),ones(Int,5))
        @test_throws ArgumentError T.swap_logratio(a,other,1.,.5)
    end

    @testset "serialized ladder continuation and independent threaded replicas" begin
        m=fixture(2;n=8);labels=[1,1,2,2,2,3,3,3];betas=[1.,.7,.37,0.]
        initial=[state(m,labels;M=5,gamma=.3+.2i,seed=410+i) for i in eachindex(betas)]
        function advance!(states,rng,steps;threaded=false)
            for _ in 1:steps
                if threaded
                    Threads.@threads for i in eachindex(states);T.hot_sweep!(states[i],betas[i]);end
                else
                    for i in eachindex(states);T.hot_sweep!(states[i],betas[i]);end
                end
                for i in 1:length(states)-1;T.attempt_swap!(states,i,i+1,betas,rng);end
            end
        end
        serial=deepcopy(initial);threaded=deepcopy(initial);a=Xoshiro(877);b=deepcopy(a)
        advance!(serial,a,35);advance!(threaded,b,35;threaded=true)
        @test statebytes(serial)==statebytes(threaded) && statebytes(a)==statebytes(b)
        checkpointed=deepcopy(initial);c=Xoshiro(877)
        advance!(checkpointed,c,13)
        restored=deserialize(IOBuffer(statebytes((states=checkpointed,swap_rng=c,betas=betas))))
        advance!(restored.states,restored.swap_rng,22)
        @test statebytes(restored.states)==statebytes(serial)
        @test statebytes(restored.swap_rng)==statebytes(a)
        for s in restored.states;audit(s);end
    end
end

end
