if !isdefined(Main,:Benchmark);include("../src/execution/Benchmark.jl");end
if !isdefined(Main.Benchmark,:MFMHyperparameters)
    Base.include(Main.Benchmark,joinpath(@__DIR__,"../src/models/mfm/algorithms/MFMHyperparameters.jl"))
end
module MFMHyperparameterTests
using Test,GeneralDPPMixtures,Random,StaticArrays,LinearAlgebra,Statistics,Serialization,QuadGK,Distributions
const G=GeneralDPPMixtures
const B=Main.Benchmark
const MF=B.MFMBaseline
const H=B.MFMHyperparameters
function statebytes(s)
    io=IOBuffer();serialize(io,s);take!(io)
end
function fixture(labels,M)
    y=SVector{1,Float64}.([-1.,-.2,.4,1.1])
    m=G.Model(y,ones(Int,4),[-2.],[2.],G.make_spectrum(1);tau=3.,omega=reshape([.7],1,1))
    xs=[mean(y[findall(==(h),labels)]) for h in 1:maximum(labels)]
    s=MF.initialize(m,G.Initialization(labels,xs,collect(1:5)),UInt64(3901));s.M=M;s.gamma=1.7
    s
end
# Finite products of Dirichlet moments, independent of loggamma differences.
function logallocation(s,g)
    g>0 || return -Inf
    sum(sum(log(g+j) for j in 0:s.clusters[h].n-1) for h in s.active)-
        sum(log(s.M*g+j) for j in 0:length(s.c)-1)
end
@testset "Exact optional MFM log-gamma slice" begin
    @testset "Jacobian, atomic safeguards, metadata and unchanged partition" begin
        s=fixture([1,1,2,2],3)
        for eta in (-8.,-2.,0.,2.,4.)
            @test H.log_gamma_density(s,eta)-H.log_gamma_density(s,0.)≈
                logallocation(s,exp(eta))-.1exp(eta)+eta-(logallocation(s,1.)-.1) atol=3e-11
        end
        before=statebytes(s)
        @test_throws ArgumentError H.slice_gamma!(s;width=0.)
        @test statebytes(s)==before
        @test_throws ErrorException H.slice_gamma!(s;width=1e-12,max_steps=1)
        @test statebytes(s)==before
        c=copy(s.c);M=s.M;lambda=s.lambda;attempts=s.gamma_attempts;accepts=s.gamma_accepts
        info=H.slice_gamma!(s)
        @test s.gamma==info.gamma && info.evaluations>=4 && info.steps>=0 && info.shrinks>=0
        @test s.c==c && s.M==M && s.lambda==lambda && s.gamma_attempts==attempts && s.gamma_accepts==accepts
    end
    @testset "independent conditional draws and quadrature at different counts" begin
        N=3000
        for (case,(labels,M)) in enumerate((([1,1,2,2],2),([1,1,2,2],5),([1,1,1,2],4)))
            s=fixture(labels,M);n=length(s.c)
            # E_Dirichlet prod(p_h^n_h) <= max_p prod(p_h^n_h).
            # Exp(.1) proposals with this bound give iid exact conditionals.
            bound=sum(s.clusters[h].n*log(s.clusters[h].n/n) for h in s.active)
            rng=Xoshiro(39800+case);before=zeros(N,6);after=similar(before)
            norm=quadgk(g->exp(logallocation(s,g)-.1g),0.,Inf;rtol=1e-10)[1]
            cutoffs=[.5,2.,10.]
            exact=[quadgk(g->exp(logallocation(s,g)-.1g),0.,cut;rtol=1e-10)[1]/norm for cut in cutoffs]
            function observe!(row,g)
                row[1]=g;row[2]=g^2;row[3]=log(g)
                for j in 1:3;row[3+j]=g<cutoffs[j];end
            end
            for t in 1:N
                sampled=false
                for _ in 1:1000000
                    g=rand(rng,Exponential(10.));probability=exp(logallocation(s,g)-bound)
                    @assert 0<=probability<=1+1e-12
                    rand(rng)<probability || continue
                    s.gamma=g;sampled=true;break
                end
                sampled || error("independent gamma reference rejection sampler exhausted cap")
                observe!(view(before,t,:),s.gamma)
                H.slice_gamma!(s)
                observe!(view(after,t,:),s.gamma)
            end
            difference=after-before;se=vec(std(difference;dims=1))/sqrt(N)
            @test all(abs.(vec(mean(difference;dims=1))).<=6se.+1/N)
            for j in 1:3
                allowance=6sqrt(exact[j]*(1-exact[j])/N)+1/N
                @test abs(mean(before[:,3+j])-exact[j])<=allowance
                @test abs(mean(after[:,3+j])-exact[j])<=allowance
            end
        end
    end
end
end
