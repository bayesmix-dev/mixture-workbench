# Standalone: julia --startup-file=no --project=. test/efficient_slice.jl
module EfficientSliceTests
using Test, Random, Distributions, SpecialFunctions, Statistics, LinearAlgebra,
    QuadGK, Serialization

module Subject
include(joinpath(@__DIR__,"..","src","SpeciesSampling.jl"))
include(joinpath(@__DIR__,"..","src","EfficientSlice.jl"))
end
const SS=Subject.SpeciesSampling
const ES=Subject.EfficientSlice
bytes(x)=begin io=IOBuffer();serialize(io,x);take!(io);end

function canonical(z)
    labels=Dict{Int,Int}()
    [get!(labels,j,length(labels)+1) for j in z]
end

# The joint Gaussian distribution obtained by drawing one Normal mean for a
# group. This reference does not use the production integrated-likelihood code.
function reference_evidence(y,ids,observation_sd,base_mean,base_sd)
    n=length(ids)
    n==0 && return 0.0
    covariance=observation_sd^2*Matrix{Float64}(I,n,n)+base_sd^2*ones(n,n)
    logpdf(MvNormal(fill(base_mean,n),covariance),y[ids])
end

@testset "Dependent slice-efficient DP sampler" begin
    @testset "ordered labels, gaps and stick shapes" begin
        model=SS.GaussianLocation([-.7,.2,1.3];observation_sd=.8,base_mean=.1,base_sd=1.2)
        s=ES.initialize(model,SS.DirichletProcess(.6),17;initial_labels=[1,3,3])
        @test s.z==[1,3,3]
        @test ES.ordered_counts(s)==[1,0,2]
        @test ES.stick_shapes(s)==[(2.,2.6),(1.,2.6),(3.,.6)]
        @test ES.occupied_count(s)==2 && ES.materialized_count(s)==3
        @test ES.check_state(s;auxiliaries=false)
        @test !ES.check_state(s)
        @test_throws ArgumentError ES.allocation_logweights(s,1)

        s.theta.=[-.4,.3,.9]
        # Numerically integrate each independent Beta prior moment. The gap
        # contributes a (1-V)^tail factor and cannot be removed from the order.
        counts=[1,0,2]
        numerical=0.0
        for j in eachindex(counts)
            tail=sum(counts[j+1:end])
            moment=quadgk(v->pdf(Beta(1.,.6),v)*v^counts[j]*(1-v)^tail,
                0.,1.;rtol=1e-10)[1]
            numerical+=log(moment)
        end
        numerical+=sum(logpdf(Normal(.1,1.2),s.theta[j]) for j in (1,3))+
            sum(logpdf(Normal(s.theta[s.z[i]],.8),model.y[i]) for i in 1:3)
        @test ES.ordered_logtarget(s)≈numerical atol=2e-8

        compact=ES.initialize(model,SS.DirichletProcess(.6),17;initial_labels=[1,2,2])
        compact.theta.=[s.theta[1],s.theta[3]]
        @test ES.marginal_logtarget(compact)≈ES.marginal_logtarget(s) atol=1e-13
        @test !(ES.ordered_logtarget(compact)≈ES.ordered_logtarget(s))
        before=bytes(s)
        ES.ordered_logtarget(s);ES.marginal_logtarget(s);ES.stick_shapes(s)
        @test bytes(s)==before
    end

    @testset "Beta laws and representable small-concentration tails" begin
        rng=Xoshiro(59276)
        for (a,b) in ((1.,.7),(3.,2.4),(8.,.3),(1.,1e-8),(4.,1e-8),(1.,1e8))
            samples=Float64[]
            for _ in 1:12000
                lv,lr=ES.draw_logstick(rng,a,b)
                @test isfinite(lv) && isfinite(lr) && lv<=0 && lr<0
                @test exp(lv)+exp(lr)≈1.0 atol=3e-15
                push!(samples,exp(lv))
            end
            reference=Beta(a,b)
            tolerance=7sqrt(var(reference)/length(samples))+1e-7
            @test abs(mean(samples)-mean(reference))<tolerance
        end
        # An ordinary Beta(1,1e-8) often rounds to one; its logarithmic residual
        # still has a finite, nonzero value and is retained by this draw.
        lv,lr=ES.draw_logstick(Xoshiro(7),1.,1e-8)
        @test exp(lv)==1. && isfinite(lr) && lr<0
        for (a,b) in ((0.,1.),(.5,1.),(1.,0.),(NaN,1.),(1.,Inf))
            @test_throws ArgumentError ES.draw_logstick(rng,a,b)
        end
        @test_throws ErrorException ES.draw_logstick(rng,1.,nextfloat(0.))
    end

    @testset "Normal component conditional and base refresh" begin
        model=SS.GaussianLocation([-.5,.2,1.1];observation_sd=.7,base_mean=.3,base_sd=1.4)
        s=ES.initialize(model,SS.DirichletProcess(.8),362;initial_labels=[1,3,3])
        samples=zeros(20000,3)
        for i in axes(samples,1)
            ES.refresh_components!(s)
            samples[i,:].=s.theta
        end
        for (j,ids) in ((1,[1]),(2,Int[]),(3,[2,3]))
            variance=1/(1/1.4^2+length(ids)/.7^2)
            mu=variance*(.3/1.4^2+sum(model.y[ids])/.7^2)
            @test abs(mean(samples[:,j])-mu)<7sqrt(variance/size(samples,1))
            @test abs(var(samples[:,j])-variance)<7variance*sqrt(2/(size(samples,1)-1))
        end
    end

    @testset "slice eligibility, exact tail stopping and allocation law" begin
        model=SS.GaussianLocation([-1.,.1,.8,1.4];observation_sd=.6,base_mean=.2,base_sd=1.1)
        s=ES.initialize(model,SS.DirichletProcess(1.7),987;initial_labels=[1,4,4,1])
        ES.refresh_components!(s)
        ES.draw_auxiliaries!(s)
        @test length(s.theta)==length(s.logweights)
        @test s.logresidual<minimum(s.logslices)
        @test sum(exp.(s.logweights))+exp(s.logresidual)≈1.0 atol=3e-15
        for i in eachindex(s.z)
            before=bytes(s)
            q=ES.allocation_probabilities(s,i)
            direct=[s.logslices[i]<s.logweights[j] ?
                pdf(Normal(s.theta[j],.6),model.y[i]) : 0. for j in eachindex(s.theta)]
            direct./=sum(direct)
            @test q≈direct atol=2e-15
            @test q[s.z[i]]>0
            @test bytes(s)==before
        end
        @test ES.check_state(s)
        @test_throws ArgumentError ES.draw_auxiliaries!(s)
        for _ in 1:300
            ES.sweep!(s)
            @test ES.check_state(s)
            @test sum(exp.(s.logweights))+exp(s.logresidual)≈1.0 atol=5e-15
        end
        @test s.iteration==300 && s.allocation_updates==1200
        @test s.stick_draws>=300
        small=ES.initialize(model,SS.DirichletProcess(1e-8),23)
        for _ in 1:30
            ES.sweep!(small)
            @test ES.check_state(small)
            @test isfinite(small.logresidual)
        end
    end

    @testset "independent exact Gaussian partition posterior" begin
        y=[-.8,.2,1.5]
        alpha=.7;observation_sd=.8;base_mean=.1;base_sd=1.3
        model=SS.GaussianLocation(y;observation_sd,base_mean,base_sd)
        partitions=[(1,1,1),(1,1,2),(1,2,1),(1,2,2),(1,2,3)]
        raw=Float64[]
        for key in partitions
            k=maximum(key)
            groups=[findall(==(j),collect(key)) for j in 1:k]
            # Direct CRP sequential probabilities, not the implementation's
            # gamma-function EPPF or its integrated component calculations.
            allocation=1.0
            for i in 2:length(key)
                previous=key[1:i-1]
                nj=count(==(key[i]),previous)
                allocation*=(nj==0 ? alpha : nj)/(alpha+i-1)
            end
            evidence=sum(reference_evidence(y,g,observation_sd,base_mean,base_sd) for g in groups)
            push!(raw,allocation*exp(evidence))
            state=ES.initialize(model,SS.DirichletProcess(alpha),1;initial_labels=collect(key))
            @test ES.marginal_logtarget(state)≈log(allocation)+evidence atol=3e-14
        end
        exact=raw/sum(raw)
        @test sum(exact)≈1.0 atol=2e-15
        s=ES.initialize(model,SS.DirichletProcess(alpha),20261109)
        for _ in 1:4000;ES.sweep!(s);end
        batches=zeros(100,length(partitions))
        for b in axes(batches,1),_ in 1:400
            ES.sweep!(s)
            key=Tuple(canonical(s.z))
            j=findfirst(==(key),partitions)
            batches[b,j]+=1/400
        end
        for j in eachindex(partitions)
            @test abs(mean(batches[:,j])-exact[j])<.012+7std(batches[:,j])/sqrt(size(batches,1))
        end
    end

    @testset "reproducibility, unsupported priors and input boundaries" begin
        model=SS.GaussianLocation([0.,1.,3.];observation_sd=.7,base_sd=1.2)
        a=ES.initialize(model,SS.DirichletProcess(.8),421)
        b=ES.initialize(model,SS.DirichletProcess(.8),421)
        c=ES.initialize(model,SS.DirichletProcess(.8),422)
        for _ in 1:100
            ES.sweep!(a);ES.sweep!(b)
            @test bytes(a)==bytes(b)
        end
        ES.sweep!(c)
        @test c.theta!=a.theta
        @test_throws ArgumentError ES.initialize(model,SS.PitmanYor(.8,.2),1)
        @test_throws ArgumentError ES.initialize(model,SS.PitmanYor(.8,0.),1)
        @test_throws ArgumentError ES.initialize(model,SS.DirichletProcess(.8),true)
        @test_throws ArgumentError ES.initialize(model,SS.DirichletProcess(.8),-1)
        @test_throws ArgumentError ES.initialize(model,SS.DirichletProcess(.8),1.5)
        rng=Xoshiro(992);before=bytes(rng)
        fromrng=ES.initialize(model,SS.DirichletProcess(.8),rng)
        @test bytes(rng)==before
        @test bytes(fromrng)==bytes(ES.initialize(model,SS.DirichletProcess(.8),rng))
        for labels in (Int[],[1,1],[1,0,1],Any[1,true,2],[1.,2.,1.])
            @test_throws ArgumentError ES.initialize(model,SS.DirichletProcess(.8),1;initial_labels=labels)
        end
        bad=deepcopy(a);bad.z[1]=0
        @test !ES.check_state(bad;auxiliaries=false)
        @test_throws ArgumentError ES.sweep!(bad)
    end
end
end
