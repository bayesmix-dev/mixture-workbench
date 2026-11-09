# Standalone: julia --project=. --threads=2 test/dpp_sequential_splitmerge.jl
if !isdefined(Main,:Benchmark);include("../src/execution/Benchmark.jl");end
if !isdefined(Main.Benchmark,:DPPSequentialSplitMerge)
    Base.include(Main.Benchmark,joinpath(@__DIR__,"../src/models/point_process/algorithms/DPPSequentialSplitMerge.jl"))
end

module DPPSequentialSplitMergeTests
using Test,GeneralDPPMixtures,Random,LinearAlgebra,StaticArrays,Serialization,SpecialFunctions
const G=GeneralDPPMixtures
const B=Main.Benchmark
const SM=B.SM
const SS=B.DPPSequentialSplitMerge
bytes(x)=begin io=IOBuffer();serialize(io,x);take!(io);end

function fixture(d=2;labels=[1,1,1,1,2,2,2,3],M=9,box=5.)
    rng=Xoshiro(5801+d)
    y=[SVector{d,Float64}(randn(rng,d)) for _ in labels]
    m=G.Model(y,ones(Int,length(y)),fill(-box,d),fill(box,d),G.make_spectrum(d);
        a=.37,tau=d+2.,omega=.7Matrix{Float64}(I,d,d))
    xs=[SVector{d,Float64}(fill(min(1.,box)*(-.7+(h-1)*.6),d)) for h in 1:maximum(labels)]
    support=M==0 ? collect(eachindex(m.spectrum.kind)) : collect(1:M)
    s=G.initialize(m,G.Initialization(labels,xs,support),:marginal_collapsed,UInt64(5802))
    fill!(s.workf,0.);fill!(s.workv,0.);fill!(s.workw,0.)
    s
end

# Raw observations, determinant and scalar gamma sums; no production cluster
# statistics or logA implementation participate in this reference formula.
function integrated(m,g)
    n=length(g);d=length(first(m.y));mu=sum(m.y[t] for t in g)/n
    scatter=sum((m.y[t]-mu)*(m.y[t]-mu)' for t in g)
    logmultigamma(a)=d*(d-1)/4*log(pi)+sum(loggamma(a+(1-j)/2) for j in 1:d)
    -(n-1)*d/2*log(pi)-d/2*log(n)+m.tau/2*logdet(Matrix(m.omega))-
        (m.tau+n-1)/2*logdet(Matrix(m.omega+scatter))+
        logmultigamma((m.tau+n-1)/2)-logmultigamma(m.tau/2)
end

function audit(s)
    @test SM.audit(s)["status"]=="passed"
    @test length(s.active)<=length(s.S)
    @test sort(unique(s.c))==sort(s.active)
    @test sum(c.n for c in s.clusters)==length(s.c)
    for h in s.active
        c=SM.cluster(s.model,findall(==(h),s.c),s.clusters[h].x)
        @test c.n==s.clusters[h].n
        @test c.mean≈s.clusters[h].mean atol=3e-12
        @test c.scatter≈s.clusters[h].scatter atol=3e-12
        @test c.scaleL≈s.clusters[h].scaleL atol=3e-12
    end
end

@testset "DPP anchor-only sequential split/merge" begin
    @testset "partial endpoint density, normalization and untouched RNG" begin
        for d in (1,2)
            s=fixture(d);m=s.model;D=[1,2,3,4];anchors=[1,2];order=[4,3];total=0.
            for values in Iterators.product(1:2,1:2)
                forced=zeros(Int,length(s.c));forced[anchors].=[1,2];forced[order].=collect(values)
                rng=Xoshiro(31);rbefore=bytes(rng);before=bytes(s)
                groups,lp=SS.sequential_split!(m,D,anchors,order,rng;forced)
                partial=[[1],[2]];expected=0.
                for t in order
                    scores=[log(length(g)+m.a)+integrated(m,vcat(g,t))-integrated(m,g) for g in partial]
                    top=maximum(scores);expected+=scores[forced[t]]-top-log(sum(exp.(scores.-top)))
                    push!(partial[forced[t]],t)
                end
                @test lp≈expected atol=8e-12
                @test groups==sort!.(partial)
                @test bytes(rng)==rbefore && bytes(s)==before
                total+=exp(lp)
            end
            @test total≈1. atol=5e-14
            rng=Xoshiro(12);before=bytes(rng)
            @test SS.sequential_split!(m,[1,2],[1,2],Int[],rng)==([[1],[2]],0.)
            @test bytes(rng)==before
            @test_throws ArgumentError SS.sequential_split!(m,D,[1,1],order,rng)
            @test_throws ArgumentError SS.sequential_split!(m,D,anchors,[3,3],rng)
            @test_throws ArgumentError SS.sequential_split!(m,D,anchors,order,rng;forced=ones(Int,length(s.c)))
            @test_throws ArgumentError SS.sequential_split!(m,[1,1,2],anchors,Int[],rng)
        end
    end

    @testset "unsimplified continuous targets, replay, atomic commits and streams" begin
        finite=0;splitchecked=0;mergechecked=0
        for d in (1,2),seed in 1:20,anchors in ((1,2),(1,5),(5,1))
            s=fixture(d);w=SS.Workspace(s.model);rng=Xoshiro(seed);before=bytes(s)
            D=findall(t->s.c[t] in (s.c[anchors[1]],s.c[anchors[2]]),eachindex(s.c))
            order=reverse(setdiff(D,collect(anchors)))
            p=SS.propose!(w,s,rng;anchors,order)
            @test bytes(s)==before
            @test p.kind==(s.c[anchors[1]]==s.c[anchors[2]] ? :split : :merge)
            @test p.k==length(s.active) && p.M==length(s.S)
            p.ratio===nothing && continue
            direct=SS.raw_ratio(s,p)
            @test p.ratio≈direct.raw atol=4e-9
            other=[s.clusters[h].x for h in s.active if !(h in p.oldids)]
            reversedensity=SM.raw_ratio(s.model,s.S,other,p.newgroups,p.newxs,p.oldgroups,p.oldxs,p.logs,
                p.kind==:split ? :merge : :split)
            @test reversedensity.raw≈-p.ratio atol=4e-9
            candidate=deepcopy(s);accepted=deepcopy(p);accepted.outcome=:accepted
            SS.commit!(candidate,accepted)
            @test G.collapsed_logtarget(candidate)-G.collapsed_logtarget(s)≈direct.target_difference atol=4e-9
            @test candidate.S==s.S && candidate.bound==s.bound
            @test bytes(candidate.streams.obs)==bytes(s.streams.obs)
            @test bytes(candidate.streams.accept)==bytes(s.streams.accept)
            @test bytes(candidate.streams.spectral)==bytes(s.streams.spectral)
            @test bytes(candidate.streams.palm)==bytes(s.streams.palm)
            @test bytes(candidate.streams.weights)==bytes(s.streams.weights)
            @test bytes(candidate.streams.loc[1:length(s.streams.loc)])==bytes(s.streams.loc)
            @test bytes(candidate.streams.cov[1:length(s.streams.cov)])==bytes(s.streams.cov)
            audit(candidate)
            for h in setdiff(s.active,p.oldids)
                @test all(getfield(s.clusters[h],f)==getfield(candidate.clusters[h],f) for f in fieldnames(typeof(s.clusters[h])))
            end
            if p.kind==:split
                forced=zeros(Int,length(s.c));forced[p.newgroups[1]].=1;forced[p.newgroups[2]].=2
                _,lp=SS.sequential_split!(s.model,D,collect(anchors),order,Xoshiro(77);forced)
                @test lp≈p.logs atol=4e-12
                q=SS.propose!(w,candidate,Xoshiro(71);anchors,order)
                @test q.kind==:merge
                @test q.logs≈p.logs atol=4e-12
                splitchecked+=1
            else
                forced=zeros(Int,length(s.c));forced[p.oldgroups[1]].=1;forced[p.oldgroups[2]].=2
                _,lp=SS.sequential_split!(s.model,D,collect(anchors),order,Xoshiro(77);forced)
                @test lp≈p.logs atol=4e-12
                mergechecked+=1
            end
            rejected=deepcopy(p);rejected.outcome=:mh_rejected
            SS.commit!(s,rejected);@test bytes(s)==before
            finite+=1
        end
        @test finite>75 && splitchecked>20 && mergechecked>40
    end

    @testset "capacity, outside-domain self-transitions and external RNG" begin
        s=fixture();s.S=copy(s.S[1:length(s.active)]);G.rebuild!(s)
        before=bytes(s);rng=Xoshiro(17);rbefore=bytes(rng)
        p=SS.propose!(SS.Workspace(),s,rng;anchors=(1,2))
        @test p.outcome==:capacity && p.ratio===nothing
        SS.commit!(s,p);@test bytes(s)==before && bytes(rng)==rbefore
        @test_throws ArgumentError SS.propose!(SS.Workspace(),s,rng;anchors=(1,1))
        @test_throws ArgumentError SS.propose!(SS.Workspace(),fixture(),rng;anchors=(1,2),order=[3,3])
        # A fallible geometry rebuild occurs before native caches/streams commit.
        s=fixture();p=SS.propose!(SS.Workspace(),s,Xoshiro(29);anchors=(1,2))
        p.outcome=:accepted;p.newxs=[p.oldxs[1],p.oldxs[1]];before=bytes(s)
        @test_throws ErrorException SS.commit!(s,p)
        @test bytes(s)==before
        outside=0
        for seed in 1:16
            s=fixture(1;box=.03);before=bytes(s);rng=Xoshiro(seed);rbefore=bytes(rng)
            # Relocate existing centers inside the tiny domain before proposing.
            for (h,id) in enumerate(s.active);s.clusters[id].x=SVector(-.02+.015(h-1));G.refresh!(s.clusters[id],s.model);end
            G.rebuild!(s);before=bytes(s)
            p=SS.attempt!(SS.Workspace(),s,rng;anchors=(1,2))
            if p.outcome==:outside_domain
                outside+=1;@test bytes(s)==before && bytes(rng)!=rbefore
            end
        end
        @test outside>0
        s=fixture(1;labels=[1],M=1);before=bytes(s);rng=Xoshiro(19);rbefore=bytes(rng)
        @test SS.attempt!(SS.Workspace(),s,rng)===nothing
        @test bytes(s)==before && bytes(rng)==rbefore
        a=fixture();b=deepcopy(a);ra=Xoshiro(21);rb=deepcopy(ra);observed=Ref(0)
        pa=SS.attempt!(SS.Workspace(),a,ra)
        pb=SS.attempt!(SS.Workspace(),b,rb;geometry_lock=ReentrantLock(),observer=(_,p)->(observed[]+=1))
        @test bytes(a)==bytes(b) && bytes(ra)==bytes(rb) && bytes(pa)==bytes(pb) && observed[]==1
    end

    @testset "stateless shared workspace has deterministic threaded proposals" begin
        states=[fixture() for _ in 1:8];serial=deepcopy(states)
        rngs=[Xoshiro(5803+j) for j in eachindex(states)];srngs=deepcopy(rngs)
        w=SS.Workspace();lock=ReentrantLock()
        a=[SS.attempt!(w,serial[j],srngs[j];geometry_lock=lock) for j in eachindex(states)]
        b=Vector{Any}(undef,length(states))
        @sync for j in eachindex(states)
            Threads.@spawn b[j]=SS.attempt!(w,states[j],rngs[j];geometry_lock=lock)
        end
        @test all(bytes(states[j])==bytes(serial[j]) for j in eachindex(states))
        @test all(bytes(rngs[j])==bytes(srngs[j]) for j in eachindex(states))
        @test all(bytes(a[j])==bytes(b[j]) for j in eachindex(states))
    end

end

end
