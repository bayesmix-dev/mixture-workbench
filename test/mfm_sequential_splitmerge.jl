# Standalone: julia --project=. test/mfm_sequential_splitmerge.jl
if !isdefined(Main,:Benchmark); include("../src/Benchmark.jl"); end

module MFMSequentialSplitMergeTests
using Test,GeneralDPPMixtures,Random,LinearAlgebra,StaticArrays,Serialization,SpecialFunctions
const G=GeneralDPPMixtures
const B=Main.Benchmark
const MF=B.MFMBaseline
const MT=B.MFMTarget
const MB=B.MFMBlockMoves
const S=B.MFMSequentialSplitMerge

function bytes(x)
    io=IOBuffer();serialize(io,x);take!(io)
end

function fixture(d=2;labels=[1,1,1,1,2,2,2,3],M=5)
    rng=Xoshiro(812+d)
    y=[SVector{d,Float64}(randn(rng,d)) for _ in labels]
    m=G.Model(y,ones(Int,length(y)),fill(-6.,d),fill(6.,d),G.make_spectrum(d);
        tau=d+2.,omega=.7Matrix{Float64}(I,d,d))
    s=MF.initialize(m,G.Initialization(labels,[zero(first(y)) for _ in 1:maximum(labels)],[1]),UInt64(173))
    s.M=M;s.gamma=.64;s.lambda=2.3
    s
end
function audit(s)
    @test sort(unique(s.c))==sort(s.active)
    @test length(s.active)<=s.M
    @test sum(c.n for c in s.clusters)==length(s.c)
    for h in eachindex(s.clusters)
        fresh=MB.cluster(s,findall(==(h),s.c))
        @test s.clusters[h].n==fresh.n
        @test s.clusters[h].mean≈fresh.mean atol=3e-12
        @test s.clusters[h].scatter≈fresh.scatter atol=3e-12
        for t in eachindex(s.c)
            g=findall(==(h),s.c)
            # For points already in the cluster, use an explicit duplicate
            # statistic rather than the member-list validator.
            c=MB.cluster(s,g);G.add!(c,s.model.y[t])
            @test MF.logpred(s,h,s.model.y[t])≈MT.log_marginal(s,c)-MT.log_marginal(s,s.clusters[h]) atol=2e-11
        end
    end
    fresh=deepcopy(s)
    for h in eachindex(s.clusters);MF.refresh!(fresh,h);end
    @test all(a≈b for (a,b) in zip(s.predL,fresh.predL))
    @test s.predmu≈fresh.predmu
    @test s.prednorm≈fresh.prednorm
end

@testset "MFM sequential split/merge" begin
    @testset "partial allocation endpoint density and normalization" begin
        s=fixture();D=[1,2,3,4];anchors=[1,2];order=[4,3]
        for beta in (0.,.6,1.)
            total=0.
            for allocation in Iterators.product(1:2,1:2)
                forced=zeros(Int,length(s.c));forced[anchors].=[1,2];forced[order].=collect(allocation)
                rng=Xoshiro(91);before=bytes(rng);original=bytes(s)
                groups,logq=S.sequential_split!(s,D,anchors,order,rng;forced,beta)
                # Independently rebuild each partial group from its members,
                # evaluating the NIW evidence difference rather than caches.
                partial=[[1],[2]];expected=0.
                for t in order
                    scores=[log(length(g)+s.gamma)+beta*(MT.log_marginal(s,vcat(g,t))-MT.log_marginal(s,g)) for g in partial]
                    top=maximum(scores);expected+=scores[forced[t]]-top-log(sum(exp.(scores.-top)))
                    push!(partial[forced[t]],t)
                end
                @test logq≈expected atol=2e-12
                @test groups==sort!.(partial)
                @test bytes(rng)==before
                @test bytes(s)==original
                total+=exp(logq)
            end
            @test total≈1. atol=2e-14
        end
        # There is no allocation draw when the union consists only of anchors.
        rng=Xoshiro(17);before=bytes(rng)
        @test S.sequential_split!(s,[1,2],[1,2],Int[],rng)==([[1],[2]],0.)
        @test bytes(rng)==before
        @test_throws ArgumentError S.sequential_split!(s,D,[1,1],order,rng)
        @test_throws ArgumentError S.sequential_split!(s,D,anchors,[3,3],rng)
        @test_throws ArgumentError S.sequential_split!(s,D,anchors,order,rng;forced=ones(Int,length(s.c)))
    end
    @testset "full target ratios and reverse route" begin
        for d in (1,2),beta in (0.,.6,1.),joint in (false,true),seed in 1:8
            s=fixture(d);w=S.Workspace(s.model);rng=Xoshiro(seed)
            before=bytes(s)
            p=S.propose!(w,s,rng;pair=(1,2),order=[4,3],joint,beta)
            @test bytes(s)==before
            @test p.logreverse==0.
            @test p.selectionterm==0.
            @test p.newM==s.M+Int(joint)
            @test S.raw_ratio(s,p).raw≈p.ratio atol=2e-11
            @test p.ratio≈p.Gterm+beta*p.H+p.countterm-p.logforward atol=1e-14
            new=deepcopy(s);p.outcome=:accepted;S.commit!(new,p)
            audit(new)
            reverse=S.propose!(w,new,Xoshiro(seed+99);pair=(1,2),order=[4,3],joint,beta)
            @test reverse.logforward==0.
            @test reverse.logreverse≈p.logforward atol=3e-12
            @test reverse.ratio≈-p.ratio atol=3e-11
            @test S.raw_ratio(new,reverse).raw≈reverse.ratio atol=3e-11
            reverse.outcome=:accepted;S.commit!(new,reverse)
            @test G.canonical_labels(new.c)==G.canonical_labels(s.c)
            @test new.M==s.M
            audit(new)
        end
    end
    @testset "atomic rejection, capacity, beta and RNG" begin
        s=fixture();w=S.Workspace(s.model);rng=Xoshiro(818)
        p=S.propose!(w,s,rng;pair=(1,5),order=[2,3,4,6,7],joint=true)
        @test p.kind==:local_joint_merge
        @test p.newM==s.M-1
        p.outcome=:rejected;before=bytes(s);S.commit!(s,p);@test bytes(s)==before
        p.outcome=:accepted;p.newgroups[1]=[1,1]
        @test_throws ErrorException S.commit!(s,p)
        @test bytes(s)==before
        s=fixture(M=3);before=bytes(s);rng=Xoshiro(13);rbefore=bytes(rng)
        p=S.propose!(w,s,rng;pair=(1,2))
        @test p.outcome==:capacity
        @test bytes(s)==before && bytes(rng)==rbefore
        p=S.propose!(w,s,rng;pair=(1,2),joint=true)
        @test p.outcome in (:accepted,:rejected) && p.newM==4
        for beta in (-.1,1.1,NaN,Inf)
            rng=Xoshiro(22);before=bytes(rng)
            @test_throws ArgumentError S.propose!(w,s,rng;beta)
            @test bytes(rng)==before
        end
        @test_throws ArgumentError S.propose!(w,s,rng;pair=(1,1))
        # High gamma at M=K is the application bottleneck; the joint count
        # normalization must still use all observations and unaffected groups.
        for gamma in (.1,2.6,3.7),beta in (0.,.6,1.)
            s=fixture(M=3);s.gamma=gamma
            p=S.propose!(w,s,Xoshiro(98);pair=(1,2),joint=true,beta)
            @test p.newM==4 && p.newk==4
            @test S.raw_ratio(s,p).raw≈p.ratio atol=3e-11
        end
        a=fixture();b=deepcopy(a);ra=Xoshiro(31);rb=deepcopy(ra)
        pa=S.attempt!(w,a,ra;joint=true)
        pb=S.attempt!(w,b,rb;joint=true,beta=1.)
        @test bytes(a)==bytes(b) && bytes(ra)==bytes(rb)
        @test bytes(pa)==bytes(pb)
        observed=Ref(false);a=fixture()
        S.attempt!(w,a,Xoshiro(33);observer=(state,p)->(observed[]=state.c==fixture().c))
        @test observed[]
        single=fixture(1;labels=[1],M=1);rng=Xoshiro(18);before=bytes(rng)
        @test S.attempt!(w,single,rng)===nothing
        @test bytes(rng)==before
    end
    @testset "shared stateless workspace is thread safe" begin
        states=[fixture() for _ in 1:8];serial=deepcopy(states)
        rngs=[Xoshiro(991+t) for t in eachindex(states)];srngs=deepcopy(rngs)
        w=S.Workspace()
        a=[S.attempt!(w,serial[t],srngs[t];joint=true,beta=.6) for t in eachindex(states)]
        b=Vector{Any}(undef,length(states))
        @sync for t in eachindex(states)
            Threads.@spawn b[t]=S.attempt!(w,states[t],rngs[t];joint=true,beta=.6)
        end
        @test all(bytes(states[t])==bytes(serial[t]) for t in eachindex(states))
        @test all(bytes(rngs[t])==bytes(srngs[t]) for t in eachindex(states))
        @test all(bytes(a[t])==bytes(b[t]) for t in eachindex(states))
    end
end

end
