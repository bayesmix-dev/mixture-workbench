# Standalone: julia --project=. test/dpp_multi_reallocation.jl
if !isdefined(Main,:Benchmark);include("../src/execution/Benchmark.jl");end
if !isdefined(Main.Benchmark,:DPPMultiReallocation)
    Base.include(Main.Benchmark,joinpath(@__DIR__,"../src/models/point_process/algorithms/DPPMultiReallocation.jl"))
end
module DPPMultiReallocationTests
using Test, GeneralDPPMixtures, Random, LinearAlgebra, StaticArrays, Statistics, Serialization, SpecialFunctions
const G=GeneralDPPMixtures
const SM=Main.Benchmark.SM
const R=Main.Benchmark.DPPMultiReallocation
function fixture(d=2)
    rng=Xoshiro(392+d)
    y=[SVector{d,Float64}(randn(rng,d)) for _ in 1:10]
    m=G.Model(y,ones(Int,10),fill(-5.,d),fill(5.,d),G.make_spectrum(d);
        a=.37,tau=d+2.,omega=.8Matrix{Float64}(I,d,d))
    labels=[1,1,1,2,2,2,3,3,4,4]
    xs=[G.uniform_point(rng,m) for _ in 1:4]
    s=G.initialize(m,G.Initialization(labels,xs,collect(1:12)),:marginal_collapsed,UInt64(321))
    refs=[copy(labels),[1,2,1,2,1,2,3,3,4,4],[1,1,2,2,3,3,4,4,1,2],ones(Int,10)]
    m,s,refs
end
function statebytes(s)
    io=IOBuffer();serialize(io,s);take!(io)
end
@testset "Exact multi-cluster DPP reallocation" begin
    @testset "frozen launches, reference mapping, mixture endpoint densities" begin
        m,s,refs=fixture();w=R.Workspace(m;references=refs)
        refs[1][1]=99
        @test w.references[1][1]==1
        D=collect(1:10);anchors=[1,4,7,9];order=[10,8,6,5,3,2]
        bank=R.launches(w,m,D,anchors)
        @test length(bank)==4 # default plus exactly three four-group references
        @test bank[2]==s.c
        @test bank[3]==w.references[2]
        @test bank[4]==bank[1] # duplicate reference labels at anchors use fallback
        @test all(initial->initial[anchors]==collect(1:4),bank)
        q=0.
        for bits in Iterators.product(ntuple(_->1:4,6)...)
            forced=zeros(Int,10);forced[anchors].=1:4;forced[order].=collect(bits)
            lp,components=R.mixture_logprob(m,D,anchors,order,Xoshiro(8),bank,forced)
            @test exp(lp)≈sum(exp,components)/length(bank) rtol=2e-13
            q+=exp(lp)
        end
        @test q≈1. atol=2e-12
        # Forced evaluation cannot advance the random stream.
        a=Xoshiro(9);b=deepcopy(a)
        R.mixture_logprob(m,D,anchors,order,a,bank,s.c)
        @test rand(a)==rand(b)
        @test_throws ArgumentError R.Workspace(m;blocks=1)
        @test_throws ArgumentError R.Workspace(m;references=[[1,2]])
        # Global split/merge consumers can hold two observations in one daughter.
        aa=[1,2,4,7,9];al=[1,1,2,3,4];oo=setdiff(D,aa)
        merge_bank=R.launches(w,m,D,aa;anchorlabels=al)
        @test all(initial->initial[aa]==al,merge_bank)
        gs,lp=R.restricted_scan!(m,D,aa,oo,Xoshiro(1),merge_bank[1];anchorlabels=al)
        @test 1 in gs[1] && 2 in gs[1] && isfinite(lp)
    end
    @testset "independent full target ratio, reversal, selection and atomicity" begin
        checked=0;changed=0
        for d in (1,2),blocks in (0,3)
            m,s,refs=fixture(d);w=R.Workspace(m;references=refs,blocks)
            for seed in 1:24
                before=statebytes(s);p=R.propose!(w,s,Xoshiro(seed))
                @test statebytes(s)==before
                r=blocks==0 ? 4 : blocks
                @test length(p.oldids)==r && p.kind==:multi_reallocate
                @test p.selectionterm≈log(prod(length.(p.oldgroups))/prod(length.(p.newgroups))) atol=1e-14
                @test all(h->p.anchors[h] in p.newgroups[h],1:r)
                p.ratio===nothing && continue
                untouched=[s.clusters[h].x for h in s.active if !(h in p.oldids)]
                direct=R.raw_ratio(m,s.S,untouched,p.oldgroups,p.oldxs,p.newgroups,p.newxs,p.logforward,p.logreverse)
                reverse=R.raw_ratio(m,s.S,untouched,p.newgroups,p.newxs,p.oldgroups,p.oldxs,p.logreverse,p.logforward)
                @test p.ratio≈direct.raw atol=3e-8
                @test p.ratio≈-reverse.raw atol=3e-8
                candidate=deepcopy(s);accepted=deepcopy(p);accepted.outcome=:accepted
                R.commit!(candidate,accepted)
                @test G.collapsed_logtarget(candidate)-G.collapsed_logtarget(s)≈direct.target_difference atol=3e-8
                @test SM.audit(candidate)["status"]=="passed"
                @test candidate.S==s.S && candidate.active==s.active
                @test p.moved_observations==sum(candidate.c.!=s.c)
                for id in setdiff(s.active,p.oldids)
                    @test all(getfield(candidate.clusters[id],field)==getfield(s.clusters[id],field) for field in fieldnames(typeof(s.clusters[id])))
                end
                D=sort(vcat(p.oldgroups...));bank=R.launches(w,m,D,p.anchors)
                forced=zeros(Int,length(s.c))
                for h in 1:r;forced[p.newgroups[h]].=h;end
                lf,components=R.mixture_logprob(m,D,p.anchors,p.order,Xoshiro(19),bank,forced)
                @test lf==p.logforward && components==p.component_logforward
                for h in 1:r;forced[p.oldgroups[h]].=h;end
                lr,components=R.mixture_logprob(m,D,p.anchors,p.order,Xoshiro(19),bank,forced)
                @test lr==p.logreverse && components==p.component_logreverse
                # Mixture probabilities differ from conditioning on the selected
                # launch; this assertion catches the most consequential shortcut.
                checked+=1;changed+=p.moved_observations>0
            end
        end
        @test checked>60 && changed>0
    end
    @testset "actual MH commits, rejection, observer and geometry lock" begin
        m,s,refs=fixture();w=R.Workspace(m;references=refs)
        rng=Xoshiro(721);accepted=0;rejected=0;moved=0
        for _ in 1:200
            before=statebytes(s);S=copy(s.S);active=copy(s.active)
            p=R.attempt!(w,s,rng)
            @test s.S==S && s.active==active
            if p.outcome==:accepted
                accepted+=1;moved+=p.moved_observations>0
                @test SM.audit(s)["status"]=="passed"
            else
                rejected+=1;@test statebytes(s)==before
            end
        end
        @test accepted>0 && rejected>0 && moved>0
        a=deepcopy(s);b=deepcopy(s);ra=Xoshiro(11);rb=Xoshiro(11)
        observed=Ref(0);observer=(_,p)->(observed[]+=1)
        for _ in 1:20
            pa=R.attempt!(w,a,ra)
            pb=R.attempt!(w,b,rb;observer,geometry_lock=ReentrantLock())
            @test all(getfield(pa,f)==getfield(pb,f) for f in fieldnames(typeof(pa)))
        end
        @test statebytes(a)==statebytes(b) && rand(ra)==rand(rb) && observed[]==20
    end
    @testset "one-step invariance from independent continuous posterior draws" begin
        # Enumerate all 25 canonical k=3 partitions of five observations. Their
        # cluster-size products differ (3 versus 4), testing the anchor-selection
        # correction as part of an independently sampled continuous target.
        y=SVector{1,Float64}.([-1.1,-.7,.1,.6,1.3])
        m=G.Model(y,ones(Int,5),[-2.],[2.],G.make_spectrum(1);
            a=.37,tau=3.,omega=reshape([.6],1,1))
        cases=Vector{Vector{Int}}()
        for labels in Iterators.product(ntuple(_->1:3,5)...)
            c=collect(labels)
            c==G.canonical_labels(c) && maximum(c)==3 && push!(cases,c)
        end
        @test length(cases)==25
        groups=[[findall(==(h),c) for h in 1:3] for c in cases]
        clusters=[[SM.cluster(m,g) for g in gs] for gs in groups]
        logs=[sum(SM.logA(m,c)+loggamma(m.a+c.n)-loggamma(m.a) for c in cs) for cs in clusters]
        weights=exp.(logs.-maximum(logs));S=[1,2,3]
        # Each feature column has squared norm 3; Hadamard bounds det(F'F)
        # by 3^3. The physical-volume factor cancels in rejection sampling.
        bound=G.envelope(m.spectrum,S)^3
        w=R.Workspace(m;references=[[1,1,1,2,3],[1,1,2,2,3]])
        N=1800;J=length(cases);before=zeros(N,J+3);after=similar(before)
        rng=Xoshiro(749131);transitionrng=Xoshiro(749132)
        function observe!(row,s)
            row.=0.;row[findfirst(==(G.canonical_labels(s.c)),cases)]=1.
            xs=[s.clusters[s.c[i]].x[1] for i in eachindex(s.c)]
            row[J+1]=mean(xs);row[J+2]=mean(abs2,xs);row[J+3]=xs[1]*xs[end]
        end
        for t in 1:N
            state=nothing
            for _ in 1:100000
                j=G.draw_weights(rng,weights)
                xs=[G.student_draw(rng,SM.location_law(m,c)...) for c in clusters[j]]
                all(x->G.inside(m,x),xs) || continue
                F=G.feature_matrix(m,S,xs); probability=det(F)^2/bound
                @assert 0<=probability<=1+1e-12
                rand(rng)<probability || continue
                state=G.initialize(m,G.Initialization(cases[j],xs,S),:marginal_collapsed,G.seedof(749133,t))
                break
            end
            state===nothing && error("independent posterior rejection sampler exhausted its cap")
            observe!(view(before,t,:),state)
            R.attempt!(w,state,transitionrng)
            observe!(view(after,t,:),state)
        end
        delta=after-before
        @test all(abs.(vec(mean(delta;dims=1))).<=6vec(std(delta;dims=1))/sqrt(N).+1/N)
    end
end
end
