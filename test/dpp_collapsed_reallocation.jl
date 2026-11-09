# Standalone: julia --project=. --threads=4 test/dpp_collapsed_reallocation.jl
if !isdefined(Main,:Benchmark);include("../src/execution/Benchmark.jl");end
module DPPCollapsedReallocationTests
using Test,GeneralDPPMixtures,LinearAlgebra,Random,StaticArrays,Statistics,Serialization
const G=GeneralDPPMixtures
const B=Main.Benchmark
const R=B.DPPCollapsedReallocation
const MR=B.DPPMultiReallocation
bytes(x)=begin io=IOBuffer();serialize(io,x);take!(io);end
function fixture(d=2)
    rng=Xoshiro(814+d);y=[SVector{d,Float64}(randn(rng,d)) for _ in 1:10]
    m=G.Model(y,ones(Int,10),fill(-5.,d),fill(5.,d),G.make_spectrum(d);a=.37,tau=d+2.,omega=.8Matrix{Float64}(I,d,d))
    c=[1,1,1,2,2,2,3,3,4,4]
    xs=[mean(y[findall(==(h),c)]) for h in 1:4]
    s=G.initialize(m,G.Initialization(c,xs,collect(1:12)),:marginal_collapsed,UInt64(953))
    fill!(s.workf,0.);fill!(s.workv,0.);fill!(s.workw,0.)
    m,s
end

@testset "Current-label adjoint collapsed DPP reallocation" begin
    @testset "full density ratios, reverse order and atomic cache commits" begin
        for d in (1,3),blocks in (0,2,3)
            m,s=fixture(d);w=R.Workspace(m;blocks);checked=0
            for seed in 1:24
                before=bytes(s);p=R.propose!(w,s,Xoshiro(seed))
                @test bytes(s)==before
                @test p.kind==:collapsed_reallocate && p.k==length(s.active) && p.M==length(s.S)
                @test length(p.oldgroups)==length(p.newgroups)==(blocks==0 ? 4 : blocks)
                @test p.component_logforward==[p.logforward] && p.component_logreverse==[p.logreverse]
                @test p.Gterm+p.H+p.logreverse-p.logforward≈0. atol=2e-9
                D=sort(vcat(p.oldgroups...));old=zeros(Int,length(s.c));new=copy(old)
                for h in eachindex(p.oldgroups);old[p.oldgroups[h]].=h;new[p.newgroups[h]].=h;end
                rng=Xoshiro(519);rngbefore=bytes(rng)
                _,qf=MR.restricted_scan!(m,D,p.anchors,p.order,rng,old;forced=new)
                _,qr=MR.restricted_scan!(m,D,p.anchors,reverse(p.order),rng,new;forced=old)
                @test qf==p.logforward && qr==p.logreverse && bytes(rng)==rngbefore
                p.ratio===nothing && continue
                @test p.ratio≈p.DK+p.selectionterm atol=2e-9
                unaffected=[s.clusters[h].x for h in s.active if !(h in p.oldids)]
                direct=R.raw_ratio(m,s.S,unaffected,p.oldgroups,p.oldxs,p.newgroups,p.newxs,p.logforward,p.logreverse)
                @test direct.raw≈p.ratio atol=3e-8
                reverse_density=R.raw_ratio(m,s.S,unaffected,p.newgroups,p.newxs,p.oldgroups,p.oldxs,p.logreverse,p.logforward)
                @test reverse_density.raw≈-p.ratio atol=3e-8
                candidate=deepcopy(s);accepted=deepcopy(p);accepted.outcome=:accepted
                R.commit!(candidate,accepted)
                @test candidate.S==s.S && candidate.active==s.active
                @test bytes(candidate.streams)==bytes(s.streams)
                @test B.SM.audit(candidate)["status"]=="passed"
                @test G.collapsed_logtarget(candidate)-G.collapsed_logtarget(s)≈direct.target_difference atol=3e-8
                @test p.moved_observations==sum(candidate.c.!=s.c)
                for h in setdiff(s.active,p.oldids)
                    @test bytes(candidate.clusters[h])==bytes(s.clusters[h])
                end
                rejected=deepcopy(p);rejected.outcome=:mh_rejected;R.commit!(s,rejected)
                @test bytes(s)==before
                if length(p.newxs)>1
                    bad=deepcopy(accepted);bad.newxs[2]=bad.newxs[1]
                    @test_throws ErrorException R.commit!(s,bad)
                    @test bytes(s)==before
                end
                checked+=1
            end
            @test checked>=20
        end
    end
    @testset "attempt observer, live transitions, validation and geometry lock" begin
        m,s=fixture();w=R.Workspace(m)
        a=deepcopy(s);b=deepcopy(s);ra=Xoshiro(174);rb=Xoshiro(174);observed=Ref(0)
        observer=(state,p)->begin
            @test state===b
            observed[]+=1
        end
        accepted=0;rejected=0;changed=0
        for _ in 1:100
            before=bytes(a);S=copy(a.S);active=copy(a.active)
            pa=R.attempt!(w,a,ra);pb=R.attempt!(w,b,rb;observer,geometry_lock=B.SM_LOCK)
            @test bytes(pa)==bytes(pb) && bytes(a)==bytes(b) && bytes(ra)==bytes(rb)
            @test a.S==S && a.active==active
            if pa.outcome==:accepted
                accepted+=1;changed+=pa.moved_observations>0
                @test B.SM.audit(a)["status"]=="passed"
            else
                rejected+=1;@test bytes(a)==before
            end
        end
        @test observed[]==100 && accepted>0 && rejected>0 && changed>0
        @test_throws ArgumentError R.Workspace(m;blocks=1)
        @test_throws ArgumentError R.propose!(w,s,Xoshiro(1);ids=[1,2])
        @test_throws ArgumentError R.propose!(w,s,Xoshiro(1);anchors=[1,2,7,9])
        @test_throws ArgumentError R.propose!(w,s,Xoshiro(1);order=Int[])
        before=bytes(s);other=MR.propose!(MR.Workspace(m),s,Xoshiro(15))
        @test_throws ArgumentError R.commit!(s,other)
        @test bytes(s)==before
    end
    @testset "shared immutable workspace, concurrent trajectories, state/RNG bytes" begin
        _,base=fixture();w=R.Workspace(base.model)
        function trajectory(seed)
            s=deepcopy(base);rng=Xoshiro(seed);rows=Any[]
            for _ in 1:36
                p=R.attempt!(w,s,rng;geometry_lock=B.SM_LOCK)
                push!(rows,(proposal=bytes(p),state=bytes(s),rng=bytes(rng)))
            end
            rows
        end
        seeds=[17,31,73,91,192,319]
        serial=trajectory.(seeds);parallel=Vector{Any}(undef,length(seeds))
        @sync for j in eachindex(seeds)
            Threads.@spawn parallel[j]=trajectory(seeds[j])
        end
        @test parallel==serial
    end
end
end
