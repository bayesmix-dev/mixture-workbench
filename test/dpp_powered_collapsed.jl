# Standalone: julia --project=. --threads=4 test/dpp_powered_collapsed.jl
# Toy density/transition checks only; no application data or application chains.
if !isdefined(Main,:Benchmark);include("../src/Benchmark.jl");end
module DPPPoweredCollapsedTests
using Test,GeneralDPPMixtures,LinearAlgebra,Random,StaticArrays,Statistics,Serialization,SpecialFunctions,QuadGK
const G=GeneralDPPMixtures
const B=Main.Benchmark
const R=B.DPPCollapsedReallocation
const SM=B.DPPSplitMerge
const MR=B.DPPMultiReallocation
const LT=B.DPPLikelihoodTempering
bytes(x)=begin io=IOBuffer();serialize(io,x);take!(io);end
function fixture(d=1)
    rng=Xoshiro(814+d);y=[SVector{d,Float64}(randn(rng,d)) for _ in 1:10]
    om=Matrix{Float64}(I,d,d)+.1ones(d,d)
    m=G.Model(y,ones(Int,10),fill(-6.,d),fill(6.,d),G.make_spectrum(d);a=.37,tau=d+2.,omega=om)
    c=[1,1,1,2,2,2,3,3,4,4];xs=[mean(y[findall(==(h),c)]) for h in 1:4]
    s=G.initialize(m,G.Initialization(c,xs,collect(1:12)),:marginal_collapsed,UInt64(953))
    fill!(s.workf,0.);fill!(s.workv,0.);fill!(s.workw,0.)
    m,s
end
function groups(labels,r=maximum(labels))
    [findall(==(h),labels) for h in 1:r]
end
@testset "Likelihood-powered current-label DPP reallocation" begin
    @testset "analytic density and numerical integral" begin
        for d in 1:3
            m,_=fixture(d)
            for members in ([1],[1,2,3],collect(1:6)),beta in (.6,.92,1.)
                c=SM.cluster(m,members)
                for offset in (-.6,.1,.8)
                    x=c.mean.+offset;cc=SM.cluster(m,members,x)
                    @test beta*G.log_marginal(cc,m)≈R.logA_beta(m,c,beta)+G.student_logdensity(x,R.location_law_beta(m,c,beta)...) atol=3e-12
                end
                if beta==1.
                    @test R.logA_beta(m,c,beta)==SM.logA(m,c)
                    @test R.location_law_beta(m,c,beta)==SM.location_law(m,c)
                end
                if d==1
                    integral,err=quadgk(x->exp(beta*G.log_marginal(SM.cluster(m,members,SVector(x)),m)),-Inf,Inf;rtol=2e-10)
                    @test log(integral)≈R.logA_beta(m,c,beta) atol=2e-9
                end
            end
        end
        m,s=fixture(3);single=SM.cluster(m,[1]);rng=Xoshiro(31)
        for beta in (0.,.5,-.1,1.1,NaN,Inf)
            before=bytes(rng);statebefore=bytes(s)
            @test_throws ArgumentError R.propose!(R.Workspace(m),s,rng;beta)
            @test bytes(rng)==before && bytes(s)==statebefore
        end
        @test_throws ArgumentError R.logA_beta(m,single,.5)
        @test_throws ArgumentError R.location_law_beta(m,single,.5)
    end
    @testset "unsimplified full LT target, adjoint and atomic caches" begin
        for d in (1,3),beta in (.6,.92,1.),blocks in (0,2)
            m,s=fixture(d);w=R.Workspace(m;blocks);checked=0
            for seed in 1:64
                before=bytes(s);p=R.propose!(w,s,Xoshiro(seed);beta)
                @test bytes(s)==before
                @test p.kind==:collapsed_reallocate && p.M==length(s.S) && p.k==length(s.active)
                @test p.Gterm+p.H+p.logreverse-p.logforward≈0. atol=3e-9
                D=sort(vcat(p.oldgroups...));old=zeros(Int,length(s.c));new=copy(old)
                for h in eachindex(p.oldgroups);old[p.oldgroups[h]].=h;new[p.newgroups[h]].=h;end
                rng=Xoshiro(135);rngbefore=bytes(rng)
                _,qf=R.restricted_scan_beta!(m,D,p.anchors,p.order,rng,old;forced=new,beta)
                _,qr=R.restricted_scan_beta!(m,D,p.anchors,reverse(p.order),rng,new;forced=old,beta)
                @test qf==p.logforward && qr==p.logreverse && bytes(rng)==rngbefore
                p.ratio===nothing && continue
                @test p.ratio≈p.DK+p.selectionterm atol=3e-9
                unaffected=[s.clusters[h].x for h in s.active if !(h in p.oldids)]
                raw=R.raw_ratio(m,s.S,unaffected,p.oldgroups,p.oldxs,p.newgroups,p.newxs,p.logforward,p.logreverse;beta)
                @test raw.raw≈p.ratio atol=4e-8
                revraw=R.raw_ratio(m,s.S,unaffected,p.newgroups,p.newxs,p.oldgroups,p.oldxs,p.logreverse,p.logforward;beta)
                @test revraw.raw≈-p.ratio atol=4e-8
                candidate=deepcopy(s);accepted=deepcopy(p);accepted.outcome=:accepted
                R.commit!(candidate,accepted)
                @test candidate.S==s.S && candidate.active==s.active
                @test bytes(candidate.streams)==bytes(s.streams)
                @test B.SM.audit(candidate)["status"]=="passed"
                @test LT.logtarget(candidate,beta)-LT.logtarget(s,beta)≈raw.target_difference atol=4e-8
                for h in setdiff(s.active,p.oldids)
                    @test bytes(candidate.clusters[h])==bytes(s.clusters[h])
                end
                rejected=deepcopy(p);rejected.outcome=:mh_rejected;R.commit!(s,rejected)
                @test bytes(s)==before
                bad=deepcopy(accepted);bad.newxs[2]=bad.newxs[1]
                @test_throws ErrorException R.commit!(s,bad)
                @test bytes(s)==before
                checked+=1
                checked==8 && break
            end
            @test checked==8
        end
    end
    @testset "cold trajectory and powered concurrency" begin
        m,s=fixture();w=R.Workspace(m);a=deepcopy(s);b=deepcopy(s);c=deepcopy(s)
        ra=Xoshiro(837);rb=deepcopy(ra);rc=deepcopy(ra)
        for _ in 1:20
            # cold_propose! is the unmodified pre-extension body.
            old=R.cold_propose!(w,a,ra);R.commit!(a,old)
            default=R.attempt!(w,b,rb);explicit=R.attempt!(w,c,rc;beta=1.)
            @test bytes(old)==bytes(default)==bytes(explicit)
            @test bytes(a)==bytes(b)==bytes(c)
            @test bytes(ra)==bytes(rb)==bytes(rc)
        end
        for beta in (.6,.92)
            states=[deepcopy(s) for _ in 1:6];serial=deepcopy(states)
            rngs=[Xoshiro(519+t) for t in eachindex(states)];srngs=deepcopy(rngs)
            first=[R.attempt!(w,serial[t],srngs[t];beta,geometry_lock=B.SM_LOCK) for t in eachindex(states)]
            parallel=Vector{Any}(undef,length(states))
            @sync for t in eachindex(states)
                Threads.@spawn parallel[t]=R.attempt!(w,states[t],rngs[t];beta,geometry_lock=B.SM_LOCK)
            end
            @test bytes(states)==bytes(serial)
            @test bytes(rngs)==bytes(srngs)
            @test all(bytes(first[t])==bytes(parallel[t]) for t in eachindex(states))
        end
        # Outside-domain proposals are single draws and leave the state intact.
        m0,s0=fixture();tight=G.Model(m0.y,m0.truth,[-.15],[.15],m0.spectrum;a=m0.a,tau=m0.tau,omega=m0.omega)
        xs=[SVector(x) for x in (-.12,-.04,.04,.12)]
        tightstate=G.initialize(tight,G.Initialization(s0.c,xs,s0.S),:marginal_collapsed,UInt64(30))
        before=bytes(tightstate);p=R.attempt!(R.Workspace(tight),tightstate,Xoshiro(1);beta=.6)
        @test p.outcome==:outside_domain
        @test bytes(tightstate)==before
    end
end

end
