# Standalone: julia --project=. test/dpp_reallocation.jl
if !isdefined(Main,:Benchmark)
    include("../src/execution/Benchmark.jl")
end
if !isdefined(Main.Benchmark,:DPPReallocation)
    Base.include(Main.Benchmark,joinpath(@__DIR__,"../src/models/point_process/algorithms/DPPReallocation.jl"))
end

module DPPReallocationTests
using Test, GeneralDPPMixtures, Random, LinearAlgebra, StaticArrays, Statistics
using SpecialFunctions, QuadGK, Serialization
const G=GeneralDPPMixtures
const SM=Main.Benchmark.SM
const R=Main.Benchmark.DPPReallocation

function fixture(d=2;unaffected=true)
    rng=Xoshiro(983+d)
    y=[SVector{d,Float64}(randn(rng,d)) for _ in 1:(unaffected ? 8 : 6)]
    m=G.Model(y,ones(Int,length(y)),fill(-4.,d),fill(4.,d),G.make_spectrum(d);
        a=.37,tau=d+2.,omega=.8Matrix{Float64}(I,d,d))
    labels=unaffected ? [1,1,2,2,1,2,3,3] : [1,1,2,2,1,2]
    xs=[G.uniform_point(rng,m) for _ in 1:maximum(labels)]
    s=G.initialize(m,G.Initialization(labels,xs,collect(1:12)),:marginal_collapsed,UInt64(421))
    m,s
end

function statebytes(s)
    io=IOBuffer(); serialize(io,s); take!(io)
end

function continuous_model()
    # Two real features, deliberately unpaired, and a nonunit physical volume.
    sp=G.Spectrum{1}(reshape([0,1],1,2),UInt8[0,1],[1,2],[1:1,2:2],[0,1],
        [.5,.5],[0.,0.],1.,2.,.5,1e-8,1.,4/3,0.,log(.25))
    G.Model([SVector(-.82),SVector(.49),SVector(1.23)],ones(Int,3),[-1.1],[1.6],sp;
        a=.37,tau=3.,omega=reshape([.6],1,1))
end

@testset "Exact fixed-k DPP reallocation" begin
    @testset "density ratio and forward/reverse endpoint laws" begin
        checked=0; changed=0
        for d in (1,2), unaffected in (false,true)
            m,s=fixture(d;unaffected); w=R.Workspace(m)
            for seed in 1:35
                before=statebytes(s)
                p=R.propose!(w,s,Xoshiro(seed);anchors=(1,3),order=[6,2,5,4])
                @test statebytes(s)==before
                @test p.oldids==[1,2] && p.union_size==6 && p.k==length(s.active)
                @test p.newgroups[1][1]==1 && 3 in p.newgroups[2]
                p.ratio===nothing && continue
                untouched=[s.clusters[h].x for h in s.active if !(h in p.oldids)]
                direct=R.raw_ratio(m,s.S,untouched,p.oldgroups,p.oldxs,p.newgroups,p.newxs,
                    p.logforward,p.logreverse)
                reverse=R.raw_ratio(m,s.S,untouched,p.newgroups,p.newxs,p.oldgroups,p.oldxs,
                    p.logreverse,p.logforward)
                @test p.ratio≈direct.raw atol=2e-8
                @test p.ratio≈-reverse.raw atol=2e-8
                # Also compare the full model target, including the fixed terms.
                candidate=deepcopy(s); accepted=deepcopy(p); accepted.outcome=:accepted
                R.commit!(candidate,accepted)
                @test G.collapsed_logtarget(candidate)-G.collapsed_logtarget(s)≈
                    direct.target_difference atol=2e-8
                @test SM.audit(candidate)["status"]=="passed"
                @test candidate.S==s.S && candidate.active==s.active
                for id in s.active[3:end]
                    @test all(getfield(candidate.clusters[id],field)==getfield(s.clusters[id],field)
                        for field in fieldnames(typeof(s.clusters[id])))
                end
                @test p.moved_observations==sum(candidate.c.!=s.c)
                # Replay both endpoint probabilities from the same launch/order.
                D=sort(vcat(p.oldgroups...)); forced=zeros(Int,length(s.c))
                for h in 1:2; forced[p.newgroups[h]].=h; end
                newgroups,lnew=SM.restricted_scan!(w.scan,m,D,1,3,p.order,Xoshiro(9);forced)
                for h in 1:2; forced[p.oldgroups[h]].=h; end
                oldgroups,lold=SM.restricted_scan!(w.scan,m,D,1,3,p.order,Xoshiro(9);forced)
                @test newgroups==p.newgroups && lnew==p.logforward
                @test oldgroups==p.oldgroups && lold==p.logreverse
                checked+=1; changed+=p.moved_observations>0
            end
        end
        @test checked>60 && changed>0
    end

    @testset "fixed-data whitened launch preserves the original target" begin
        base,_=fixture(2); A=[100. 50.; 0. .05]
        y=[SVector{2,Float64}(A*x) for x in base.y]
        m=G.Model(y,base.truth,[-1000.,-1.],[1000.,1.],base.spectrum;
            a=base.a,tau=base.tau,omega=A*base.omega*A')
        s=G.initialize(m,G.Initialization([1,1,2,2,1,2,3,3],
            [mean(y[[1,2,5]]),mean(y[[3,4,6]]),mean(y[[7,8]])],collect(1:12)),
            :marginal_collapsed,UInt64(411))
        plain=R.Workspace(m); white=R.Workspace(m;whiten=true)
        @test plain.launch_data===m.y
        @test mean(white.launch_data)≈zeros(2) atol=1e-12
        @test cov(reduce(hcat,white.launch_data);dims=2)≈Matrix{Float64}(I,2,2) atol=1e-12
        D=collect(1:6); order=[6,2,5,4]
        # Explicitly passing original data is byte/RNG-equivalent to the old
        # default restricted scan; the new option introduces no default draws.
        for seed in 1:12
            r1=Xoshiro(seed); r2=Xoshiro(seed)
            a,la=SM.restricted_scan!(SM.Workspace(m),m,D,1,3,order,r1)
            b,lb=SM.restricted_scan!(SM.Workspace(m),m,D,1,3,order,r2;launch_data=m.y)
            @test a==b && la==lb && rand(r1)==rand(r2)
        end
        @test any(SM.launch_partition(m.y,D,i,j)!=
            SM.launch_partition(white.launch_data,D,i,j) for i in D for j in D if i!=j)
        checked=0
        for seed in 1:30
            p=R.propose!(white,s,Xoshiro(seed);anchors=(1,3),order)
            p.ratio===nothing && continue
            untouched=[s.clusters[h].x for h in s.active if !(h in p.oldids)]
            raw=R.raw_ratio(m,s.S,untouched,p.oldgroups,p.oldxs,p.newgroups,p.newxs,
                p.logforward,p.logreverse)
            @test p.ratio≈raw.raw atol=2e-7
            forced=zeros(Int,length(y))
            for h in 1:2; forced[p.oldgroups[h]].=h; end
            _,reverse=SM.restricted_scan!(white.scan,m,D,1,3,order,Xoshiro(1);
                forced,launch_data=white.launch_data)
            @test reverse==p.logreverse
            checked+=1
        end
        @test checked>15
    end

    @testset "atomic transitions, no-ops, spectrum and caches" begin
        m,s=fixture(); w=R.Workspace(m); rng=Xoshiro(591)
        before=statebytes(s); other=deepcopy(rng)
        p=R.attempt!(w,s,rng;anchors=(1,2))
        @test p.outcome==:same_cluster && statebytes(s)==before
        @test rand(rng)==rand(other)
        @test_throws ArgumentError R.attempt!(w,s,rng;anchors=(1,1))
        @test_throws ArgumentError R.attempt!(w,s,rng;anchors=(0,3))
        @test_throws ErrorException R.propose!(w,s,rng;anchors=(1,3),order=[2])
        observed=Ref(0); accepted=0; rejected=0; relocated=0
        observer=(_,p)->(observed[]+=1)
        for _ in 1:250
            before=statebytes(s); spectrum=copy(s.S); active=copy(s.active)
            p=R.attempt!(w,s,rng;observer)
            @test s.S==spectrum && s.active==active
            if p.outcome==:accepted
                accepted+=1; relocated+=p.moved_observations>0
                @test SM.audit(s)["status"]=="passed"
            else
                rejected+=1
                @test statebytes(s)==before
            end
        end
        @test accepted>0 && rejected>0 && relocated>0 && observed[]==250
        # Observer instrumentation cannot alter transition probabilities/RNG use.
        a=deepcopy(s); b=deepcopy(s); ra=Xoshiro(11); rb=Xoshiro(11)
        for _ in 1:20
            R.attempt!(R.Workspace(m),a,ra)
            R.attempt!(R.Workspace(m),b,rb;observer)
        end
        @test statebytes(a)==statebytes(b) && rand(ra)==rand(rb)
        # Narrow geometry locking changes neither proposals nor random draws.
        a=deepcopy(s); b=deepcopy(s); ra=Xoshiro(12); rb=Xoshiro(12)
        geometry_lock=ReentrantLock()
        for _ in 1:40
            pa=R.attempt!(R.Workspace(m),a,ra)
            pb=R.attempt!(R.Workspace(m),b,rb;geometry_lock)
            @test all(getfield(pa,field)==getfield(pb,field) for field in fieldnames(typeof(pa)))
        end
        @test statebytes(a)==statebytes(b) && rand(ra)==rand(rb)
        tiny=G.Model(m.y[1:1],[1],m.lower,m.upper,m.spectrum;a=m.a,tau=m.tau,omega=m.omega)
        singleton=G.initialize(tiny,G.Initialization([1],[tiny.y[1]],[1,2]),:marginal_collapsed,UInt64(1))
        before=statebytes(singleton); other=deepcopy(rng)
        @test R.attempt!(R.Workspace(tiny),singleton,rng)===nothing
        @test statebytes(singleton)==before && rand(rng)==rand(other)
    end

    @testset "independent continuous fixed-k posterior invariance" begin
        # Draw iid states from the exact fixed-spectrum/k=2 posterior by rejection
        # from an independently bounded envelope. Thus the one-step check needs
        # no MCMC convergence assumption. Quadrature provides separate partition
        # probabilities using scalar marginal likelihoods and a 2x2 determinant.
        m=continuous_model(); partitions=[[1,2,1],[1,2,2],[1,1,2]]
        groups=[[findall(==(h),c) for h in 1:2] for c in partitions]
        lo=m.lower[1]; hi=m.upper[1]; V=hi-lo
        f(x)=sqrt(2)*cospi(2*(x-lo)/V)
        loglik(ids,x)=-length(ids)/2*log(pi)+m.tau/2*log(.6)-
            (m.tau+length(ids))/2*log(.6+sum((m.y[i][1]-x)^2 for i in ids))+
            loggamma((m.tau+length(ids))/2)-loggamma(m.tau/2)
        exact=[begin
            prior=exp(sum(loggamma(m.a+length(g))-loggamma(m.a) for g in gs))
            prior*quadgk(x->quadgk(y->exp(loglik(gs[1],x)+loglik(gs[2],y))*
                (f(x)-f(y))^2/V^2,lo,hi;rtol=1e-8)[1],lo,hi;rtol=1e-8)[1]
        end for gs in groups]
        exact./=sum(exact)
        clusters=[[SM.cluster(m,g) for g in gs] for gs in groups]
        logs=[sum(loggamma(m.a+c.n)-loggamma(m.a)+SM.logA(m,c) for c in cs) for cs in clusters]
        weights=exp.(logs.-maximum(logs)); rng=Xoshiro(821130); transitionrng=Xoshiro(821131)
        N=2500; before=zeros(N,6); after=similar(before); w=R.Workspace(m)
        function observe!(row,s)
            index=findfirst(==(G.canonical_labels(s.c)),partitions)
            row[1:3].=0.; row[index]=1.
            xs=[s.clusters[s.c[i]].x[1] for i in 1:3]
            row[4]=sum(xs)/3; row[5]=sum(abs2,xs)/3; row[6]=xs[1]*xs[3]
        end
        for t in 1:N
            state=nothing
            for _ in 1:100000
                j=G.draw_weights(rng,weights)
                xs=[G.student_draw(rng,SM.location_law(m,c)...) for c in clusters[j]]
                all(x->G.inside(m,x),xs) || continue
                # For features (1,f), det(K)=(f(x)-f(y))^2/V^2 <= 9/V^2.
                probability=(f(xs[1][1])-f(xs[2][1]))^2/9
                @assert 0<=probability<=1
                rand(rng)<probability || continue
                state=G.initialize(m,G.Initialization(partitions[j],xs,[1,2]),:marginal_collapsed,UInt64(991))
                break
            end
            state===nothing && error("independent reference sampler exhausted its cap")
            observe!(view(before,t,:),state)
            R.attempt!(w,state,transitionrng)
            observe!(view(after,t,:),state)
        end
        delta=after-before; difference=vec(mean(delta;dims=1)); se=vec(std(delta;dims=1))/sqrt(N)
        # Six-SE criteria and finite-count allowance fixed before execution.
        @test all(abs.(difference).<=6se.+1/N)
        for j in 1:3
            bound=6sqrt(exact[j]*(1-exact[j])/N)+1/N
            @test abs(mean(before[:,j])-exact[j])<=bound
            @test abs(mean(after[:,j])-exact[j])<=bound
        end
    end

    @testset "local plus split-merge plus reallocation joint invariance" begin
        # Include every nonempty spectrum of the two-feature model, both k=1
        # and k=2, so this checks the actual production composition rather than
        # only a fixed-k conditional. The envelope draws are iid joint targets.
        m=continuous_model(); lo=m.lower[1]; hi=m.upper[1]; V=hi-lo
        f(x)=sqrt(2)*cospi(2*(x-lo)/V)
        loglik(ids,x)=-length(ids)/2*log(pi)+m.tau/2*log(.6)-
            (m.tau+length(ids))/2*log(.6+sum((m.y[i][1]-x)^2 for i in ids))+
            loggamma((m.tau+length(ids))/2)-loggamma(m.tau/2)
        cases=[(c,S) for c in ([1,2,1],[1,2,2],[1,1,2],[1,1,1])
            for S in ([1],[2],[1,2]) if maximum(c)<=length(S)]
        groups=[[findall(==(h),c) for h in 1:maximum(c)] for (c,_) in cases]
        clusters=[[SM.cluster(m,g) for g in gs] for gs in groups]
        bounds=[sum(j==1 ? 1. : 2. for j in S)/V for (_,S) in cases]
        function determinant(S,xs)
            length(xs)==2 && return (f(xs[1][1])-f(xs[2][1]))^2/V^2
            sum(j==1 ? 1. : f(xs[1][1])^2 for j in S)/V
        end
        logs=[begin
            c,S=cases[j]; M=length(S)
            loggamma(m.a*M)-loggamma(length(m.y)+m.a*M)+
                sum(loggamma(m.a+cc.n)-loggamma(m.a)+SM.logA(m,cc) for cc in clusters[j])+
                length(groups[j])*log(bounds[j])
        end for j in eachindex(cases)]
        weights=exp.(logs.-maximum(logs))
        exact=[begin
            gs=groups[j]; _,S=cases[j]; M=length(S)
            prior=exp(loggamma(m.a*M)-loggamma(length(m.y)+m.a*M)+
                sum(loggamma(m.a+length(g))-loggamma(m.a) for g in gs))
            if length(gs)==1
                prior*quadgk(x->exp(loglik(gs[1],x))*determinant(S,[SVector(x)]),lo,hi;rtol=1e-9)[1]
            else
                prior*quadgk(x->quadgk(y->exp(loglik(gs[1],x)+loglik(gs[2],y))*
                    determinant(S,[SVector(x),SVector(y)]),lo,hi;rtol=1e-8)[1],lo,hi;rtol=1e-8)[1]
            end
        end for j in eachindex(cases)]
        exact./=sum(exact)
        N=2000; J=length(cases); before=zeros(N,J+3); after=similar(before)
        rng=Xoshiro(841130); transitionrng=Xoshiro(841131)
        reallocation=R.Workspace(m); splitmerge=SM.Workspace(m); geometry_lock=ReentrantLock()
        function observe_joint!(row,s)
            index=findfirst(==((G.canonical_labels(s.c),s.S)),cases)
            index===nothing && error("joint reference omitted a possible state")
            row[1:J].=0.; row[index]=1.
            xs=[s.clusters[s.c[i]].x[1] for i in eachindex(s.c)]
            row[J+1]=sum(xs)/length(xs); row[J+2]=sum(abs2,xs)/length(xs)
            row[J+3]=xs[1]*xs[3]
        end
        for t in 1:N
            state=nothing
            for _ in 1:100000
                j=G.draw_weights(rng,weights); c,S=cases[j]
                xs=[G.student_draw(rng,SM.location_law(m,cc)...) for cc in clusters[j]]
                all(x->G.inside(m,x),xs) || continue
                probability=determinant(S,xs)/bounds[j]^length(xs)
                @assert -1e-14<=probability<=1+1e-14
                rand(rng)<probability || continue
                # Give every iid draw independent local-kernel streams as well.
                state=G.initialize(m,G.Initialization(c,xs,S),:marginal_collapsed,G.seedof(841132,t))
                break
            end
            state===nothing && error("joint reference sampler exhausted its cap")
            observe_joint!(view(before,t,:),state)
            G.marginal_collapsed!(state)
            lock(geometry_lock) do; SM.attempt!(splitmerge,state,transitionrng); end
            R.attempt!(reallocation,state,transitionrng;geometry_lock)
            observe_joint!(view(after,t,:),state)
        end
        delta=after-before; difference=vec(mean(delta;dims=1)); se=vec(std(delta;dims=1))/sqrt(N)
        @test all(abs.(difference).<=6se.+1/N)
        for j in 1:J
            bound=6sqrt(exact[j]*(1-exact[j])/N)+1/N
            @test abs(mean(before[:,j])-exact[j])<=bound
            @test abs(mean(after[:,j])-exact[j])<=bound
        end
    end
end

end
