# Standalone: julia --project=. test/dpp_likelihood_tempering.jl
if !isdefined(Main,:Benchmark);include("../src/execution/Benchmark.jl");end
if !isdefined(Main.Benchmark,:DPPLikelihoodTempering)
    Base.include(Main.Benchmark,joinpath(@__DIR__,"../src/models/point_process/algorithms/DPPLikelihoodTempering.jl"))
end
module DPPLikelihoodTemperingTests
using Test, GeneralDPPMixtures, LinearAlgebra, Random, StaticArrays, Statistics, Serialization, SpecialFunctions, QuadGK
const G=GeneralDPPMixtures
const SM=Main.Benchmark.SM
const T=Main.Benchmark.DPPLikelihoodTempering
function statebytes(s)
    io=IOBuffer();serialize(io,s);take!(io)
end
function continuous_model()
    sp=G.Spectrum{1}(reshape([0,1],1,2),UInt8[0,1],[1,2],[1:1,2:2],[0,1],
        [.5,.5],[0.,0.],1.,2.,.5,1e-8,1.,4/3,0.,log(.25))
    G.Model([SVector(-.82),SVector(.49),SVector(1.23)],ones(Int,3),[-1.1],[1.6],sp;
        a=.37,tau=3.,omega=reshape([.6],1,1))
end
@testset "Exact collapsed-likelihood DPP tempering" begin
    @testset "predictive factors, powered Student normalization, envelopes" begin
        for d in (1,2,3)
            rng=Xoshiro(114+d);y=[SVector{d,Float64}(randn(rng,d)) for _ in 1:8]
            m=G.Model(y,ones(Int,8),fill(-5.,d),fill(5.,d),G.make_spectrum(d);
                a=.37,tau=d+2.,omega=.8Matrix{Float64}(I,d,d))
            c=SM.cluster(m,[1,2,3],y[2])
            for i in 4:8
                old=G.log_marginal(c,m);pred=G.predictive_mass(c,m,y[i])-log(c.n+m.a)
                aug=deepcopy(c);G.add!(aug,y[i]);G.refresh!(aug,m)
                @test pred≈G.log_marginal(aug,m)-old atol=2e-12
            end
            for beta in (0.,.01,.3,.7,1.)
                law=T.new_location_envelope(m,beta)
                singleton=SM.cluster(m,[1],y[1])
                @test law.logmax≈G.log_marginal(singleton,m) atol=2e-12
                @test law.offset<=beta*law.logmax+1e-12
                if law.mode==:student
                    for x in (y[1],y[2],y[3])
                        cc=SM.cluster(m,[1],x)
                        logq=G.student_logdensity(x,y[1],m.omegaL,law.df)
                        @test beta*G.log_marginal(cc,m)-logq≈law.logZ atol=3e-12
                    end
                end
                for ids in ([1],[1,2,3])
                    cc=SM.cluster(m,ids);df=beta*(m.tau+cc.n)-d
                    df>0 || continue
                    factor=cholesky(Symmetric((m.omega+cc.scatter)/cc.n)).L
                    constants=Float64[]
                    for x in (cc.mean,y[2],y[4])
                        cand=SM.cluster(m,ids,x)
                        push!(constants,beta*G.log_marginal(cand,m)-G.student_logdensity(x,cc.mean,factor,df))
                    end
                    @test maximum(constants)-minimum(constants)<4e-12
                end
            end
            @test T.new_location_envelope(m,0.).mode==:uniform
            # A tiny domain can make the box envelope tighter even if the
            # untruncated powered Student is proper.
            tiny=G.Model(y,ones(Int,8),fill(-.001,d),fill(.001,d),m.spectrum;
                a=m.a,tau=m.tau,omega=m.omega)
            @test T.new_location_envelope(tiny,.99).mode==:uniform
        end
    end
    @testset "cold byte equivalence, target powers and swap ratios" begin
        m=continuous_model();s=G.initialize(m,G.Initialization([1,2,1],[SVector(-.5),SVector(.8)],[1,2]),:marginal_collapsed,UInt64(338))
        a=deepcopy(s);b=deepcopy(s)
        for _ in 1:25
            G.marginal_collapsed!(a);T.sweep!(b,1.)
            @test statebytes(a)==statebytes(b)
        end
        for beta in (0.,.3,.7,1.)
            @test T.logtarget(s,beta)-T.logtarget(s,0.)≈beta*T.loglikelihood(s) atol=2e-12
            @test T.swap_logratio(s,a,beta,1.)≈T.logtarget(a,beta)+T.logtarget(s,1.)-T.logtarget(s,beta)-T.logtarget(a,1.) atol=2e-12
            @test T.swap_logratio(s,a,beta,beta)==0.
            @test T.swap_logratio(s,a,beta,1.)≈-T.swap_logratio(a,s,beta,1.) atol=2e-12
        end
        @test_throws ArgumentError T.sweep!(s,-.01)
        @test_throws ArgumentError T.sweep!(s,1.01)
        @test_throws ArgumentError T.sweep!(s,NaN)
    end
    @testset "threaded hot location proposals preserve per-cluster streams" begin
        if Threads.nthreads(:default)>=2 && Sys.CPU_THREADS>=2
            rng=Xoshiro(4201);y=[SVector{2,Float64}(randn(rng,2)) for _ in 1:12]
            m=G.Model(y,ones(Int,12),[-4.,-4.],[4.,4.],G.make_spectrum(2);
                a=.37,tau=4.,omega=.8Matrix{Float64}(I,2,2))
            labels=repeat(collect(1:3);inner=4)
            xs=[mean(y[findall(==(h),labels)]) for h in 1:3]
            initial=G.initialize(m,G.Initialization(labels,xs,collect(1:12)),:marginal_collapsed,UInt64(4202))
            for beta in (0.,.3,.7)
                a=deepcopy(initial);b=deepcopy(initial)
                # Scratch has no statistical content. Different previous heap
                # contents must not affect transitions or serialized outputs.
                fill!(a.workf,17.);fill!(b.workf,-19.)
                fill!(a.workv,31.);fill!(b.workv,-37.)
                for _ in 1:100
                    T.sweep!(a,beta,G.Execution())
                    T.sweep!(b,beta,G.Execution(:within,2))
                    @test statebytes(a)==statebytes(b)
                end
            end
        else
            @test true # The standalone two-thread invocation exercises this path.
        end
    end
    @testset "independent powered-posterior invariance at beta 0, .3, .7, 1" begin
        # All nonempty spectra and partitions of a two-feature model. Exact
        # scalar likelihoods, determinant formulas and bounded iid rejection
        # draws provide a reference independent of the transition implementation.
        m=continuous_model();V=m.volume;lo=m.lower[1];hi=m.upper[1]
        f(x)=sqrt(2)*cospi(2*(x-lo)/V)
        loglik(ids,x)=-length(ids)/2*log(pi)+m.tau/2*log(.6)-
            (m.tau+length(ids))/2*log(.6+sum((m.y[i][1]-x)^2 for i in ids))+
            loggamma((m.tau+length(ids))/2)-loggamma(m.tau/2)
        cases=[(c,S) for c in ([1,2,1],[1,2,2],[1,1,2],[1,1,1])
            for S in ([1],[2],[1,2]) if maximum(c)<=length(S)]
        groups=[[findall(==(h),c) for h in 1:maximum(c)] for (c,_) in cases]
        bounds=[sum(j==1 ? 1. : 2. for j in S)/V for (_,S) in cases]
        maxima=[[loglik(g,mean(m.y[i][1] for i in g)) for g in gs] for gs in groups]
        function determinant(S,xs)
            length(xs)==2 && return (f(xs[1])-f(xs[2]))^2/V^2
            sum(j==1 ? 1. : f(xs[1])^2 for j in S)/V
        end
        N=2200;J=length(cases)
        for (bi,beta) in enumerate((0.,.3,.7,1.))
            logs=[begin
                c,S=cases[j];M=length(S);k=maximum(c)
                loggamma(m.a*M)-loggamma(length(m.y)+m.a*M)+
                    sum(loggamma(m.a+length(g))-loggamma(m.a) for g in groups[j])+
                    beta*sum(maxima[j])+k*log(V*bounds[j])
            end for j in eachindex(cases)]
            weights=exp.(logs.-maximum(logs))
            exact=[begin
                c,S=cases[j];M=length(S);gs=groups[j]
                prior=exp(loggamma(m.a*M)-loggamma(length(m.y)+m.a*M)+
                    sum(loggamma(m.a+length(g))-loggamma(m.a) for g in gs))
                if length(gs)==1
                    prior*quadgk(x->exp(beta*loglik(gs[1],x))*determinant(S,[x]),lo,hi;rtol=1e-9)[1]
                else
                    prior*quadgk(x->quadgk(y->exp(beta*(loglik(gs[1],x)+loglik(gs[2],y)))*
                        determinant(S,[x,y]),lo,hi;rtol=1e-8)[1],lo,hi;rtol=1e-8)[1]
                end
            end for j in eachindex(cases)]
            exact./=sum(exact)
            before=zeros(N,J+3);after=similar(before);rng=Xoshiro(341130+bi)
            function observe!(row,s)
                j=findfirst(==((G.canonical_labels(s.c),s.S)),cases)
                row.=0.;row[j]=1.
                xs=[s.clusters[s.c[i]].x[1] for i in eachindex(s.c)]
                row[J+1]=mean(xs);row[J+2]=mean(abs2,xs);row[J+3]=xs[1]*xs[end]
            end
            for t in 1:N
                state=nothing
                for _ in 1:100000
                    j=G.draw_weights(rng,weights);c,S=cases[j];k=maximum(c)
                    xs=[lo+V*rand(rng) for _ in 1:k]
                    ll=sum(loglik(g,x) for (g,x) in zip(groups[j],xs))
                    probability=exp(beta*(ll-sum(maxima[j])))*determinant(S,xs)/bounds[j]^k
                    @assert -1e-14<=probability<=1+1e-12
                    rand(rng)<probability || continue
                    state=G.initialize(m,G.Initialization(c,SVector{1,Float64}.(xs),S),:marginal_collapsed,G.seedof(341131,bi,t))
                    break
                end
                state===nothing && error("independent powered posterior sampler exhausted cap")
                observe!(view(before,t,:),state)
                T.sweep!(state,beta)
                observe!(view(after,t,:),state)
                @test SM.audit(state)["status"]=="passed"
            end
            delta=after-before;difference=vec(mean(delta;dims=1));se=vec(std(delta;dims=1))/sqrt(N)
            @test all(abs.(difference).<=6se.+1/N)
            for j in 1:J
                bound=6sqrt(exact[j]*(1-exact[j])/N)+1/N
                @test abs(mean(before[:,j])-exact[j])<=bound
                @test abs(mean(after[:,j])-exact[j])<=bound
            end
        end
    end
    @testset "positive-beta uniform-location fallback in two dimensions" begin
        sp=G.Spectrum{2}([0 1;0 0],UInt8[0,1],[1,2],[1:1,2:2],[0,1],
            [.5,.5],[0.,0.],1.,2.,.5,1e-8,1.,4/3,0.,log(.25))
        m=G.Model([SVector(-.6,-.3),SVector(.4,.8)],ones(Int,2),[-1.1,-1.1],[1.6,1.6],sp;
            a=.37,tau=4.,omega=.6Matrix{Float64}(I,2,2))
        beta=.3;V=m.volume;span=m.upper-m.lower
        @test beta*(m.tau+2)-2<0 # every location proposal uses the uniform fallback
        lgamma2(a)=.5log(pi)+loggamma(a)+loggamma(a-.5)
        function loglik(g,x)
            R=.6Matrix{Float64}(I,2,2)
            for i in g
                delta=m.y[i]-x;R.+=delta*delta'
            end
            n=length(g)
            -n*log(pi)+m.tau*log(.6)-(m.tau+n)/2*log(det(R))+
                lgamma2((m.tau+n)/2)-lgamma2(m.tau/2)
        end
        f(x)=sqrt(2)*cospi(2*(x[1]-m.lower[1])/span[1])
        function determinant(S,xs)
            length(xs)==2 && return (f(xs[1])-f(xs[2]))^2/V^2
            sum(j==1 ? 1. : f(xs[1])^2 for j in S)/V
        end
        cases=[([1,1],[1]),([1,1],[2]),([1,1],[1,2]),([1,2],[1,2])]
        groups=[[findall(==(h),c) for h in 1:maximum(c)] for (c,_) in cases]
        maxima=[[loglik(g,mean(m.y[g])) for g in gs] for gs in groups]
        bounds=[sum(j==1 ? 1. : 2. for j in S)/V for (_,S) in cases]
        logs=[begin
            c,S=cases[j];M=length(S);k=maximum(c)
            loggamma(m.a*M)-loggamma(2+m.a*M)+
                sum(loggamma(m.a+length(g))-loggamma(m.a) for g in groups[j])+
                beta*sum(maxima[j])+k*log(V*bounds[j])
        end for j in eachindex(cases)]
        weights=exp.(logs.-maximum(logs));N=1600;J=length(cases)
        before=zeros(N,J+3);after=similar(before);rng=Xoshiro(342131)
        function observe!(row,s)
            row.=0.;row[findfirst(==((G.canonical_labels(s.c),s.S)),cases)]=1.
            xs=[s.clusters[s.c[i]].x for i in 1:2]
            row[J+1]=mean(x[1] for x in xs);row[J+2]=mean(x[2]^2 for x in xs)
            row[J+3]=xs[1][1]*xs[2][2]
        end
        for t in 1:N
            state=nothing
            for _ in 1:100000
                j=G.draw_weights(rng,weights);c,S=cases[j];k=maximum(c)
                xs=[G.uniform_point(rng,m) for _ in 1:k]
                ll=sum(loglik(g,x) for (g,x) in zip(groups[j],xs))
                probability=exp(beta*(ll-sum(maxima[j])))*determinant(S,xs)/bounds[j]^k
                @assert -1e-14<=probability<=1+1e-12
                rand(rng)<probability || continue
                state=G.initialize(m,G.Initialization(c,xs,S),:marginal_collapsed,G.seedof(342132,t))
                break
            end
            state===nothing && error("two-dimensional reference sampler exhausted cap")
            observe!(view(before,t,:),state);T.sweep!(state,beta);observe!(view(after,t,:),state)
            @test SM.audit(state)["status"]=="passed"
        end
        delta=after-before
        @test all(abs.(vec(mean(delta;dims=1))).<=6vec(std(delta;dims=1))/sqrt(N).+1/N)
    end
end
end
