# Standalone: julia --startup-file=no --project=. test/poisson_lognormal.jl
module PoissonLognormalTests
using Test, Random, Distributions, SpecialFunctions, QuadGK, Statistics, Serialization, TOML
include(joinpath(@__DIR__,"..","examples","poisson_lognormal","run.jl"))
const CLI=PoissonLognormalCLI
const P=CLI.P
const G=CLI.G
bytes(x)=begin io=IOBuffer();serialize(io,x);take!(io);end

function setstate!(s,z,eta)
    s.z.=z;s.eta.=eta
    fill!(s.n,0);fill!(s.sumy,0)
    for i in eachindex(z)
        s.n[z[i]]+=1;s.sumy[z[i]]+=s.model.y[i]
    end
    s
end

# A separate probability calculation through Distributions, including the
# multinomial coefficient that converts count probabilities to labelled z.
function independent_target(m,z,eta)
    counts=[count(==(h),z) for h in 1:m.M]
    logpdf(DirichletMultinomial(length(z),fill(m.gamma,m.M)),counts)-
        loggamma(length(z)+1)+sum(loggamma(n+1) for n in counts)+
        sum(logpdf(Normal(m.mu,m.sd),e) for e in eta)+
        sum(logpdf(Poisson(exp(eta[z[i]])),m.y[i]) for i in eachindex(z))
end

@testset "Fixed finite nonconjugate Poisson-lognormal extension" begin
    @testset "Poisson likelihood and labelled target" begin
        for y in (0,1,7,100),eta in (-20.,-1.,0.,3.,6.)
            @test P.loglikelihood(y,eta)≈logpdf(Poisson(exp(eta)),y) atol=2e-12
        end
        @test P.loglikelihood(1,1000.)==-Inf
        @test P.loglikelihood(0,-1000.)==0.
        m=P.Model([0,1,3,2];components=3,weight_shape=.7,base_mean=.3,base_sd=1.2)
        s=P.initialize(m,123)
        for z in Iterators.product(ntuple(_->1:m.M,length(m.y))...)
            setstate!(s,collect(z),[-.8,.2,1.1])
            @test P.logtarget(s)≈independent_target(m,s.z,s.eta) atol=2e-12
        end
    end
    @testset "allocation full conditionals and symmetric Metropolis ratios" begin
        m=P.Model([0,2,5,1];components=3,weight_shape=.4,base_mean=.2,base_sd=1.3,proposal_sd=.6)
        s=setstate!(P.initialize(m,31),[1,1,2,2],[-.5,.8,1.4])
        for i in eachindex(s.z)
            before=bytes(s);scores=P.allocation_logweights!(zeros(m.M),s,i)
            @test bytes(s)==before
            direct=[begin
                labels=copy(s.z);labels[i]=h
                independent_target(m,labels,s.eta)
            end for h in 1:m.M]
            @test scores.-scores[1]≈direct.-direct[1] atol=4e-12
        end
        for h in 1:m.M,delta in (-1.1,.3,1.4)
            proposal=s.eta[h]+delta
            eta=copy(s.eta);eta[h]=proposal
            expected=independent_target(m,s.z,eta)-independent_target(m,s.z,s.eta)
            @test P.mh_logratio(s,h,proposal)≈expected atol=4e-12
            @test logpdf(Normal(s.eta[h],m.proposal_sd),proposal)≈
                logpdf(Normal(proposal,m.proposal_sd),s.eta[h]) atol=2e-14
        end
        # Component one is empty: its refresh is a direct base-measure draw.
        setstate!(s,[2,2,2,2],[-.5,.8,1.4])
        rng=deepcopy(s.rng);expected=m.mu+m.sd*randn(rng)
        @test P.log_rate_conditional(s,1,.2)≈logpdf(Normal(m.mu,m.sd),.2)
        P.rate_sweep!(s)
        @test s.eta[1]==expected
        @test s.empty_refreshes==2 && s.attempted==1
        @test P.mh_logratio(s,2,1000.)==-Inf
    end
    @testset "independent quadrature and exhaustive partition probabilities" begin
        m=P.Model([0,1,3];components=2,weight_shape=.6,base_mean=.4,base_sd=1.0,proposal_sd=.8)
        # These integrals use probability densities from Distributions, with
        # no production loglikelihood, logbase or logallocation calls.
        function evidence(ids)
            isempty(ids) && return 1.0
            quadgk(e->pdf(Normal(m.mu,m.sd),e)*prod(pdf(Poisson(exp(e)),m.y[i]) for i in ids),
                -40.,40.;rtol=1e-10)[1]
        end
        labelled=Dict{Tuple,Float64}()
        for z in Iterators.product(ntuple(_->1:m.M,length(m.y))...)
            groups=[findall(==(h),collect(z)) for h in 1:m.M]
            counts=length.(groups)
            pz=exp(logpdf(DirichletMultinomial(length(z),fill(m.gamma,m.M)),counts)-
                loggamma(length(z)+1)+sum(loggamma(n+1) for n in counts))
            key=Tuple(G.canonical_labels(collect(z)))
            labelled[key]=get(labelled,key,0.)+pz*prod(evidence(g) for g in groups)
        end
        # Enumerate canonical partitions independently, and include the number
        # M!/(M-k)! of labelled-component assignments to occupied groups.
        canonical=Dict{Tuple,Float64}()
        for key in ((1,1,1),(1,1,2),(1,2,1),(1,2,2))
            k=maximum(key);groups=[findall(==(h),collect(key)) for h in 1:k]
            multiplicity=factorial(m.M)/factorial(m.M-k)
            prior=multiplicity*exp(loggamma(m.M*m.gamma)-loggamma(length(key)+m.M*m.gamma)+
                sum(loggamma(length(g)+m.gamma)-loggamma(m.gamma) for g in groups))
            canonical[key]=prior*prod(evidence(g) for g in groups)
            @test labelled[key]≈canonical[key] rtol=2e-13
        end
        normalizer=sum(values(canonical))
        exact=Dict(key=>mass/normalizer for (key,mass) in canonical)
        @test sum(values(exact))≈1. atol=1e-14
        # Compare actual posterior exploration to quadrature using batch means.
        # The broad absolute tolerance bounds the role of Monte Carlo noise.
        s=P.initialize(m,81932)
        for _ in 1:3000;P.sweep!(s);end
        keys_order=sort!(collect(keys(exact)))
        batches=zeros(100,length(keys_order))
        for b in axes(batches,1),_ in 1:400
            P.sweep!(s)
            j=findfirst(==(Tuple(G.canonical_labels(s.z))),keys_order)
            batches[b,j]+=1/400
        end
        for (j,key) in enumerate(keys_order)
            @test abs(mean(batches[:,j])-exact[key]) < .015+6std(batches[:,j])/sqrt(size(batches,1))
        end
    end
    @testset "state invariants, output labels and reproducibility" begin
        m=P.Model([0,0,1,2,8,9,11];components=4)
        a=P.initialize(m,G.seedof(17,1,9101));b=deepcopy(a)
        for _ in 1:200
            P.sweep!(a);P.sweep!(b)
            @test bytes(a)==bytes(b)
            @test a.n==[count(==(h),a.z) for h in 1:m.M]
            @test a.sumy==[sum((m.y[i] for i in eachindex(a.z) if a.z[i]==h);init=0) for h in 1:m.M]
            @test isfinite(P.logtarget(a))
        end
        c=P.initialize(m,G.seedof(17,2,9101))
        P.sweep!(c)
        @test c.eta!=a.eta
        setstate!(a,[3,3,1,1,3,1,3],[.2,.5,1.4,-.2])
        labels,eta=P.canonical_output(a)
        @test labels==[1,1,2,2,1,2,1]
        @test eta==[1.4,.2,.5,-.2]
        @test eta[labels]==a.eta[a.z]
        canonical=deepcopy(a);setstate!(canonical,labels,eta)
        @test P.logtarget(canonical)≈P.logtarget(a) atol=2e-13
        @test P.logtarget(canonical)≈independent_target(m,labels,eta) atol=2e-12
    end
    @testset "strict configuration, count data and CLI output protocol" begin
        for y in (Int[],[-1],[.5],[NaN],[Inf],[true],[2.0^53],[2^53+1],[2^52,2^52])
            @test_throws ArgumentError P.Model(y)
        end
        @test P.Model([2^53-1]).y==[2^53-1]
        @test P.Model([2^52,2^52-1]).y==[2^52,2^52-1]
        for (key,value) in ((:components,0),(:components,2.0),(:components,true),(:weight_shape,0.),
                (:base_sd,-1.),(:base_sd,Inf),(:proposal_sd,0.),(:base_mean,NaN))
            @test_throws ArgumentError P.Model([1,2];Dict(key=>value)...)
        end
        for (key,value) in (("unknown",1),("method","hdp"),("iterations",0),("warmup",-1),
                ("thin",0),("chains",false),("seed",-1),("save_allocations",1),
                ("iterations",true),("iterations",1.0),("base_mean",Inf),
                ("base_sd",0.0),("proposal_sd",NaN),("weight_shape",-1.))
            @test_throws ArgumentError P.configuration(Dict("method"=>"poisson_lognormal",key=>value),[1,2])
        end
        @test_throws ArgumentError P.configuration(Dict("method"=>"poisson_lognormal","iterations"=>10,"warmup"=>10),[1,2])
        @test_throws ArgumentError P.configuration(Dict("method"=>"poisson_lognormal","iterations"=>10,"warmup"=>9,"thin"=>2),[1,2])
        mktempdir() do dir
            config=joinpath(dir,"config.toml");data=joinpath(dir,"counts.csv")
            raw=Dict("method"=>"poisson_lognormal","components"=>2,"iterations"=>40,
                "warmup"=>20,"thin"=>5,"chains"=>2,"seed"=>23)
            open(config,"w") do io;TOML.print(io,raw);end
            write(data,"count\n0\n1\n4\n5\n")
            out=CLI.run(config,data,joinpath(dir,"first"))
            second=CLI.run(config,data,joinpath(dir,"second"))
            for file in ("draws.csv","allocations.csv","diagnostics.csv","config.toml")
                @test read(joinpath(out,file))==read(joinpath(second,file))
            end
            header,A=CLI.IOProtocol.read_numeric_csv(joinpath(out,"draws.csv"))
            @test header==["chain","iteration","k","M","entropy","log_target","eta_1","eta_2"]
            @test size(A)==(8,8) && all(isfinite,A)
            @test all(A[:,4].==2)
            @test length(readlines(joinpath(out,"allocations.csv")))==1+8*4
            metadata=TOML.parsefile(joinpath(out,"metadata.toml"))
            @test metadata["run_status"]=="completed" && metadata["empirical_bayes"]==false
            @test length(metadata["source_digest"])==length(metadata["extension_source_digest"])==64
            @test_throws ArgumentError CLI.run(config,data,out)
            write(data,"a,b\n1,2\n")
            @test_throws ArgumentError CLI.run(config,data,joinpath(dir,"badshape"))
            @test !ispath(joinpath(dir,"badshape"))
            for value in ("-1","1.5","NaN","Inf","missing","9007199254740992","9007199254740993")
                write(data,"count\n$value\n")
                @test_throws ArgumentError CLI.run(config,data,joinpath(dir,"baddata"))
                @test !ispath(joinpath(dir,"baddata"))
            end
            raw["save_allocations"]=false
            open(config,"w") do io;TOML.print(io,raw);end
            write(data,"count\n0\n1\n4\n5\n")
            nosave=CLI.run(config,data,joinpath(dir,"nosave"))
            @test !ispath(joinpath(nosave,"allocations.csv"))
        end
    end
end
end
