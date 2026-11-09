# Standalone: julia --startup-file=no --project=. test/neal_samplers.jl
module NealSamplerTests
using Test, Random, Distributions, SpecialFunctions, QuadGK, Statistics, Serialization, LinearAlgebra
using GeneralDPPMixtures
const G = GeneralDPPMixtures
isdefined(G,:SpeciesSampling) || Base.include(G,joinpath(@__DIR__,"..","src","models","species_sampling","SpeciesSampling.jl"))
isdefined(G,:NealSamplers) || Base.include(G,joinpath(@__DIR__,"..","src","models","species_sampling","algorithms","NealSamplers.jl"))
const S = G.SpeciesSampling
const N = G.NealSamplers
bytes(x) = begin io=IOBuffer(); serialize(io,x); take!(io); end

function partitions(n)
    result=Vector{Vector{Int}}()
    function visit(z)
        length(z)==n && (push!(result,copy(z));return)
        for h in 1:(isempty(z) ? 1 : maximum(z)+1)
            push!(z,h);visit(z);pop!(z)
        end
    end
    visit(Int[])
    result
end

function prior_logmass(alpha,z)
    counts=[count(==(h),z) for h in 1:maximum(z)]
    loggamma(alpha)-loggamma(alpha+length(z))+length(counts)*log(alpha)+sum(loggamma,counts)
end

# A Gaussian vector calculation, independent of the production conjugacy code.
function gaussian_evidence(m,indices)
    n=length(indices)
    n==0 && return 1.0
    covariance=m.observation_sd^2*Matrix{Float64}(I,n,n)+m.base_sd^2*ones(n,n)
    pdf(MvNormal(fill(m.base_mean,n),Symmetric(covariance)),m.y[indices])
end

function partition_reference(m,alpha,evidence)
    masses=Dict(Tuple(z)=>exp(prior_logmass(alpha,z))*
        prod(evidence(m,findall(==(h),z)) for h in 1:maximum(z)) for z in partitions(length(m.y)))
    total=sum(values(masses))
    Dict(key=>value/total for (key,value) in masses)
end

function check_partition_frequencies!(s,step!,exact;warmup=2000,batch_size=200,batches=100)
    for _ in 1:warmup; step!(s); end
    keys_order=sort!(collect(keys(exact)))
    frequencies=zeros(batches,length(keys_order))
    for b in 1:batches,_ in 1:batch_size
        step!(s)
        key=Tuple(s.z)
        j=findfirst(==(key),keys_order)
        frequencies[b,j]+=1/batch_size
    end
    for (j,key) in enumerate(keys_order)
        error=abs(mean(frequencies[:,j])-exact[key])
        @test error < .015+6std(frequencies[:,j])/sqrt(batches)
    end
    frequencies
end

@testset "Neal 2, 3 and 8 DP mixture kernels" begin
    @testset "retained and collapsed targets use the same DP partition law" begin
        m=S.GaussianLocation([-.8,.1,1.4];observation_sd=.7,base_mean=.2,base_sd=1.3)
        alpha=.8
        for z in partitions(3)
            s=N.initialize(m,S.DirichletProcess(alpha),12;partition=z)
            s.theta .= [-.4+.55(h-1) for h in eachindex(s.theta)]
            expected=prior_logmass(alpha,z)+sum(logpdf(Normal(m.base_mean,m.base_sd),t) for t in s.theta)+
                sum(logpdf(Normal(s.theta[z[i]],m.observation_sd),m.y[i]) for i in eachindex(z))
            @test N.joint_logtarget(s)≈expected atol=3e-13
            collapsed=prior_logmass(alpha,z)+sum(log(gaussian_evidence(m,findall(==(h),z))) for h in 1:maximum(z))
            @test N.collapsed_logtarget(s)≈collapsed atol=3e-13
            @test N.check_state(s)
            before=bytes(s)
            N.joint_logtarget(s);N.collapsed_logtarget(s);N.check_state(s)
            @test bytes(s)==before
        end
        @test_throws ArgumentError N.initialize(m,S.PitmanYor(1.,0.),1)
        @test_throws ArgumentError N.neal3_sweep!(S.initialize(m,S.PitmanYor(1.,.2),1))
    end

    @testset "auxiliary weights and singleton recycling" begin
        m=S.GaussianLocation([-.8,.1,1.4];observation_sd=.7,base_mean=.2,base_sd=1.3)
        p=S.DirichletProcess(.8)
        counts=[2,1];theta=[-.2,.7];auxiliary=[-1.,.2,.8]
        scores=N.auxiliary_logweights(m,p,.3,counts,theta,auxiliary)
        expected=vcat([log(counts[h])+logpdf(Normal(theta[h],m.observation_sd),.3) for h in 1:2],
            [log(.8/3)+logpdf(Normal(t,m.observation_sd),.3) for t in auxiliary])
        @test scores≈expected atol=3e-13
        @test_throws ArgumentError N.auxiliary_logweights(m,p,.3,[0],theta[1:1],auxiliary)
        @test_throws ArgumentError N.auxiliary_logweights(m,p,.3,counts,theta,Float64[])
        singleton=N.initialize(S.GaussianLocation([1.2]),p,15)
        old=only(singleton.theta)
        rng=deepcopy(singleton.rng);rand(rng) # one categorical draw, no base draw
        result=N.neal8_reassign!(singleton,1;auxiliaries=1)
        @test result.singleton && result.auxiliary==[old]
        @test singleton.theta==[old] && singleton.z==[1]
        @test bytes(singleton.rng)==bytes(rng)
        # An old singleton occupies one auxiliary slot even when m > 1.
        s=N.initialize(m,p,21;partition=[1,2,2])
        old=s.theta[1]
        result=N.neal8_reassign!(s,1;auxiliaries=4)
        @test result.singleton && length(result.auxiliary)==4 && result.auxiliary[1]==old
        @test N.check_state(s)
        # A nonsingleton supplies m independent base-measure candidates.
        s=N.initialize(m,p,18;partition=[1,1,2])
        rng=deepcopy(s.rng)
        expected=[m.base_mean+m.base_sd*randn(rng) for _ in 1:3]
        result=N.neal8_reassign!(s,1;auxiliaries=3)
        @test !result.singleton && result.auxiliary==expected
        @test N.check_state(s)
        for value in (0,-1,1.5,true)
            before=bytes(s)
            @test_throws ArgumentError N.neal8_sweep!(s;auxiliaries=value)
            @test bytes(s)==before
        end
    end

    @testset "Gaussian posterior partition frequencies for all three algorithms" begin
        m=S.GaussianLocation([-1.,.1,1.3];observation_sd=.75,base_mean=.1,base_sd=1.1)
        alpha=.9
        exact=partition_reference(m,alpha,gaussian_evidence)
        @test sum(values(exact))≈1.
        for (algorithm,step!) in ((2,N.neal2_sweep!),(3,N.neal3_sweep!),
                (8,s->N.neal8_sweep!(s;auxiliaries=1)),(8,s->N.neal8_sweep!(s;auxiliaries=4)))
            s=algorithm==3 ? S.initialize(m,S.DirichletProcess(alpha),172+algorithm;initial_clusters=3) :
                N.initialize(m,S.DirichletProcess(alpha),172+algorithm;initial_clusters=3)
            check_partition_frequencies!(s,step!,exact)
            @test S.check_state(s)
        end
    end

    @testset "state, private RNG, and Gaussian posterior atom refresh" begin
        m=S.GaussianLocation([-2.,-1.,0.,1.,2.];observation_sd=.8,base_mean=.2,base_sd=1.2)
        for step! in (N.neal2_sweep!,s->N.neal8_sweep!(s;auxiliaries=3))
            a=N.initialize(m,S.DirichletProcess(.7),12;initial_clusters=3)
            b=deepcopy(a)
            for _ in 1:100
                step!(a);step!(b)
                @test bytes(a)==bytes(b)
                @test N.check_state(a)
                @test a.sweeps==b.sweeps
                @test a.atom_attempts==a.atom_accepts==0
            end
        end
        s=N.initialize(m,S.DirichletProcess(.7),912)
        variance=inv(inv(m.base_sd^2)+length(m.y)/m.observation_sd^2)
        mean_expected=variance*(m.base_mean/m.base_sd^2+sum(m.y)/m.observation_sd^2)
        samples=[begin N.refresh_atoms!(s);s.theta[1] end for _ in 1:4000]
        @test abs(mean(samples)-mean_expected)<6sqrt(variance/length(samples))
        @test abs(var(samples)-variance)<.08variance
        for i in (0,6,true)
            before=bytes(s)
            @test_throws ArgumentError N.neal2_reassign!(s,i)
            @test_throws ArgumentError N.neal8_reassign!(s,i)
            @test bytes(s)==before
        end
    end

    @testset "nonconjugate Poisson Algorithm 8 and quadrature reference" begin
        m=N.PoissonLogRate([0,1,3];base_mean=.3,base_sd=1.1,proposal_sd=.7)
        alpha=.8
        for y in (0,1,7),theta in (-20.,-.4,.5,3.)
            @test S.loglikelihood(m,y,theta)≈logpdf(Poisson(exp(theta)),y) atol=2e-12
            @test S.logbase(m,theta)≈logpdf(Normal(m.base_mean,m.base_sd),theta) atol=2e-13
        end
        @test S.loglikelihood(m,0,-1000.)==0.
        @test S.loglikelihood(m,1,1000.)==-Inf
        s=N.initialize(m,S.DirichletProcess(alpha),19;partition=[1,2,2])
        s.theta .= [-.3,.8]
        for h in 1:2,proposed in (-1.,.2,1.4)
            before=bytes(s)
            expected=logpdf(Normal(m.base_mean,m.base_sd),proposed)+
                sum(logpdf(Poisson(exp(proposed)),m.y[i]) for i in eachindex(s.z) if s.z[i]==h)
            @test N.atom_logconditional(s,h,proposed)≈expected atol=2e-13
            candidate=deepcopy(s);candidate.theta[h]=proposed
            @test N.joint_logtarget(candidate)-N.joint_logtarget(s)≈
                N.atom_logconditional(s,h,proposed)-N.atom_logconditional(s,h,s.theta[h]) atol=2e-13
            @test N.atom_mh_logratio(s,h,proposed)≈
                N.joint_logtarget(candidate)-N.joint_logtarget(s) atol=2e-13
            @test bytes(s)==before
        end
        @test_throws ArgumentError N.collapsed_logtarget(s)
        before=bytes(s)
        @test_throws ArgumentError N.neal2_sweep!(s)
        @test bytes(s)==before
        collapsed=S.State(m,S.DirichletProcess(alpha),[1,2,2],Xoshiro(1),0)
        before=bytes(collapsed)
        @test_throws ArgumentError N.neal3_sweep!(collapsed)
        @test bytes(collapsed)==before
        function evidence(m,indices)
            quadgk(t->pdf(Normal(m.base_mean,m.base_sd),t)*
                prod(pdf(Poisson(exp(t)),m.y[i]) for i in indices),-35.,25.;rtol=1e-10)[1]
        end
        exact=partition_reference(m,alpha,evidence)
        for auxiliary_count in (1,4)
            state=N.initialize(m,S.DirichletProcess(alpha),672+auxiliary_count;initial_clusters=3)
            check_partition_frequencies!(state,s->N.neal8_sweep!(s;auxiliaries=auxiliary_count),exact;
                warmup=4000,batch_size=300,batches=100)
            @test state.atom_attempts>0 && 0<state.atom_accepts<state.atom_attempts
            @test N.check_state(state)
        end
        for y in (Int[],[-1],[.5],[NaN],[Inf],[true],[2.0^53],[2^52,2^52])
            @test_throws ArgumentError N.PoissonLogRate(y)
        end
        for (key,value) in ((:base_mean,Inf),(:base_mean,true),(:base_sd,0.),(:proposal_sd,-1.))
            @test_throws ArgumentError N.PoissonLogRate([1];Dict(key=>value)...)
        end
        # Large factorial terms must cancel before evaluating an MH ratio.
        # A high-precision calculation is independent of the centred expm1
        # expression used by the sampler. Count support is unchanged.
        large=N.initialize(N.PoissonLogRate([2^52];base_mean=.2,base_sd=1.7),
            S.DirichletProcess(.8),1)
        large.theta[1]=log(Float64(only(large.model.y)))
        for shift in (-1e-7,-1e-8,1e-8,1e-7,-.1,.1)
            proposed=large.theta[1]+shift
            expected=setprecision(BigFloat,256) do
                old=BigFloat(large.theta[1]);candidate=BigFloat(proposed)
                mean=BigFloat(large.model.base_mean);sd=BigFloat(large.model.base_sd)
                count=BigFloat(only(large.model.y))
                Float64(count*(candidate-old)-exp(candidate)+exp(old)-
                    ((candidate-mean)^2-(old-mean)^2)/(2sd^2))
            end
            @test N.atom_mh_logratio(large,1,proposed)≈expected rtol=2e-13 atol=2e-7
        end
        @test N.atom_mh_logratio(large,1,1000.)==-Inf
        @test N.atom_mh_logratio(large,1,large.theta[1])==0.
    end
end
end
