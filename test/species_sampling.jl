# Standalone: julia --startup-file=no --project=. test/species_sampling.jl
module SpeciesSamplingTests
using Test, Random, Distributions, SpecialFunctions, QuadGK, LinearAlgebra, Statistics, Serialization
using GeneralDPPMixtures
const S = GeneralDPPMixtures.SpeciesSampling

serialised(x) = begin io=IOBuffer(); serialize(io,x); take!(io); end

# Restricted-growth strings enumerate each set partition exactly once.
function partitions(n)
    n == 0 && return [Int[]]
    result = Vector{Int}[]
    function visit(z)
        if length(z) == n
            push!(result, copy(z))
        else
            for h in 1:(maximum(z)+1)
                push!(z,h); visit(z); pop!(z)
            end
        end
    end
    visit([1])
    result
end

# Independent sequential construction from the seating probabilities.
function seating_probability(prior, z)
    isempty(z) && return 1.0
    a,d = S.strength(prior), S.discount(prior)
    counts = [1]
    probability = 1.0
    for i in 2:length(z)
        h = z[i]
        if h > length(counts)
            probability *= (a+d*length(counts))/(a+i-1)
            push!(counts,1)
        else
            probability *= (counts[h]-d)/(a+i-1)
            counts[h] += 1
        end
    end
    probability
end

# A multivariate Normal calculation integrates the common location directly.
function evidence(model, indices)
    isempty(indices) && return 1.0
    n = length(indices)
    covariance = model.observation_sd^2*Matrix{Float64}(I,n,n) + model.base_sd^2*ones(n,n)
    pdf(MvNormal(fill(model.base_mean,n),Symmetric(covariance)), model.y[indices])
end

function independent_mass(model, prior, z)
    seating_probability(prior,z)*prod(evidence(model,findall(==(h),z)) for h in 1:maximum(z))
end

struct Unsupported <: S.ComponentModel
    y::Vector{Float64}
end

@testset "Dirichlet and Pitman–Yor species-sampling mixtures" begin
    @testset "normalised EPPFs, full support and seating probabilities" begin
        priors = (S.DirichletProcess(.8), S.PitmanYor(.8,.4), S.PitmanYor(0.,.6),
            S.PitmanYor(-.3,.6), S.PitmanYor(2.,0.))
        for prior in priors, n in 0:7
            allpartitions = partitions(n)
            masses = Float64[]
            for z in allpartitions
                counts = S.cluster_counts(z)
                mass = exp(S.logeppf(prior,counts))
                push!(masses,mass)
                @test mass ≈ seating_probability(prior,z) atol=2e-14 rtol=5e-14
                @test S.logeppf(prior,reverse(counts)) ≈ S.logeppf(prior,counts) atol=2e-14
            end
            @test sum(masses) ≈ 1.0 atol=5e-14
        end
        for a in (.03,.8,10.), n in 0:6, z in partitions(n)
            counts = S.cluster_counts(z)
            @test S.logeppf(S.DirichletProcess(a),counts) == S.logeppf(S.PitmanYor(a,0.),counts)
            if n > 0
                reference = length(counts)*log(a)+loggamma(a)-loggamma(a+n)+sum(loggamma(c) for c in counts)
                @test S.logeppf(S.DirichletProcess(a),counts) ≈ reference atol=3e-14
            end
        end
        for prior in priors, n in 1:5, z in partitions(n)
            counts = S.cluster_counts(z)
            logs = S.seating_logweights(prior,counts)
            probabilities = exp.(logs)./(n+S.strength(prior))
            @test sum(probabilities) ≈ 1.0 atol=1e-14
            for h in eachindex(logs)
                nextcounts = copy(counts)
                if h > length(counts); push!(nextcounts,1); else; nextcounts[h] += 1; end
                @test exp(S.logeppf(prior,nextcounts)-S.logeppf(prior,counts)) ≈ probabilities[h] atol=2e-14
            end
        end
        for prior in priors
            @test S.seating_logweights(prior,Int[]) == [0.0]
            @test S.logeppf(prior,Int[]) == 0.0
            @test S.logeppf(prior,[1]) == 0.0
        end
        for a in (0.,-1.,Inf,NaN,true,"a")
            @test_throws ArgumentError S.DirichletProcess(a)
        end
        for (a,d) in ((0.,0.),(-.4,.4),(-.8,.4),(1.,-.1),(1.,1.),(Inf,.5),(1.,NaN),(true,.5),(1.,true))
            @test_throws ArgumentError S.PitmanYor(a,d)
        end
        for counts in ([0],[1,-1],[1.,2.],[true])
            @test_throws ArgumentError S.logeppf(priors[1],counts)
            @test_throws ArgumentError S.seating_logweights(priors[1],counts)
        end
    end

    @testset "Gaussian integration, posterior and predictive distributions" begin
        model = S.GaussianLocation([-1.2,.3,1.4,2.2];observation_sd=.7,base_mean=.2,base_sd=1.3)
        for mask in 0:15
            indices = [i for i in eachindex(model.y) if (mask >> (i-1)) & 1 == 1]
            reference = evidence(model,indices)
            @test exp(S.logmarginal(model,indices)) ≈ reference rtol=2e-13
            integral = quadgk(theta -> pdf(Normal(model.base_mean,model.base_sd),theta)*
                prod((pdf(Normal(theta,model.observation_sd),model.y[i]) for i in indices);init=1.0),-Inf,Inf;rtol=1e-10)[1]
            @test exp(S.logmarginal(model,indices)) ≈ integral rtol=2e-9
            @test S.logmarginal(model,reverse(indices)) ≈ S.logmarginal(model,indices) atol=8e-15
            n = length(indices)
            variance = inv(inv(model.base_sd^2)+n/model.observation_sd^2)
            mean = variance*(model.base_mean/model.base_sd^2+sum(model.y[indices])/model.observation_sd^2)
            posterior_mean,posterior_sd = S.posterior_parameters(model,indices)
            @test posterior_mean ≈ mean atol=2e-15
            @test posterior_sd^2 ≈ variance atol=2e-15
            for theta in (-1.,.5,2.)
                @test S.logbase(model,theta)+sum((S.loglikelihood(model,model.y[i],theta) for i in indices);init=0.)-
                    S.logmarginal(model,indices) ≈ logpdf(Normal(mean,sqrt(variance)),theta) atol=2e-14
            end
            rng = Xoshiro(123); reference_rng = deepcopy(rng)
            @test S.randposterior(rng,model,indices) ≈ rand(reference_rng,Normal(mean,sqrt(variance))) atol=2e-15
            for i in eachindex(model.y)
                @test S.logpredictive(model,indices,i) ≈
                    logpdf(Normal(mean,sqrt(variance+model.observation_sd^2)),model.y[i]) atol=1e-14
            end
        end
        rng = Xoshiro(32); reference_rng = deepcopy(rng)
        @test S.randbase(rng,model) == rand(reference_rng,Normal(model.base_mean,model.base_sd))
        @test S.observations(model) === model.y
        for y in (Float64[],[NaN],[Inf],[true],["1"],[1e308,-1e308])
            @test_throws ArgumentError S.GaussianLocation(y)
        end
        for (key,value) in ((:observation_sd,0.),(:observation_sd,true),(:base_sd,-1.),
                (:base_sd,Inf),(:base_sd,1e-200),(:observation_sd,1e200),(:base_mean,NaN))
            @test_throws ArgumentError S.GaussianLocation([0.,1.];Dict(key=>value)...)
        end
    end

    @testset "complete target and finite-state transition invariance" begin
        model = S.GaussianLocation([-.9,-.2,1.1,1.5];observation_sd=.8,base_mean=.1,base_sd=1.4)
        allpartitions = partitions(length(model.y))
        index = Dict(Tuple(z)=>j for (j,z) in enumerate(allpartitions))
        for prior in (S.DirichletProcess(.7),S.PitmanYor(.7,.5),S.PitmanYor(-.2,.4))
            masses = [independent_mass(model,prior,z) for z in allpartitions]
            probabilities = masses./sum(masses)
            sweep_matrix = Matrix{Float64}(I,length(masses),length(masses))
            for i in eachindex(model.y)
                transition = zeros(length(masses),length(masses))
                for (row,z) in enumerate(allpartitions)
                    state = S.initialize(model,prior,43;partition=z)
                    @test S.collapsed_logtarget(state) ≈ log(masses[row]) atol=1e-14
                    before = serialised(state)
                    choices = S.collapsed_allocation(state,i)
                    @test serialised(state) == before
                    weights = exp.(choices.logweights.-maximum(choices.logweights))
                    conditional = weights./sum(weights)
                    raw = [independent_mass(model,prior,candidate) for candidate in choices.partitions]
                    @test conditional ≈ raw./sum(raw) atol=2e-14
                    for (candidate,p) in zip(choices.partitions,conditional)
                        transition[row,index[Tuple(candidate)]] += p
                    end
                end
                @test vec(sum(transition;dims=2)) ≈ ones(length(masses)) atol=2e-14
                flow = Diagonal(probabilities)*transition
                @test flow ≈ transpose(flow) atol=2e-14
                @test transpose(probabilities)*transition ≈ transpose(probabilities) atol=2e-14
                sweep_matrix = sweep_matrix*transition
            end
            @test transpose(probabilities)*sweep_matrix ≈ transpose(probabilities) atol=3e-14
        end
    end

    @testset "tiny posterior frequencies with recorded Monte Carlo bounds" begin
        model = S.GaussianLocation([-.8,.1,1.2];observation_sd=.8,base_mean=.2,base_sd=1.1)
        allpartitions = partitions(3)
        index = Dict(Tuple(z)=>j for (j,z) in enumerate(allpartitions))
        for (prior,seed) in ((S.DirichletProcess(.8),81492),(S.PitmanYor(.3,.5),71932),
                (S.PitmanYor(-.2,.5),43921))
            masses = [independent_mass(model,prior,z) for z in allpartitions]
            reference = masses./sum(masses)
            state = S.initialize(model,prior,seed;initial_clusters=3)
            for _ in 1:2000; S.collapsed_sweep!(state); end
            batches = zeros(100,length(allpartitions))
            for b in axes(batches,1), _ in 1:200
                S.collapsed_sweep!(state)
                batches[b,index[Tuple(state.z)]] += 1/200
            end
            for j in eachindex(reference)
                @test abs(mean(batches[:,j])-reference[j]) < .015+6std(batches[:,j])/sqrt(size(batches,1))
            end
        end
    end

    @testset "state ownership, reproducibility and explicit failures" begin
        model = S.GaussianLocation([-1.,-.8,0.,.2,1.,1.4])
        prior = S.PitmanYor(.7,.3)
        rng = Xoshiro(113); before = serialised(rng)
        a = S.initialize(model,prior,rng;initial_clusters=3)
        @test serialised(rng) == before
        @test a.rng !== rng
        b = S.initialize(model,prior,113;initial_clusters=3)
        @test serialised(a) == serialised(b)
        for sweep in 1:100
            S.collapsed_sweep!(a); S.collapsed_sweep!(b)
            @test serialised(a) == serialised(b)
            @test S.check_state(a)
            @test a.sweeps == sweep
            @test a.z == S.canonical_labels(a.z)
            @test S.cluster_counts(a.z) == [count(==(h),a.z) for h in 1:maximum(a.z)]
            @test isfinite(S.collapsed_logtarget(a))
        end
        labels = [7,7,2,2,7,9]
        state = S.initialize(model,prior,1;partition=labels)
        @test state.z == [1,1,2,2,1,3]
        @test labels == [7,7,2,2,7,9]
        @test S.canonical_labels(Int[]) == Int[]
        @test S.cluster_counts(Int[]) == Int[]
        for seed in (-1,true,1.0)
            @test_throws ArgumentError S.initialize(model,prior,seed)
        end
        for k in (0,7,true,1.0)
            @test_throws ArgumentError S.initialize(model,prior,1;initial_clusters=k)
        end
        for z in ([1],[0,1,1,1,1,1],[true,true,true,true,true,true],[1.,1.,1.,1.,1.,1.])
            @test_throws ArgumentError S.initialize(model,prior,1;partition=z)
        end
        state.z .= [2,2,1,1,2,3]
        @test_throws ArgumentError S.check_state(state)
        for logs in (Float64[],[-Inf,-Inf],[0.,NaN],[0.,Inf])
            @test_throws ArgumentError S.sample_logweights(Xoshiro(1),logs)
        end
        @test S.sample_logweights(Xoshiro(1),[-Inf,0.,-Inf]) == 2
        unsupported = Unsupported([0.,1.])
        @test_throws ArgumentError S.initialize(unsupported,prior,1)
        @test_throws ArgumentError S.loglikelihood(unsupported,0.,0.)
        @test_throws ArgumentError S.logbase(unsupported,0.)
        @test_throws ArgumentError S.randbase(Xoshiro(1),unsupported)
        @test_throws ArgumentError S.randposterior(Xoshiro(1),unsupported,[1])
        singleton = S.initialize(S.GaussianLocation([3.]),S.PitmanYor(-.2,.6),8)
        S.collapsed_sweep!(singleton)
        @test singleton.z == [1]
        @test singleton.sweeps == 1
    end
end
end
