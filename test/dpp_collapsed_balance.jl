# Standalone: julia --project=. test/dpp_collapsed_balance.jl
# Exhaustive tiny partition kernels and read-only proposals, no application MCMC.
if !isdefined(Main,:Benchmark);include("../src/Benchmark.jl");end
module DPPCollapsedBalanceTests
using Test,GeneralDPPMixtures,Random,LinearAlgebra,StaticArrays,Statistics,SpecialFunctions,Serialization
const G=GeneralDPPMixtures
const B=Main.Benchmark
const MR=B.DPPMultiReallocation
const CR=B.DPPCollapsedReallocation
bytes(x)=begin io=IOBuffer();serialize(io,x);take!(io);end

function model(n)
    values=n==5 ? [-.91,-.48,.03,.43,.89] : [-.91,-.58,-.14,.19,.56,.88]
    G.Model(SVector{1,Float64}.(values),ones(Int,n),[-2.],[2.],G.make_spectrum(1);
        a=.37,tau=3.,omega=reshape([.48],1,1),seed=UInt64(58112))
end
groups(labels,r=maximum(labels))=[findall(==(h),labels) for h in 1:r]
function permutations(v)
    isempty(v) && return [Int[]]
    [vcat(v[j],tail) for j in eachindex(v) for tail in permutations(vcat(v[1:j-1],v[j+1:end]))]
end
function anchored_states(n,anchors,anchorlabels)
    r=maximum(anchorlabels);other=setdiff(1:n,anchors);out=Vector{Vector{Int}}()
    for values in Iterators.product(ntuple(_->1:r,length(other))...)
        labels=zeros(Int,n);labels[anchors].=anchorlabels;labels[other].=collect(values);push!(out,labels)
    end
    out
end

# Independent raw-data formula for pi0(c) proportional to product Gamma(a+n) A.
# The integrated center law is on R; the production Student proposal uses that
# same untruncated integral, with out-of-box proposals becoming self-transitions.
function logpi0(m,labels)
    sum(begin
        n=length(g);mu=sum(m.y[i][1] for i in g)/n
        scatter=sum((m.y[i][1]-mu)^2 for i in g);omega=m.omega[1,1];tau=m.tau
        logA=-(n-1)/2*log(pi)-.5log(n)+tau/2*log(omega)-
            (tau+n-1)/2*log(omega+scatter)+loggamma((tau+n-1)/2)-loggamma(tau/2)
        loggamma(m.a+n)-loggamma(m.a)+logA
    end for g in groups(labels))
end
function logq(m,anchors,order,initial,forced,rng)
    last(MR.restricted_scan!(m,collect(eachindex(initial)),anchors,order,rng,initial;forced))
end

function check_adjoint(n,r)
    m=model(n);anchors=collect(1:r);states=anchored_states(n,anchors,anchors)
    orders=permutations(collect(r+1:n));rng=Xoshiro(5172);before=bytes(rng)
    logs=[logpi0(m,c) for c in states];weights=exp.(logs.-maximum(logs));weights./=sum(weights)
    matrices=[Float64[logq(m,anchors,order,old,new,rng) for old in states,new in states] for order in orders]
    wrong_order_error=0.
    for (t,order) in enumerate(orders)
        reverseindex=findfirst(==(reverse(order)),orders);L=matrices[t];R=matrices[reverseindex]
        Q=exp.(L)
        @test all(isapprox.(vec(sum(Q;dims=2)),1.;atol=5e-13))
        @test vec(weights' * Q)≈weights atol=5e-13
        for i in eachindex(states),j in eachindex(states)
            @test logs[i]+L[i,j]≈logs[j]+R[j,i] atol=4e-11
            wrong_order_error=max(wrong_order_error,abs(logs[i]+L[i,j]-logs[j]-L[j,i]))
        end
    end
    @test wrong_order_error>1e-5 # Same-order replay would miss the adjoint.
    @test bytes(rng)==before
end

function check_anchor_auxiliary_kernel()
    m=model(5);n=length(m.y);cases=Vector{Vector{Int}}()
    for values in Iterators.product(ntuple(_->1:2,n)...)
        c=collect(values)
        c==G.canonical_labels(c) && maximum(c)==2 && push!(cases,c)
    end
    @test length(cases)==15
    indices=Dict(Tuple(c)=>j for (j,c) in enumerate(cases));orders=permutations(collect(1:n-2))
    Q=zeros(length(cases),length(cases));uncorrected=similar(Q);fill!(uncorrected,0.)
    rng=Xoshiro(5173);before=bytes(rng)
    for (i,old) in enumerate(cases)
        oldgroups=groups(old);oldproduct=prod(length.(oldgroups))
        for pair in Iterators.product(oldgroups...)
            anchors=collect(pair);remaining=setdiff(1:n,anchors)
            for permutation in orders
                order=remaining[permutation]
                for new in anchored_states(n,anchors,[1,2])
                    probability=exp(logq(m,anchors,order,old,new,rng))/(oldproduct*length(orders))
                    j=indices[Tuple(G.canonical_labels(new))]
                    acceptance=min(1.,oldproduct/prod(length.(groups(new))))
                    Q[i,j]+=probability*acceptance
                    Q[i,i]+=probability*(1-acceptance)
                    uncorrected[i,j]+=probability
                end
            end
        end
    end
    logs=logpi0.(Ref(m),cases);weights=exp.(logs.-maximum(logs));weights./=sum(weights)
    @test all(isapprox.(vec(sum(Q;dims=2)),1.;atol=8e-13))
    flow=Diagonal(weights)*Q;badflow=Diagonal(weights)*uncorrected
    @test maximum(abs,flow-flow')<8e-13
    @test vec(weights'*Q)≈weights atol=8e-13
    # This fixture catches omitting the state-dependent anchor correction.
    @test maximum(abs,badflow-badflow')>1e-6
    @test bytes(rng)==before
end

@testset "Collapsed DPP reallocation exact Gibbs adjoints and anchor balance" begin
    @testset "all anchored states and all scan permutations" begin
        check_adjoint(5,2)
        check_adjoint(6,3)
    end
    @testset "complete canonical partition kernel with anchor auxiliaries" begin
        check_anchor_auxiliary_kernel()
    end
    @testset "actual proposal replays the reverse-order endpoint" begin
        m=model(6);labels=[1,1,2,2,3,3]
        s=G.initialize(m,G.Initialization(labels,[SVector(-.73),SVector(.03),SVector(.72)],collect(1:9)),
            :marginal_collapsed,UInt64(5174))
        before=bytes(s);w=CR.Workspace(m);anchors=[1,3,5];order=[6,4,2]
        changed=0;different_reverse=0;finite_ratios=0
        for seed in 1:30
            p=CR.propose!(w,s,Xoshiro(seed);ids=[1,2,3],anchors,order)
            new=zeros(Int,6)
            for h in 1:3;new[p.newgroups[h]].=h;end
            qf=logq(m,anchors,order,labels,new,Xoshiro(91))
            qr=logq(m,anchors,reverse(order),new,labels,Xoshiro(92))
            @test p.logforward==qf && p.logreverse==qr
            @test logpi0(m,new)-logpi0(m,labels)+qr-qf≈0. atol=4e-11
            @test p.Gterm+p.H+qr-qf≈0. atol=4e-11
            @test p.selectionterm≈log(prod(length.(p.oldgroups))/prod(length.(p.newgroups))) atol=1e-14
            changed+=new!=labels
            different_reverse+=abs(logq(m,anchors,order,new,labels,Xoshiro(93))-qr)>1e-7
            if p.ratio!==nothing
                @test p.ratio≈p.DK+p.selectionterm atol=4e-11
                finite_ratios+=1
            end
        end
        @test changed>0 && different_reverse>0 && finite_ratios>15
        @test bytes(s)==before
    end
end
end
