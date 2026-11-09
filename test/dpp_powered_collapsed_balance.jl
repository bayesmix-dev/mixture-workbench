# Standalone: julia --project=. test/dpp_powered_collapsed_balance.jl
# Exact anchored toy transition laws only; no application MCMC or screening.
if !isdefined(Main,:Benchmark);include("../src/Benchmark.jl");end
module DPPPoweredCollapsedBalanceTests
using Test,GeneralDPPMixtures,Random,LinearAlgebra,StaticArrays,SpecialFunctions,Serialization
const G=GeneralDPPMixtures
const B=Main.Benchmark
const CR=B.DPPCollapsedReallocation
const MR=B.DPPMultiReallocation
bytes(x)=begin io=IOBuffer();serialize(io,x);take!(io);end

function model()
    y=SVector{2,Float64}.([[-.91,-.62],[-.47,.23],[.02,-.18],[.46,.67],[.88,-.39]])
    G.Model(y,ones(Int,5),[-2.,-2.],[2.,2.],G.make_spectrum(2);
        a=.37,tau=3.7,omega=[.58 .11;.11 .41],seed=UInt64(591612))
end
groups(labels)=[findall(==(h),labels) for h in 1:maximum(labels)]
function permutations(v)
    isempty(v) && return [Int[]]
    [vcat(v[j],tail) for j in eachindex(v) for tail in permutations(vcat(v[1:j-1],v[j+1:end]))]
end
function anchored_states(n,anchors)
    r=length(anchors);other=setdiff(1:n,anchors);out=Vector{Vector{Int}}()
    for allocation in Iterators.product(ntuple(_->1:r,length(other))...)
        labels=zeros(Int,n);labels[anchors].=1:r;labels[other].=collect(allocation)
        push!(out,labels)
    end
    out
end

# Raw observations determine every sufficient statistic here. In particular,
# this route does not use production logA_beta, cluster caches, or logA.
multigamma(a,d)=d*(d-1)/4*log(pi)+sum(loggamma(a+(1-j)/2) for j in 1:d)
function logA_raw(m,g,beta)
    n=length(g);d=length(first(m.y));mu=sum(m.y[i] for i in g)/n
    scatter=zeros(d,d)
    for i in g
        delta=m.y[i]-mu;scatter.+=delta*delta'
    end
    Xi=Matrix(m.omega)+scatter;ld=logdet(Symmetric(Xi));p=beta*(m.tau+n)/2
    C=-n*d/2*log(pi)+m.tau/2*logdet(Symmetric(Matrix(m.omega)))+
        multigamma((m.tau+n)/2,d)-multigamma(m.tau/2,d)
    beta*(C-(m.tau+n)/2*ld)+d/2*log(pi)-d/2*log(n)+.5ld+
        loggamma(p-d/2)-loggamma(p)
end
logproxy(m,labels,beta)=sum(loggamma(m.a+length(g))-loggamma(m.a)+logA_raw(m,g,beta) for g in groups(labels))
function logq(m,anchors,order,initial,forced,rng,beta)
    last(CR.restricted_scan_beta!(m,collect(eachindex(initial)),anchors,order,rng,initial;forced,beta))
end

function check_flux(r,beta)
    m=model();n=length(m.y);anchors=collect(1:r);states=anchored_states(n,anchors)
    orders=permutations(collect(r+1:n));rng=Xoshiro(591613);rngbefore=bytes(rng);modelbefore=bytes(m)
    logs=[logproxy(m,c,beta) for c in states]
    weights=exp.(logs.-maximum(logs));weights./=sum(weights)
    laws=[Float64[logq(m,anchors,order,old,new,rng,beta) for old in states,new in states] for order in orders]
    wrong_order_error=0.
    for (t,order) in enumerate(orders)
        reverseindex=findfirst(==(reverse(order)),orders);L=laws[t];R=laws[reverseindex];Q=exp.(L)
        @test all(isapprox.(vec(sum(Q;dims=2)),1.;atol=6e-13))
        @test vec(weights'*Q)≈weights atol=6e-13
        for i in eachindex(states),j in eachindex(states)
            @test logs[i]+L[i,j]≈logs[j]+R[j,i] atol=5e-11
            wrong_order_error=max(wrong_order_error,abs(logs[i]+L[i,j]-logs[j]-L[j,i]))
        end
    end
    @test wrong_order_error>1e-5 # Same-order replay would fail the adjoint.
    @test bytes(rng)==rngbefore
    @test bytes(m)==modelbefore
    if beta==1.
        # The cold path must preserve the exact old arithmetic and draws.
        for order in orders
            initial=first(states);actualrng=Xoshiro(591614);oldrng=copy(actualrng)
            actual=CR.restricted_scan_beta!(m,collect(1:n),anchors,order,actualrng,initial;beta)
            old=MR.restricted_scan!(m,collect(1:n),anchors,order,oldrng,initial)
            @test actual==old
            @test bytes(actualrng)==bytes(oldrng)
        end
    end
end

function state(m,labels)
    xs=[sum(m.y[i] for i in findall(==(h),labels))/count(==(h),labels) for h in 1:maximum(labels)]
    G.initialize(m,G.Initialization(labels,xs,collect(1:9)),:marginal_collapsed,UInt64(591615))
end

@testset "Powered DPP collapsed scan exact anchored adjoints" begin
    for beta in (.6,.92,1.),r in (2,3)
        @testset "beta=$beta, r=$r, every anchored allocation and order" begin
            check_flux(r,beta)
        end
    end
    @testset "global properness threshold precedes every random draw" begin
        m=model();d=length(first(m.y));threshold=d/(m.tau+1)
        initial=[1,2,1,2,1];anchors=[1,2];order=[3,4,5]
        rng=Xoshiro(591616);rngbefore=bytes(rng);modelbefore=bytes(m)
        for beta in (-.1,0.,prevfloat(threshold),threshold,1.01,Inf,NaN)
            @test_throws ArgumentError CR.restricted_scan_beta!(m,collect(1:5),anchors,order,rng,initial;beta)
            @test bytes(rng)==rngbefore
            @test bytes(m)==modelbefore
            for labels in ([1,1,2,2,3],ones(Int,5))
                s=state(m,labels);before=bytes(s)
                @test_throws ArgumentError CR.propose!(CR.Workspace(m),s,rng;beta)
                @test bytes(rng)==rngbefore
                @test bytes(s)==before
            end
        end
    end
end
end
