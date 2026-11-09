# Standalone: julia --project=. test/dpp_sequential_balance.jl
# Exhaustive finite partition/proposal tests. Continuous DPP geometry and center
# densities are tested independently in dpp_sequential_splitmerge.jl.
if !isdefined(Main,:Benchmark);include("../src/execution/Benchmark.jl");end
if !isdefined(Main.Benchmark,:DPPSequentialSplitMerge)
    Base.include(Main.Benchmark,joinpath(@__DIR__,"../src/models/point_process/algorithms/DPPSequentialSplitMerge.jl"))
end

module DPPSequentialBalanceTests
using Test,GeneralDPPMixtures,Random,LinearAlgebra,StaticArrays,SpecialFunctions,Serialization
const G=GeneralDPPMixtures
const B=Main.Benchmark
const SS=B.DPPSequentialSplitMerge
bytes(x)=begin io=IOBuffer();serialize(io,x);take!(io);end

function fixture(n,labels=ones(Int,n);M=3)
    y=SVector{1,Float64}.([-1.07,-.26,.38,1.13][1:n])
    m=G.Model(y,ones(Int,n),[-2.],[2.],G.make_spectrum(1);
        a=.37,tau=3.4,omega=reshape([.63],1,1),seed=UInt64(591201))
    xs=[SVector(-1.2+.7(h-1)) for h in 1:maximum(labels)]
    G.initialize(m,G.Initialization(labels,xs,collect(1:M)),:marginal_collapsed,UInt64(591202))
end
groups(labels)=[findall(==(h),labels) for h in 1:maximum(labels)]
function permutations(v)
    isempty(v) && return [Int[]]
    [vcat(v[j],tail) for j in eachindex(v) for tail in permutations(vcat(v[1:j-1],v[j+1:end]))]
end
function partitions(n)
    out=Vector{Vector{Int}}()
    function extend!(labels)
        if length(labels)==n;push!(out,labels);return;end
        for h in 1:maximum(labels)+1;extend!(vcat(labels,h));end
    end
    extend!([1]);out
end
function canonical(n,gg)
    labels=zeros(Int,n)
    for (h,g) in enumerate(gg);labels[g].=h;end
    G.canonical_labels(labels)
end

# Independent untruncated integrated cluster evidence, evaluated from scalar
# observations. This proxy partition target excludes DPP geometry deliberately;
# it isolates the complete ordered-pair/permutation/allocation probability law.
function evidence(m,g,beta=1.)
    n=length(g);mu=sum(m.y[i][1] for i in g)/n
    scatter=sum((m.y[i][1]-mu)^2 for i in g)
    omega=m.omega[1,1];tau=m.tau
    if beta==1
        return -(n-1)/2*log(pi)-.5log(n)+tau/2*log(omega)-
            (tau+n-1)/2*log(omega+scatter)+loggamma((tau+n-1)/2)-loggamma(tau/2)
    end
    C=-n/2*log(pi)+tau/2*log(omega)+loggamma((tau+n)/2)-loggamma(tau/2)
    v=beta*(tau+n);ld=log(omega+scatter)
    beta*(C-(tau+n)/2*ld)+.5log(pi)-.5log(n)+ld/2+loggamma((v-1)/2)-loggamma(v/2)
end
logtarget(m,labels,beta=1.)=sum(loggamma(m.a+length(g))-loggamma(m.a)+evidence(m,g,beta) for g in groups(labels))

function anchored_labels(n,D,anchors)
    other=setdiff(D,anchors);out=Vector{Vector{Int}}()
    for values in Iterators.product(ntuple(_->1:2,length(other))...)
        labels=zeros(Int,n);labels[anchors].=[1,2];labels[other].=collect(values);push!(out,labels)
    end
    out
end
function sequential_logq(m,anchors,order,forced,beta=1.)
    gg=[[anchors[1]],[anchors[2]]];lp=0.
    for i in order
        scores=[log(m.a+length(g))+evidence(m,vcat(g,i),beta)-evidence(m,g,beta) for g in gg]
        top=maximum(scores);normalizer=top+log(sum(exp.(scores.-top)))
        h=forced[i];lp+=scores[h]-normalizer;push!(gg[h],i)
    end
    lp
end

function production_checks!(s,old,pair,order,beta=1.)
    rng=Xoshiro(591203);before=bytes(s);rbefore=bytes(rng)
    D=findall(h->h==old[pair[1]] || h==old[pair[2]],old)
    for forced in anchored_labels(length(old),D,collect(pair))
        gg,lp=SS.sequential_split!(s.model,D,collect(pair),order,rng;forced,beta)
        @test lp≈sequential_logq(s.model,collect(pair),order,forced,beta) atol=9e-12
        @test gg==[findall(==(h),forced) for h in 1:2]
    end
    @test bytes(rng)==rbefore && bytes(s)==before
    p=SS.propose!(SS.Workspace(),s,rng;anchors=pair,order,beta)
    split=old[pair[1]]==old[pair[2]]
    if split && maximum(old)==length(s.S)
        @test p.outcome==:capacity && p.ratio===nothing
        @test bytes(s)==before
        return
    end
    outside=[g for g in groups(old) if !(pair[1] in g || pair[2] in g)]
    new=canonical(length(old),vcat(outside,p.newgroups))
    forced=zeros(Int,length(old))
    if split
        for h in 1:2;forced[p.newgroups[h]].=h;end
    else
        forced[findall(==(old[pair[1]]),old)].=1;forced[findall(==(old[pair[2]]),old)].=2
    end
    @test p.logs≈sequential_logq(s.model,collect(pair),order,forced,beta) atol=9e-12
    proxychange=logtarget(s.model,new,beta)-logtarget(s.model,old,beta)
    @test (split ? p.Gterm+p.H : -p.Gterm-p.H)≈proxychange atol=2e-11
    if p.ratio!==nothing
        exact=proxychange+p.DK+(split ? -p.logs : p.logs)
        @test p.ratio≈exact atol=2e-11
        @test SS.raw_ratio(s,p;beta).raw≈p.ratio atol=3e-9
    end
    @test bytes(s)==before
end

function exact_kernel(n,M;beta=1.)
    states=filter(c->maximum(c)<=M,partitions(n));N=length(states)
    indices=Dict(Tuple(c)=>j for (j,c) in enumerate(states))
    Q=zeros(N,N);wronganchors=zeros(N,N);wrongtwo=zeros(N,N)
    m=fixture(n;M).model;pairprob=1/(n*(n-1));capacitymass=0.;unaffectedseen=0
    for (a,old) in enumerate(states)
        s=fixture(n,old;M)
        for i in 1:n,j in 1:n
            i==j && continue
            anchors=[i,j];split=old[i]==old[j]
            D=findall(h->h==old[i] || h==old[j],old)
            orders=permutations(setdiff(D,anchors));auxprob=pairprob/length(orders)
            outside=[g for g in groups(old) if !(i in g || j in g)];unaffectedseen+=!isempty(outside)
            boundary=split && maximum(old)==M
            if boundary
                for matrix in (Q,wronganchors,wrongtwo);matrix[a,a]+=pairprob;end
                capacitymass+=pairprob
            end
            for order in orders
                production_checks!(s,old,(i,j),order,beta)
                boundary && continue
                candidates=split ? anchored_labels(n,D,anchors) : [Int[]]
                totalq=0.
                for forced in candidates
                    if split
                        logq=sequential_logq(m,anchors,order,forced,beta);q=exp(logq);totalq+=q
                        replacement=[findall(==(h),forced) for h in 1:2]
                        logratioq=-logq
                    else
                        forced=zeros(Int,n);forced[findall(==(old[i]),old)].=1;forced[findall(==(old[j]),old)].=2
                        replacement=[D];q=1.;logratioq=sequential_logq(m,anchors,order,forced,beta)
                    end
                    new=canonical(n,vcat(outside,replacement));b=indices[Tuple(new)]
                    ratio=logtarget(m,new,beta)-logtarget(m,old,beta)+logratioq
                    oldaffected=[findall(==(old[i]),old)]
                    split || push!(oldaffected,findall(==(old[j]),old))
                    wrongselection=log(prod(length.(oldaffected))/prod(length.(replacement)))
                    for (matrix,offset) in ((Q,0.),(wronganchors,wrongselection),(wrongtwo,split ? log(2.) : -log(2.)))
                        acceptance=min(1.,exp(ratio+offset));mass=auxprob*q
                        matrix[a,b]+=mass*acceptance;matrix[a,a]+=mass*(1-acceptance)
                    end
                end
                split && (@test totalq≈1. atol=5e-14)
            end
        end
    end
    logs=[logtarget(m,c,beta) for c in states];weights=exp.(logs.-maximum(logs));weights./=sum(weights)
    flow=Diagonal(weights)*Q
    @test all(isapprox.(vec(sum(Q;dims=2)),1.;atol=8e-13))
    @test minimum(Q)>=0.
    @test maximum(abs,flow-flow')<8e-13
    @test vec(weights'*Q)≈weights atol=8e-13
    @test maximum(abs,Diagonal(weights)*wronganchors-(Diagonal(weights)*wronganchors)')>1e-6
    @test maximum(abs,Diagonal(weights)*wrongtwo-(Diagonal(weights)*wrongtwo)')>1e-6
    @test capacitymass>0. && unaffectedseen>0
    @test sum(Q[a,b] for a in 1:N,b in 1:N if maximum(states[a])!=maximum(states[b]))>0.
end

@testset "DPP sequential exhaustive ordered-anchor and scan-order balance" begin
    @test length(partitions(3))==5 && length(partitions(4))==15
    exact_kernel(3,2)
    exact_kernel(4,2)
    exact_kernel(4,3)
    exact_kernel(3,2;beta=.37)
    exact_kernel(4,3;beta=.71)
end

end
