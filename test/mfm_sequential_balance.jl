# Standalone: julia --project=. test/mfm_sequential_balance.jl
# Exact finite toy transitions only; no application sampling or screening.
if !isdefined(Main,:Benchmark);include("../src/Benchmark.jl");end
if !isdefined(Main.Benchmark,:MFMSequentialSplitMerge)
    Base.include(Main.Benchmark,joinpath(@__DIR__,"../src/MFMSequentialSplitMerge.jl"))
end
module MFMSequentialBalanceTests
using Test,GeneralDPPMixtures,Random,LinearAlgebra,StaticArrays,SpecialFunctions,Serialization
const G=GeneralDPPMixtures
const B=Main.Benchmark
const MF=B.MFMBaseline
const SS=B.MFMSequentialSplitMerge
bytes(x)=begin io=IOBuffer();serialize(io,x);take!(io);end

function fixture(n,labels=ones(Int,n);M=max(3,maximum(labels)))
    values=[-1.07,-.26,.38,1.13][1:n]
    m=G.Model(SVector{1,Float64}.(values),ones(Int,n),[-3.],[3.],G.make_spectrum(1);
        a=.37,tau=3.4,omega=reshape([.63],1,1),seed=UInt64(541912))
    xs=[sum(m.y[i] for i in findall(==(h),labels))/count(==(h),labels) for h in 1:maximum(labels)]
    s=MF.initialize(m,G.Initialization(labels,xs,collect(1:8)),UInt64(541913))
    s.M=M;s.gamma=.73;s.lambda=1.91
    s
end
groups(labels)=[findall(==(h),labels) for h in 1:maximum(labels)]
function permutations(v)
    isempty(v) && return [Int[]]
    [vcat(v[j],tail) for j in eachindex(v) for tail in permutations(vcat(v[1:j-1],v[j+1:end]))]
end
function partitions(n)
    out=Vector{Vector{Int}}()
    function extend!(labels)
        if length(labels)==n
            push!(out,labels);return
        end
        for h in 1:maximum(labels)+1;extend!(vcat(labels,h));end
    end
    extend!([1]);out
end
function canonical(n,gg)
    labels=zeros(Int,n)
    for (h,g) in enumerate(gg);labels[g].=h;end
    G.canonical_labels(labels)
end

# Independent scalar NIW evidence from the observations themselves. Neither
# production cluster caches nor the production target/proposal code is used.
function evidence(s,g)
    isempty(g) && return 0.
    n=length(g);mu=sum(s.model.y[i][1] for i in g)/n
    scatter=sum((s.model.y[i][1]-mu)^2 for i in g)
    kappa=s.kappa;kn=kappa+n;tau=s.model.tau;omega=s.model.omega[1,1]
    scale=omega+scatter+kappa*n/kn*(mu-s.mu0[1])^2
    -n/2*log(pi)+.5*(log(kappa)-log(kn))+tau/2*log(omega)-
        (tau+n)/2*log(scale)+loggamma((tau+n)/2)-loggamma(tau/2)
end
function logtarget(s,labels,M,beta)
    gg=groups(labels);k=length(gg);n=length(labels);gamma=s.gamma
    M>=k || return -Inf
    sum(log(M-j) for j in 0:k-1)+loggamma(M*gamma)-loggamma(M*gamma+n)+
        sum(loggamma(gamma+length(g))-loggamma(gamma)+beta*evidence(s,g) for g in gg)-
        s.lambda+(M-1)*log(s.lambda)-loggamma(M)
end

function anchored_labels(n,D,anchors)
    other=setdiff(D,anchors);out=Vector{Vector{Int}}()
    for allocation in Iterators.product(ntuple(_->1:2,length(other))...)
        labels=zeros(Int,n);labels[anchors].=[1,2];labels[other].=collect(allocation)
        push!(out,labels)
    end
    out
end
function sequential_logq(s,anchors,order,forced,beta)
    gg=[[anchors[1]],[anchors[2]]];lp=0.
    for i in order
        scores=[log(s.gamma+length(g))+beta*(evidence(s,vcat(g,i))-evidence(s,g)) for g in gg]
        mx=maximum(scores);norm=mx+log(sum(exp.(scores.-mx)))
        h=forced[i];lp+=scores[h]-norm;push!(gg[h],i)
    end
    lp
end

function production_checks!(s,old,M,beta,joint,pair,order)
    w=SS.Workspace(s.model);rng=Xoshiro(541914)
    statebefore=bytes(s);rngbefore=bytes(rng)
    D=findall(h->h==old[pair[1]] || h==old[pair[2]],old)
    # Every endpoint of the sequential law, including assignments that differ
    # from the current full partition, is replayed from anchor-only groups.
    for forced in anchored_labels(length(old),D,collect(pair))
        gg,lp=SS.sequential_split!(s,D,collect(pair),order,rng;forced,beta)
        @test lp≈sequential_logq(s,collect(pair),order,forced,beta) atol=8e-12
        @test gg==[findall(==(h),forced) for h in 1:2]
    end
    @test bytes(rng)==rngbefore
    @test bytes(s)==statebefore
    p=SS.propose!(w,s,rng;pair,order,joint,beta)
    split=old[pair[1]]==old[pair[2]]
    if split && maximum(old)==M && !joint
        @test p.outcome==:capacity
        @test p.ratio===nothing
        @test bytes(s)==statebefore
        return
    end
    unaffected=[g for g in groups(old) if !(pair[1] in g || pair[2] in g)]
    new=canonical(length(old),vcat(unaffected,p.newgroups))
    forced=zeros(Int,length(old))
    if split
        for h in 1:2;forced[p.newgroups[h]].=h;end
        qf=sequential_logq(s,collect(pair),order,forced,beta);qr=0.
    else
        forced[findall(==(old[pair[1]]),old)].=1
        forced[findall(==(old[pair[2]]),old)].=2
        qf=0.;qr=sequential_logq(s,collect(pair),order,forced,beta)
    end
    @test p.logforward≈qf atol=8e-12
    @test p.logreverse≈qr atol=8e-12
    @test p.selectionterm==0.
    exact=logtarget(s,new,p.newM,beta)-logtarget(s,old,M,beta)+qr-qf
    @test p.ratio≈exact atol=2e-11
    raw=SS.raw_ratio(s,p)
    @test raw.raw≈exact atol=2e-11
    @test bytes(s)==statebefore
end

function exact_kernel(n,Mmax,beta;joint=false,checkproduction=true)
    pp=partitions(n)
    states=joint ? [(c,M) for c in pp for M in maximum(c):Mmax] : [(c,Mmax) for c in pp if maximum(c)<=Mmax]
    indices=Dict((Tuple(c),M)=>h for (h,(c,M)) in enumerate(states))
    Q=zeros(length(states),length(states));s=fixture(n)
    pairprob=1/(n*(n-1));capacitymass=0.;nonemptyoutside=0
    for (a,(old,M)) in enumerate(states)
        productionstate=checkproduction ? fixture(n,old;M) : nothing
        for i in 1:n,j in 1:n
            i==j && continue
            anchors=[i,j];split=old[i]==old[j]
            D=findall(h->h==old[i] || h==old[j],old)
            oo=permutations(setdiff(D,anchors));auxprob=pairprob/length(oo)
            unaffected=[g for g in groups(old) if !(i in g || j in g)]
            nonemptyoutside+=!isempty(unaffected)
            newM=joint ? M+(split ? 1 : -1) : M
            # The finite joint test truncates the target at Mmax by rejecting
            # outward proposals. This is solely a test boundary, not a kernel
            # option. Ordinary fixed-M capacity is also an exact self-loop.
            boundary=split && (joint ? M==Mmax : maximum(old)==M)
            if boundary
                Q[a,a]+=pairprob;capacitymass+=pairprob
            end
            for order in oo
                checkproduction && production_checks!(productionstate,old,M,beta,joint,(i,j),order)
                boundary && continue
                if split
                    totalq=0.
                    for forced in anchored_labels(n,D,anchors)
                        qf=sequential_logq(s,anchors,order,forced,beta);q=exp(qf);totalq+=q
                        daughters=[findall(==(h),forced) for h in 1:2]
                        new=canonical(n,vcat(unaffected,daughters));b=indices[(Tuple(new),newM)]
                        ratio=logtarget(s,new,newM,beta)-logtarget(s,old,M,beta)-qf
                        accept=min(1.,exp(ratio));mass=auxprob*q
                        Q[a,b]+=mass*accept;Q[a,a]+=mass*(1-accept)
                    end
                    @test totalq≈1. atol=5e-14
                else
                    new=canonical(n,vcat(unaffected,[D]));b=indices[(Tuple(new),newM)]
                    forced=zeros(Int,n)
                    forced[findall(==(old[i]),old)].=1;forced[findall(==(old[j]),old)].=2
                    qr=sequential_logq(s,anchors,order,forced,beta)
                    ratio=logtarget(s,new,newM,beta)-logtarget(s,old,M,beta)+qr
                    accept=min(1.,exp(ratio))
                    Q[a,b]+=auxprob*accept;Q[a,a]+=auxprob*(1-accept)
                end
            end
        end
    end
    logs=[logtarget(s,c,M,beta) for (c,M) in states]
    weights=exp.(logs.-maximum(logs));weights./=sum(weights)
    flow=Diagonal(weights)*Q
    @test all(isapprox.(vec(sum(Q;dims=2)),1.;atol=7e-13))
    @test minimum(Q)>=0.
    @test maximum(abs,flow-flow')<8e-13
    @test vec(weights'*Q)≈weights atol=8e-13
    @test capacitymass>0
    @test nonemptyoutside>0
    # Both occupied-count and M changes have positive probability whenever
    # permitted; joint moves preserve the invariant difference M-K.
    @test sum(Q[a,b] for a in eachindex(states),b in eachindex(states) if maximum(states[a][1])!=maximum(states[b][1]))>0.
    if joint
        @test all(Q[a,b]==0. || states[a][2]-maximum(states[a][1])==states[b][2]-maximum(states[b][1]) for a in eachindex(states),b in eachindex(states))
        @test sum(Q[a,b] for a in eachindex(states),b in eachindex(states) if states[a][2]!=states[b][2])>0.
    end
    Q
end

@testset "MFM sequential anchored split-merge exhaustive balance" begin
    @test length(partitions(3))==5
    @test length(partitions(4))==15
    for beta in (0.,.6,1.)
        @testset "beta=$beta fixed-M and joint-M complete toy kernels" begin
            exact_kernel(3,2,beta)
            exact_kernel(4,3,beta)
            exact_kernel(3,4,beta;joint=true)
            exact_kernel(4,4,beta;joint=true)
        end
    end
end
end
