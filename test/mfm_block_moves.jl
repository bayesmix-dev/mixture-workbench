# Standalone: julia --project=. test/mfm_block_moves.jl
if !isdefined(Main,:Benchmark); include("../src/Benchmark.jl"); end
if !isdefined(Main.Benchmark,:MFMTarget)
    Base.include(Main.Benchmark,joinpath(@__DIR__,"../src/MFMTarget.jl"))
end
if !isdefined(Main.Benchmark,:MFMBlockMoves)
    Base.include(Main.Benchmark,joinpath(@__DIR__,"../src/MFMBlockMoves.jl"))
end

module MFMBlockMoveTests
using Test, GeneralDPPMixtures, Random, LinearAlgebra, StaticArrays, Statistics, Serialization
const G=GeneralDPPMixtures
const MF=Main.Benchmark.MF
const MT=Main.Benchmark.MFMTarget
const MR=Main.Benchmark.DPPMultiReallocation
const R=Main.Benchmark.MFMBlockMoves

function statebytes(s)
    io=IOBuffer(); serialize(io,s); take!(io)
end
function state(m,labels;M=5,gamma=.37,seed=61)
    s=MF.initialize(m,G.Initialization(labels,[zero(first(m.y)) for _ in 1:maximum(labels)],[1]),seed)
    s.M=M; s.gamma=gamma; s
end
function fixture(d=2)
    rng=Xoshiro(292+d); y=[SVector{d,Float64}(randn(rng,d)) for _ in 1:10]
    m=G.Model(y,ones(Int,10),fill(-5.,d),fill(5.,d),G.make_spectrum(d);
        tau=d+2.,omega=.8Matrix{Float64}(I,d,d))
    labels=[1,1,1,2,2,2,3,3,3,3]
    refs=[copy(labels),[1,1,2,2,3,3,3,3,2,1],
        [1,1,2,2,3,3,4,4,4,4],[1,1,1,1,2,2,2,2,2,2],[1,2,1,2,1,2,1,2,1,2]]
    m,state(m,labels),refs
end
function audit(s)
    @test sort(unique(s.c))==sort(s.active)
    @test sum(c.n for c in s.clusters)==length(s.c)
    @test length(s.active)<=s.M
    for h in eachindex(s.clusters)
        @test s.clusters[h].n==count(==(h),s.c)
        fresh=R.cluster(s,findall(==(h),s.c))
        @test s.clusters[h].mean≈fresh.mean atol=2e-12
        @test s.clusters[h].scatter≈fresh.scatter atol=2e-11
    end
    fresh=deepcopy(s)
    for h in eachindex(s.clusters); MF.refresh!(fresh,h); end
    @test all(a≈b for (a,b) in zip(s.predL,fresh.predL))
    @test s.predmu≈fresh.predmu && s.prednorm≈fresh.prednorm
end
function partitions(n,M)
    result=Vector{Vector{Int}}()
    function recurse(prefix)
        if length(prefix)==n; push!(result,prefix); return; end
        for h in 1:min(M,maximum(prefix)+1); recurse(vcat(prefix,h)); end
    end
    recurse([1]); result
end
function permutations(v)
    isempty(v) && return [Int[]]
    [vcat(x,tail) for x in v for tail in permutations(filter(!=(x),v))]
end

function temperedtarget(s;groups=nothing,M=s.M,beta=1.)
    evidence=groups===nothing ? MT.likelihood(s) : sum(MT.log_marginal(s,g) for g in groups)
    MT.logtarget(s;groups,M)+(beta-1)*evidence
end

# Enumerate the actual local proposal law over every ordered observation pair,
# every scan order and every supported local endpoint. Hyperparameters are fixed.
# A finite joint-M verification target uses a TEST-only upper boundary self-loop.
function local_transition_matrix(states,cases,w;joint=false,Mmax=nothing,beta=1.)
    n=length(first(states).c); T=zeros(length(states),length(states))
    logs=[temperedtarget(s;beta) for s in states]
    bank=MR.Workspace(w.bank.references,w.bank.whitened,2)
    for (a,s) in enumerate(states),i in 1:n,j in 1:n
        i==j && continue
        ids=unique([s.c[i],s.c[j]]); split=length(ids)==1; k=length(s.active)
        newM=joint ? s.M+(split ? 1 : -1) : s.M
        if (!joint && split && k==s.M) || (Mmax!==nothing && newM>Mmax)
            T[a,a]+=1/(n*(n-1)); continue
        end
        oldgroups=[findall(==(h),s.c) for h in ids]
        unaffected=[findall(==(h),s.c) for h in s.active if !(h in ids)]
        D=sort(vcat(oldgroups...)); anchors=[i,j]; nonanchors=setdiff(D,anchors)
        oldanchorlabels=split ? [1,1] : [1,2]
        newanchorlabels=split ? [1,2] : [1,1]; r=maximum(newanchorlabels)
        oldlabels=zeros(Int,n)
        for h in eachindex(oldgroups); oldlabels[oldgroups[h]].=h; end
        forward=MR.launches(bank,s.model,D,anchors;anchorlabels=newanchorlabels)
        reverse=MR.launches(bank,s.model,D,anchors;anchorlabels=oldanchorlabels)
        for order in permutations(nonanchors)
            qr,_=R.mixture_logprob(s,D,anchors,order,Xoshiro(1),reverse,oldlabels;anchorlabels=oldanchorlabels,beta)
            auxiliary=1/(n*(n-1)*factorial(length(nonanchors)))
            for allocation in Iterators.product(ntuple(_->1:r,length(nonanchors))...)
                labels=zeros(Int,n); labels[anchors].=newanchorlabels; labels[nonanchors].=collect(allocation)
                newgroups=[findall(==(h),labels) for h in 1:r]
                qf,_=R.mixture_logprob(s,D,anchors,order,Xoshiro(1),forward,labels;anchorlabels=newanchorlabels,beta)
                complete=vcat(unaffected,newgroups); endpoint=zeros(Int,n)
                for h in eachindex(complete); endpoint[complete[h]].=h; end
                delta=temperedtarget(s;groups=complete,M=newM,beta)-logs[a]
                alpha=min(1.,exp(delta+qr-qf))
                b=findfirst(==((newM,G.canonical_labels(endpoint))),cases)
                b===nothing && error("missing local verification endpoint")
                mass=auxiliary*exp(qf); T[a,b]+=mass*alpha; T[a,a]+=mass*(1-alpha)
            end
        end
    end
    T
end

function global_transition_matrix(states,cases,w;joint=false,Mmax=nothing,beta=1.)
    n=length(first(states).c); T=zeros(length(states),length(states)); D=collect(1:n)
    logs=[temperedtarget(s;beta) for s in states]
    for (a,s) in enumerate(states),i in D,j in D
        i==j && continue
        primary=unique([s.c[i],s.c[j]]); split=length(primary)==1; k=length(s.active)
        newM=joint ? s.M+(split ? 1 : -1) : s.M
        if (!joint && split && k==s.M) || (Mmax!==nothing && newM>Mmax)
            T[a,a]+=1/(n*(n-1)); continue
        end
        remaining=setdiff(s.active,primary)
        for ordered in permutations(remaining)
            othergroups=[findall(==(h),s.c) for h in ordered]
            for choice in Iterators.product(othergroups...)
                aa=collect(Int,choice); anchors=vcat([i,j],aa); ids=vcat(primary,ordered)
                oldgroups=[findall(==(h),s.c) for h in ids]; oldlabels=zeros(Int,n)
                for h in eachindex(oldgroups); oldlabels[oldgroups[h]].=h; end
                oldanchorlabels=split ? vcat([1,1],collect(2:k)) : collect(1:k)
                newanchorlabels=split ? collect(1:k+1) : vcat([1,1],collect(2:k-1))
                r=maximum(newanchorlabels); nonanchors=setdiff(D,anchors)
                forward=MR.launches(w.bank,s.model,D,anchors;anchorlabels=newanchorlabels)
                reverse=MR.launches(w.bank,s.model,D,anchors;anchorlabels=oldanchorlabels)
                for order in permutations(nonanchors)
                    auxiliary=1/(n*(n-1)*factorial(length(remaining))*prod(length.(othergroups))*factorial(length(nonanchors)))
                    qr,_=R.mixture_logprob(s,D,anchors,order,Xoshiro(1),reverse,oldlabels;anchorlabels=oldanchorlabels,beta)
                    for allocation in Iterators.product(ntuple(_->1:r,length(nonanchors))...)
                        labels=zeros(Int,n); labels[anchors].=newanchorlabels; labels[nonanchors].=collect(allocation)
                        groups=[findall(==(h),labels) for h in 1:r]
                        qf,_=R.mixture_logprob(s,D,anchors,order,Xoshiro(1),forward,labels;anchorlabels=newanchorlabels,beta)
                        othernew=split ? groups[3:end] : groups[2:end]
                        selection=sum(log(length(g)) for g in othergroups;init=0.)-sum(log(length(g)) for g in othernew;init=0.)
                        delta=temperedtarget(s;groups,M=newM,beta)-logs[a]
                        alpha=min(1.,exp(delta+qr-qf+selection))
                        b=findfirst(==((newM,G.canonical_labels(labels))),cases)
                        mass=auxiliary*exp(qf); T[a,b]+=mass*alpha; T[a,a]+=mass*(1-alpha)
                    end
                end
            end
        end
    end
    T
end

function reallocation_transition_matrix(states,w;beta=1.)
    n=length(first(states).c); k=length(first(states).active)
    r=w.bank.blocks==0 ? k : min(w.bank.blocks,k)
    T=zeros(length(states),length(states)); labelslist=[G.canonical_labels(s.c) for s in states]
    logs=[temperedtarget(s;beta) for s in states]
    for (a,s) in enumerate(states),mask in 1:2^k-1
        count_ones(mask)==r || continue
        ids=[h for h in 1:k if mask & (1<<(h-1)) != 0]
        oldgroups=[findall(==(h),s.c) for h in ids]
        unaffected=[findall(==(h),s.c) for h in s.active if !(h in ids)]
        D=sort(vcat(oldgroups...)); oldlabels=zeros(Int,n)
        for h in 1:r; oldlabels[oldgroups[h]].=h; end
        for choice in Iterators.product(oldgroups...)
            anchors=collect(Int,choice); nonanchors=setdiff(D,anchors)
            bank=MR.launches(w.bank,s.model,D,anchors)
            for order in permutations(nonanchors)
                auxiliary=1/(binomial(k,r)*prod(length.(oldgroups))*factorial(length(nonanchors)))
                qr,_=R.mixture_logprob(s,D,anchors,order,Xoshiro(1),bank,oldlabels;beta)
                for allocation in Iterators.product(ntuple(_->1:r,length(nonanchors))...)
                    labels=zeros(Int,n); labels[anchors].=1:r; labels[nonanchors].=collect(allocation)
                    groups=[findall(==(h),labels) for h in 1:r]; complete=vcat(unaffected,groups)
                    qf,_=R.mixture_logprob(s,D,anchors,order,Xoshiro(1),bank,labels;beta)
                    selection=sum(log(length(g)) for g in oldgroups)-sum(log(length(g)) for g in groups)
                    delta=temperedtarget(s;groups=complete,beta)-logs[a]
                    alpha=min(1.,exp(delta+qr-qf+selection)); endpoint=zeros(Int,n)
                    for h in eachindex(complete); endpoint[complete[h]].=h; end
                    b=findfirst(==(G.canonical_labels(endpoint)),labelslist)
                    mass=auxiliary*exp(qf); T[a,b]+=mass*alpha; T[a,a]+=mass*(1-alpha)
                end
            end
        end
    end
    T
end

@testset "Exact NIW-collapsed MFM block moves" begin
    @testset "predictive scores, normalized mixture law and frozen launches" begin
        m,s,refs=fixture(); w=R.Workspace(m;references=refs,blocks=2)
        refs[1][1]=99; @test w.bank.references[1][1]==1
        for members in (Int[],[1],[1,2,4])
            c=R.cluster(s,members); temporary=R.cache_state(s,[c],Int[],[1])
            i=first(setdiff(1:10,members)); before=MT.log_marginal(s,c)
            @test MF.logpred(temporary,1,m.y[i])≈MT.log_marginal(s,vcat(members,i))-before atol=2e-11
        end
        D=[1,2,3,4]; anchors=[1,3]; order=[4,2]
        bank=MR.launches(w.bank,m,D,anchors)
        total=0.
        for a in 1:2,b in 1:2
            forced=zeros(Int,10); forced[anchors].=1:2; forced[order].=[a,b]
            lp,components=R.mixture_logprob(s,D,anchors,order,Xoshiro(1),bank,forced)
            @test exp(lp)≈sum(exp,components)/length(bank) atol=1e-14
            total+=exp(lp)
        end
        @test total≈1. atol=2e-13
        ra=Xoshiro(99); rb=deepcopy(ra); forced=zeros(Int,10); forced[D].=[1,1,2,2]
        R.mixture_logprob(s,D,anchors,order,ra,bank,forced)
        @test rand(ra)==rand(rb)
    end

    @testset "full target ratio, count factors, reverse replay and atomic caches" begin
        checked=0; selectionchanged=0
        for d in (1,3), gamma in (.13,1.7), blocks in (0,2)
            m,s,refs=fixture(d); s.gamma=gamma; w=R.Workspace(m;references=refs,blocks)
            for seed in 1:18, kind in (:reallocate,:global)
                before=statebytes(s)
                pair=isodd(seed) ? (1,2) : (1,4)
                p=kind==:global ? R.propose!(w,s,Xoshiro(seed);kind,pair) : R.propose!(w,s,Xoshiro(seed);kind)
                @test statebytes(s)==before
                @test p.M==s.M && p.gamma==s.gamma
                direct=R.raw_ratio(s,p)
                @test p.ratio≈direct.raw atol=2e-10
                expected=p.newk>p.k ? log(s.M-p.k) : p.newk<p.k ? -log(s.M-p.k+1) : 0.
                @test p.countterm==expected
                candidate=deepcopy(s); accepted=deepcopy(p); accepted.outcome=:accepted
                oldhyper=(s.M,s.gamma,s.lambda,s.gamma_attempts,s.gamma_accepts,deepcopy(s.rng))
                R.commit!(candidate,accepted)
                @test MT.partition_logtarget(candidate)-MT.partition_logtarget(s)≈direct.target_difference atol=2e-10
                @test MT.logtarget(candidate)-MT.logtarget(s)≈direct.target_difference atol=2e-10
                @test (candidate.M,candidate.gamma,candidate.lambda,candidate.gamma_attempts,candidate.gamma_accepts)==oldhyper[1:5]
                @test rand(deepcopy(candidate.rng))==rand(oldhyper[6])
                audit(candidate)
                D=sort(vcat(p.oldgroups...)); oldlabels=zeros(Int,length(s.c)); newlabels=copy(oldlabels)
                for h in eachindex(p.oldgroups); oldlabels[p.oldgroups[h]].=h; end
                for h in eachindex(p.newgroups); newlabels[p.newgroups[h]].=h; end
                forward=MR.launches(w.bank,m,D,p.anchors;anchorlabels=p.newanchorlabels)
                reverse=MR.launches(w.bank,m,D,p.anchors;anchorlabels=p.oldanchorlabels)
                lf,_=R.mixture_logprob(s,D,p.anchors,p.order,Xoshiro(9),forward,newlabels;anchorlabels=p.newanchorlabels)
                lr,_=R.mixture_logprob(s,D,p.anchors,p.order,Xoshiro(9),reverse,oldlabels;anchorlabels=p.oldanchorlabels)
                @test lf==p.logforward && lr==p.logreverse
                @test -p.ratio≈-direct.target_difference+lf-lr-p.selectionterm atol=2e-10
                checked+=1; selectionchanged+=p.selectionterm!=0
            end
        end
        @test checked==288 && selectionchanged>0
    end

    @testset "actual transitions, support, observation and rejection atomicity" begin
        m,s,refs=fixture(); w=R.Workspace(m;references=refs,blocks=0); rng=Xoshiro(810)
        cap=deepcopy(s); cap.M=length(cap.active); before=statebytes(cap)
        @test R.attempt!(w,cap,rng;kind=:global,pair=(1,2)).outcome==:capacity
        @test statebytes(cap)==before
        @test_throws ArgumentError R.propose!(w,s,rng;kind=:unknown)
        @test_throws ArgumentError R.propose!(w,s,rng;kind=:global,pair=(1,1))
        @test_throws ArgumentError R.propose!(w,s,rng;kind=:global,pair=(1,4),otheranchors=[1])
        accepted=0; rejected=0; splits=0; merges=0
        for it in 1:300
            before=statebytes(s); hyper=(s.M,s.gamma,s.lambda)
            p=R.attempt!(w,s,rng;kind=isodd(it) ? :global : :reallocate)
            p===nothing && continue
            @test (s.M,s.gamma,s.lambda)==hyper
            if p.outcome==:accepted
                accepted+=1; splits+=p.kind==:global_split; merges+=p.kind==:global_merge
                audit(s)
            else
                rejected+=1; @test statebytes(s)==before
            end
        end
        @test accepted>0 && rejected>0 && splits>0 && merges>0
        a=deepcopy(s); b=deepcopy(s); ra=Xoshiro(82); rb=Xoshiro(82); observed=Ref(0)
        for _ in 1:30
            pa=R.attempt!(w,a,ra;kind=:global)
            pb=R.attempt!(w,b,rb;kind=:global,observer=(_,p)->(observed[]+=1))
            @test all(getfield(pa,f)==getfield(pb,f) for f in fieldnames(typeof(pa)))
        end
        @test observed[]==30 && statebytes(a)==statebytes(b) && rand(ra)==rand(rb)
        # Baseline local hyperparameter updates remain composable with the cache.
        for _ in 1:20
            MF.sweep!(s); R.attempt!(w,s,rng;kind=:global); audit(s)
        end
    end

    @testset "joint M/k target ratio, count prior and capacity escape" begin
        checked=0
        for d in (1,3), M in (3,5), gamma in (.13,1.7), lambda in (.2,2.4)
            m,s,refs=fixture(d); s.M=M; s.gamma=gamma; s.lambda=lambda
            w=R.Workspace(m;references=refs,blocks=0)
            for pair in ((1,2),(1,4))
                before=statebytes(s)
                p=R.propose!(w,s,Xoshiro(87);kind=:global_joint,pair)
                @test statebytes(s)==before && p.outcome!=:capacity
                @test p.newM-p.M==p.newk-p.k && abs(p.newM-p.M)==1
                @test p.kind in (:global_joint_split,:global_joint_merge)
                direct=R.raw_ratio(s,p)
                @test p.ratio≈direct.raw atol=3e-10
                n=length(s.c)
                expected=MT.logfallingfactorial(p.newM,p.newk)-MT.logfallingfactorial(p.M,p.k)+
                    G.loggamma(p.newM*gamma)-G.loggamma(p.newM*gamma+n)-
                    G.loggamma(p.M*gamma)+G.loggamma(p.M*gamma+n)+
                    (p.newM-p.M)*log(lambda)-G.loggamma(p.newM)+G.loggamma(p.M)
                @test p.countterm≈expected atol=3e-10
                candidate=deepcopy(s); accepted=deepcopy(p); accepted.outcome=:accepted
                R.commit!(candidate,accepted)
                @test candidate.M==p.newM && length(candidate.active)==p.newk
                @test candidate.gamma==gamma && candidate.lambda==lambda
                @test MT.logtarget(candidate)-MT.logtarget(s)≈direct.target_difference atol=3e-10
                audit(candidate); checked+=1
            end
        end
        @test checked==32
    end

    @testset "local fixed/joint target ratios, untouched blocks and atomic caches" begin
        checked=0
        for d in (1,3),M in (3,5),gamma in (.13,1.7),lambda in (.2,2.4),blocks in (0,2)
            m,s,refs=fixture(d); s.M=M; s.gamma=gamma; s.lambda=lambda
            w=R.Workspace(m;references=refs,blocks)
            for pair in ((1,2),(1,4)),kind in (:local,:local_joint)
                before=statebytes(s); seed=87
                p=R.propose!(w,s,Xoshiro(seed);kind,pair)
                @test statebytes(s)==before
                @test p.kind in (:local_split,:local_merge,:local_joint_split,:local_joint_merge)
                @test isempty(p.otheranchors) && length(p.anchors)==2 && p.anchors==collect(pair)
                @test p.oldids==unique(s.c[collect(pair)])
                @test p.union_size==sum(length,p.oldgroups)
                if p.outcome==:capacity
                    @test kind==:local && M==3 && pair==(1,2)
                    @test p.newM==s.M && p.ratio===nothing
                    @test statebytes(R.commit!(s,p))==before
                    continue
                end
                @test p.selectionterm==0.
                @test abs(p.newk-p.k)==1 && p.newM-p.M==(kind==:local_joint ? p.newk-p.k : 0)
                # Every reference is included locally, even when its number of
                # global blocks differs and when called with blocks=0 workspace.
                @test length(p.component_logforward)==length(refs)+1
                @test length(p.component_logreverse)==length(refs)+1
                direct=R.raw_ratio(s,p)
                @test p.ratio≈direct.raw atol=3e-10
                @test direct.log_forward==p.logforward && direct.log_reverse==p.logreverse
                expected=MT.logfallingfactorial(p.newM,p.newk)-MT.logfallingfactorial(p.M,p.k)+
                    G.loggamma(p.newM*gamma)-G.loggamma(p.newM*gamma+length(s.c))-
                    G.loggamma(p.M*gamma)+G.loggamma(p.M*gamma+length(s.c))+
                    (p.newM-p.M)*log(lambda)-G.loggamma(p.newM)+G.loggamma(p.M)
                @test p.countterm≈expected atol=3e-10
                candidate=deepcopy(s); accepted=deepcopy(p); accepted.outcome=:accepted
                R.commit!(candidate,accepted)
                @test length(candidate.active)==p.newk && candidate.M==p.newM
                @test MT.logtarget(candidate)-MT.logtarget(s)≈direct.target_difference atol=3e-10
                @test candidate.gamma==gamma && candidate.lambda==lambda
                @test rand(deepcopy(candidate.rng))==rand(deepcopy(s.rng))
                unaffected=[findall(==(h),s.c) for h in s.active if !(h in p.oldids)]
                for g in unaffected
                    @test findall(==(candidate.c[first(g)]),candidate.c)==g
                end
                audit(candidate)
                D=sort(vcat(p.oldgroups...)); oldlabels=zeros(Int,length(s.c)); newlabels=copy(oldlabels)
                for h in eachindex(p.oldgroups); oldlabels[p.oldgroups[h]].=h; end
                for h in eachindex(p.newgroups); newlabels[p.newgroups[h]].=h; end
                localbank=MR.Workspace(w.bank.references,w.bank.whitened,2)
                reversed_forward=MR.launches(localbank,m,D,p.anchors;anchorlabels=p.oldanchorlabels)
                reversed_reverse=MR.launches(localbank,m,D,p.anchors;anchorlabels=p.newanchorlabels)
                lf,_=R.mixture_logprob(candidate,D,p.anchors,p.order,Xoshiro(9),reversed_forward,oldlabels;anchorlabels=p.oldanchorlabels)
                lr,_=R.mixture_logprob(candidate,D,p.anchors,p.order,Xoshiro(9),reversed_reverse,newlabels;anchorlabels=p.newanchorlabels)
                @test lf==p.logreverse && lr==p.logforward
                @test -p.ratio≈-direct.target_difference+lr-lf atol=3e-10
                checked+=1
            end
        end
        @test checked==112
        m,s,refs=fixture(); pairwork=R.Workspace(m;references=refs,blocks=2)
        allwork=R.Workspace(m;references=refs,blocks=0)
        for kind in (:local,:local_joint),pair in ((1,2),(1,4))
            ra=Xoshiro(78); rb=deepcopy(ra)
            a=R.propose!(pairwork,s,ra;kind,pair); b=R.propose!(allwork,s,rb;kind,pair)
            @test all(getfield(a,f)==getfield(b,f) for f in fieldnames(typeof(a)))
            @test rand(ra)==rand(rb)
        end
        @test_throws ArgumentError R.propose!(pairwork,s,Xoshiro(1);kind=:local,pair=(1,1))
        @test_throws ArgumentError R.propose!(pairwork,s,Xoshiro(1);kind=:local,pair=(0,2))
        @test_throws ArgumentError R.propose!(pairwork,s,Xoshiro(1);kind=:local,pair=(1,2),order=[4])
        for kind in (:local,:local_joint)
            a=deepcopy(s); b=deepcopy(s); ra=Xoshiro(638); rb=deepcopy(ra); accepted=0; rejected=0
            for _ in 1:50
                before=statebytes(a)
                pa=R.attempt!(pairwork,a,ra;kind)
                pb=R.attempt!(pairwork,b,rb;kind,observer=(_,p)->nothing)
                @test all(getfield(pa,f)==getfield(pb,f) for f in fieldnames(typeof(pa)))
                @test statebytes(a)==statebytes(b)
                if pa.outcome==:accepted; accepted+=1; audit(a)
                else; rejected+=1; @test statebytes(a)==before; end
            end
            @test accepted>0 && rejected>0 && rand(ra)==rand(rb)
        end
    end

    @testset "exact local fixed-M and joint-M transition matrices" begin
        y=[SVector(-.8),SVector(-.2),SVector(.4),SVector(1.1)]
        m=G.Model(y,ones(Int,4),[-2.],[2.],G.make_spectrum(1);tau=3.,omega=reshape([.7],1,1))
        refs=[[1,1,1,1],[1,1,2,2],[1,2,1,2],[1,2,3,3],[1,2,3,4]]
        w=R.Workspace(m;references=refs,blocks=2)
        for gamma in (.23,1.1),joint in (false,true)
            Mmax=4
            # Production has no M cap; outward proposals become self-loops
            # only in the finite joint-M verification matrix below.
            cases=joint ? [(M,c) for M in 1:Mmax for c in partitions(4,M)] : [(3,c) for c in partitions(4,3)]
            states=[state(m,c;M,gamma) for (M,c) in cases]
            for s in states; s.lambda=1.7; end
            logs=MT.logtarget.(states); probabilities=exp.(logs.-maximum(logs)); probabilities./=sum(probabilities)
            T=local_transition_matrix(states,cases,w;joint,Mmax=joint ? Mmax : nothing)
            @test all(abs.(vec(sum(T;dims=2)).-1).<3e-12)
            @test vec(probabilities'T)≈probabilities atol=3e-12
            flux=Diagonal(probabilities)*T
            @test flux≈flux' atol=3e-12
            @test any(T[a,b]>0 && maximum(cases[a][2])!=maximum(cases[b][2]) for a in eachindex(cases) for b in eachindex(cases))
        end
    end

    @testset "likelihood tempering target ratios, support and beta-one compatibility" begin
        for d in (1,3),blocks in (0,2),beta in (0.,.27,.73,1.),kind in (:reallocate,:local,:local_joint,:global,:global_joint),pair in ((1,2),(1,4))
            m,s,refs=fixture(d); w=R.Workspace(m;references=refs,blocks)
            rng=Xoshiro(72); before=statebytes(s)
            p=kind==:reallocate ? R.propose!(w,s,rng;kind,beta) : R.propose!(w,s,rng;kind,beta,pair)
            @test statebytes(s)==before && p.beta==beta
            direct=R.raw_ratio(s,p)
            @test p.ratio≈direct.raw atol=5e-10
            @test p.ratio≈p.Gterm+beta*p.H+p.countterm+p.selectionterm+p.logreverse-p.logforward atol=5e-13
            # Count and hyperparameter priors remain untempered, including the
            # full-data M*gamma normalizer in a local joint-count proposal.
            n=length(s.c); gamma=s.gamma
            expected=MT.logfallingfactorial(p.newM,p.newk)-MT.logfallingfactorial(p.M,p.k)+
                G.loggamma(p.newM*gamma)-G.loggamma(p.newM*gamma+n)-
                G.loggamma(p.M*gamma)+G.loggamma(p.M*gamma+n)+
                (p.newM-p.M)*log(s.lambda)-G.loggamma(p.newM)+G.loggamma(p.M)
            @test p.countterm≈expected atol=5e-10
            candidate=deepcopy(s); accepted=deepcopy(p); accepted.outcome=:accepted; R.commit!(candidate,accepted)
            @test temperedtarget(candidate;beta)-temperedtarget(s;beta)≈direct.target_difference atol=5e-10
            audit(candidate)
        end
        m,s,refs=fixture(); w=R.Workspace(m;references=refs,blocks=2)
        for kind in (:reallocate,:local,:local_joint,:global,:global_joint)
            a=deepcopy(s); b=deepcopy(s); ra=Xoshiro(82); rb=deepcopy(ra)
            for _ in 1:12
                pa=R.attempt!(w,a,ra;kind); pb=R.attempt!(w,b,rb;kind,beta=1.)
                @test all(isequal(getfield(pa,f),getfield(pb,f)) for f in fieldnames(typeof(pa)))
                @test statebytes(a)==statebytes(b)
            end
            @test rand(ra)==rand(rb)
        end
        for beta in (-.1,1.1,Inf,-Inf,NaN,"invalid",1+im)
            rng=Xoshiro(92); copyrng=deepcopy(rng); before=statebytes(s)
            @test_throws ArgumentError R.propose!(w,s,rng;kind=:local_joint,beta)
            @test statebytes(s)==before && rand(rng)==rand(copyrng)
            @test_throws ArgumentError R.restricted_scan!(s,[1,2],[1,2],Int[],rng,ones(Int,10);beta)
        end
    end

    @testset "independent tempered detailed balance of every block kernel" begin
        y=[SVector(-.8),SVector(-.2),SVector(.4),SVector(1.1)]
        m=G.Model(y,ones(Int,4),[-2.],[2.],G.make_spectrum(1);tau=3.,omega=reshape([.7],1,1))
        refs=[[1,1,1,1],[1,1,2,2],[1,2,1,2],[1,2,3,3],[1,2,3,4]]
        w=R.Workspace(m;references=refs,blocks=0)
        for beta in (0.,.27,.73,1.),joint in (false,true),scope in (:local,:global)
            cases=joint ? [(M,c) for M in 1:4 for c in partitions(4,M)] : [(3,c) for c in partitions(4,3)]
            states=[state(m,c;M,gamma=.43) for (M,c) in cases]
            for s in states; s.lambda=1.7; end
            logs=[temperedtarget(s;beta) for s in states]
            probabilities=exp.(logs.-maximum(logs)); probabilities./=sum(probabilities)
            matrix=scope==:local ? local_transition_matrix : global_transition_matrix
            T=matrix(states,cases,w;joint,Mmax=joint ? 4 : nothing,beta)
            @test all(abs.(vec(sum(T;dims=2)).-1).<3e-12)
            @test vec(probabilities'T)≈probabilities atol=3e-12
            flux=Diagonal(probabilities)*T
            @test flux≈flux' atol=3e-12
        end
        for beta in (0.,.27,.73,1.),k in (2,3),blocks in (0,2)
            pp=filter(c->maximum(c)==k,partitions(4,3)); states=[state(m,c;M=4,gamma=.43) for c in pp]
            logs=[temperedtarget(s;beta) for s in states]
            probabilities=exp.(logs.-maximum(logs)); probabilities./=sum(probabilities)
            work=R.Workspace(m;references=refs,blocks)
            T=reallocation_transition_matrix(states,work;beta)
            @test all(abs.(vec(sum(T;dims=2)).-1).<3e-12)
            @test vec(probabilities'T)≈probabilities atol=3e-12
            flux=Diagonal(probabilities)*T
            @test flux≈flux' atol=3e-12
        end
    end

    @testset "exact enumerated global transition matrix conditional on M,gamma" begin
        # Exhaust all auxiliary pairs, other-cluster anchor selections/orderings,
        # scan orders and supported endpoint partitions, not a Monte Carlo test.
        y=SVector{1,Float64}.([[-.8],[-.2],[.4],[1.1]])
        m=G.Model(y,ones(Int,4),[-2.],[2.],G.make_spectrum(1);tau=3.,omega=reshape([.7],1,1))
        pp=partitions(4,3); states=[state(m,c;M=3,gamma=.61) for c in pp]
        logs=MT.partition_logtarget.(states); probabilities=exp.(logs.-maximum(logs)); probabilities./=sum(probabilities)
        refs=[[1,1,1,1],[1,1,2,2],[1,2,1,2],[1,2,3,3]]
        w=R.Workspace(m;references=refs,blocks=0); T=zeros(length(pp),length(pp)); D=collect(1:4)
        for (a,s) in enumerate(states), i in D, j in D
            i==j && continue
            primary=unique([s.c[i],s.c[j]]); split=length(primary)==1; k=length(s.active)
            if split && k==s.M; T[a,a]+=1/12; continue; end
            remaining=setdiff(s.active,primary)
            for ordered in permutations(remaining)
                othergroups=[findall(==(h),s.c) for h in ordered]
                for choice in Iterators.product(othergroups...)
                    aa=collect(Int,choice); anchors=vcat([i,j],aa); ids=vcat(primary,ordered)
                    oldgroups=[findall(==(h),s.c) for h in ids]
                    oldlabels=zeros(Int,4)
                    for h in eachindex(oldgroups); oldlabels[oldgroups[h]].=h; end
                    oldanchors=split ? vcat([1,1],collect(2:k)) : collect(1:k)
                    newanchors=split ? collect(1:k+1) : vcat([1,1],collect(2:k-1))
                    r=maximum(newanchors); nonanchors=setdiff(D,anchors)
                    for order in permutations(nonanchors)
                        auxiliary=1/(12factorial(length(remaining))*prod(length.(othergroups))*factorial(length(nonanchors)))
                        forward=MR.launches(w.bank,m,D,anchors;anchorlabels=newanchors)
                        reverse=MR.launches(w.bank,m,D,anchors;anchorlabels=oldanchors)
                        qr,_=R.mixture_logprob(s,D,anchors,order,Xoshiro(1),reverse,oldlabels;anchorlabels=oldanchors)
                        for allocation in Iterators.product(ntuple(_->1:r,length(nonanchors))...)
                            newlabels=zeros(Int,4); newlabels[anchors].=newanchors; newlabels[nonanchors].=collect(allocation)
                            newgroups=[findall(==(h),newlabels) for h in 1:r]
                            qf,_=R.mixture_logprob(s,D,anchors,order,Xoshiro(1),forward,newlabels;anchorlabels=newanchors)
                            othernew=split ? newgroups[3:end] : newgroups[2:end]
                            selection=sum(log(length(g)) for g in othergroups;init=0.)-sum(log(length(g)) for g in othernew;init=0.)
                            delta=MT.partition_logtarget(s;groups=newgroups)-logs[a]
                            alpha=min(1.,exp(delta+qr-qf+selection))
                            b=findfirst(==(G.canonical_labels(newlabels)),pp)
                            mass=auxiliary*exp(qf); T[a,b]+=mass*alpha; T[a,a]+=mass*(1-alpha)
                        end
                    end
                end
            end
        end
        @test all(abs.(vec(sum(T;dims=2)).-1).<3e-12)
        @test vec(probabilities'T)≈probabilities atol=3e-12
        flux=Diagonal(probabilities)*T
        @test flux≈flux' atol=3e-12
        # A zero count correction would fail this: all three occupied counts
        # have positive reference mass and M=3 makes the split factors unequal.
        @test all(sum(probabilities[j] for j in eachindex(pp) if maximum(pp[j])==k)>0 for k in 1:3)
    end

    @testset "independent finite posterior one-step composition invariance" begin
        y=[SVector(-.9),SVector(-.4),SVector(.1),SVector(.6),SVector(1.2)]
        m=G.Model(y,ones(Int,5),[-2.],[2.],G.make_spectrum(1);tau=3.,omega=reshape([.9],1,1))
        for gamma in (.2,1.3)
            pp=partitions(5,4); states=[state(m,c;M=4,gamma) for c in pp]
            logs=MT.partition_logtarget.(states); probabilities=exp.(logs.-maximum(logs)); probabilities./=sum(probabilities)
            pair=R.Workspace(m;references=pp[1:6],blocks=2)
            allwork=R.Workspace(m;references=pp,blocks=0)
            N=3000; J=length(pp); before=zeros(N,J+1); after=zeros(N,J+1)
            rng=Xoshiro(7300+round(Int,100gamma)); transitionrng=Xoshiro(7400+round(Int,100gamma))
            for t in 1:N
                a=G.draw_weights(rng,probabilities); s=deepcopy(states[a]); before[t,a]=1.; before[t,end]=length(s.active)
                R.attempt!(pair,s,transitionrng); R.attempt!(allwork,s,transitionrng); R.attempt!(allwork,s,transitionrng;kind=:global)
                b=findfirst(==(G.canonical_labels(s.c)),pp); after[t,b]=1.; after[t,end]=length(s.active)
            end
            delta=after-before; difference=vec(mean(delta;dims=1)); se=vec(std(delta;dims=1))/sqrt(N)
            @test all(abs.(difference).<=6se.+1/N)
            @test all(abs.(vec(mean(before[:,1:J];dims=1)).-probabilities).<=6sqrt.(probabilities.*(1 .-probabilities)./N).+1/N)
        end
    end

    @testset "exact joint M/k detailed balance on bounded verification target" begin
        # The production joint move has no upper bound on M. For an exactly
        # finite verification matrix, condition the target on M<=4 and replace
        # outward proposals at that TEST boundary with self-transitions.
        y=[SVector(-.8),SVector(.1),SVector(.9)]
        m=G.Model(y,ones(Int,3),[-2.],[2.],G.make_spectrum(1);tau=3.,omega=reshape([.7],1,1))
        Mmax=4; cases=[(M,c) for M in 1:Mmax for c in partitions(3,M)]
        states=[state(m,c;M,gamma=.43) for (M,c) in cases]
        for s in states; s.lambda=1.7; end
        logs=MT.logtarget.(states); probabilities=exp.(logs.-maximum(logs)); probabilities./=sum(probabilities)
        w=R.Workspace(m;references=[[1,1,1],[1,1,2],[1,2,3]],blocks=0)
        T=zeros(length(cases),length(cases)); D=collect(1:3)
        for (a,s) in enumerate(states), i in D, j in D
            i==j && continue
            primary=unique([s.c[i],s.c[j]]); split=length(primary)==1; k=length(s.active)
            newM=s.M+(split ? 1 : -1)
            if newM>Mmax; T[a,a]+=1/6; continue; end
            remaining=setdiff(s.active,primary)
            for ordered in permutations(remaining)
                othergroups=[findall(==(h),s.c) for h in ordered]
                for choice in Iterators.product(othergroups...)
                    aa=collect(Int,choice); anchors=vcat([i,j],aa); ids=vcat(primary,ordered)
                    oldgroups=[findall(==(h),s.c) for h in ids]; oldlabels=zeros(Int,3)
                    for h in eachindex(oldgroups); oldlabels[oldgroups[h]].=h; end
                    oldanchors=split ? vcat([1,1],collect(2:k)) : collect(1:k)
                    newanchors=split ? collect(1:k+1) : vcat([1,1],collect(2:k-1))
                    r=maximum(newanchors); nonanchors=setdiff(D,anchors)
                    for order in permutations(nonanchors)
                        auxiliary=1/(6factorial(length(remaining))*prod(length.(othergroups))*factorial(length(nonanchors)))
                        forward=MR.launches(w.bank,m,D,anchors;anchorlabels=newanchors)
                        reverse=MR.launches(w.bank,m,D,anchors;anchorlabels=oldanchors)
                        qr,_=R.mixture_logprob(s,D,anchors,order,Xoshiro(1),reverse,oldlabels;anchorlabels=oldanchors)
                        for allocation in Iterators.product(ntuple(_->1:r,length(nonanchors))...)
                            newlabels=zeros(Int,3); newlabels[anchors].=newanchors; newlabels[nonanchors].=collect(allocation)
                            newgroups=[findall(==(h),newlabels) for h in 1:r]
                            qf,_=R.mixture_logprob(s,D,anchors,order,Xoshiro(1),forward,newlabels;anchorlabels=newanchors)
                            othernew=split ? newgroups[3:end] : newgroups[2:end]
                            selection=sum(log(length(g)) for g in othergroups;init=0.)-sum(log(length(g)) for g in othernew;init=0.)
                            delta=MT.logtarget(s;groups=newgroups,M=newM)-logs[a]
                            alpha=min(1.,exp(delta+qr-qf+selection))
                            b=findfirst(==((newM,G.canonical_labels(newlabels))),cases)
                            mass=auxiliary*exp(qf); T[a,b]+=mass*alpha; T[a,a]+=mass*(1-alpha)
                        end
                    end
                end
            end
        end
        @test all(abs.(vec(sum(T;dims=2)).-1).<3e-12)
        @test vec(probabilities'T)≈probabilities atol=3e-12
        flux=Diagonal(probabilities)*T
        @test flux≈flux' atol=3e-12
        @test any(T[a,b]>0 && cases[a][1]!=cases[b][1] for a in eachindex(cases) for b in eachindex(cases))
    end
end

end
