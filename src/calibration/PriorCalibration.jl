module PriorCalibration
using GeneralDPPMixtures, Random, Statistics, LinearAlgebra, Distributions
const G=GeneralDPPMixtures

"Calibrate the actual retained DPP spectrum conditional on nonemptiness."
function dpp(;d=4,target=5.,c0=.999,nu=4.,cutoff=1e-8)
    target>1 && 0<c0<1 && nu>0 || error("invalid DPP calibration")
    f(a)=G.build_spectrum(Val(d),a,c0,nu,cutoff)
    lo=1.; hi=2.
    while f(lo).conditional_mean<target; lo/=1.2; end
    while f(hi).conditional_mean>target; hi*=1.2; end
    for _ in 1:55
        mid=(lo+hi)/2
        f(mid).conditional_mean>target ? (lo=mid) : (hi=mid)
    end
    s=f((lo+hi)/2)
    abs(s.conditional_mean-target)<1e-5 || error("cutoff prevents exact count calibration")
    s
end

"Kernel-based pair-correlation diagnostics before conditioning on nonemptiness."
function dpp_metrics(s)
    ls=s.lambda[s.shell]; mass=sum(ls)
    integrated=sum(abs2,ls)/mass^2
    grid=range(0,.5;length=2001)
    gs=[1-(sum(ls.*cos.(2pi*t.*s.frequencies[1,:]))/mass)^2 for t in grid]
    idx=findfirst(>=(.5),gs)
    (integrated_deficit=integrated,half_range=isnothing(idx) ? NaN : grid[idx],grid=collect(grid),g=gs)
end

function conflicts(xs,R)
    sum((sum(abs2,xs[i]-xs[j])<=R^2 for i in eachindex(xs) for j in 1:i-1);init=0)
end
point(rng,m)=m.lower+(m.upper-m.lower).*rand(rng,length(m.lower))

"Stratified uniform-configuration integration of the nonempty Strauss count law."
function strauss(m,R,logalpha;target=5.,draws=30000,maxcount=30,seed=73001)
    logalpha<=0 || error("Strauss factor must be at most one")
    rng=Xoshiro(seed); coeff=ones(maxcount); se=zeros(maxcount)
    for k in 2:maxcount
        ws=[exp(logalpha*conflicts([point(rng,m) for _ in 1:k],R)) for _ in 1:draws]
        coeff[k]=mean(ws);se[k]=std(ws)/sqrt(draws)
    end
    function law(beta)
        # Recurrence for beta^k/k!, multiplied by the integrated interaction.
        terms=zeros(maxcount); t=1.
        for k in 1:maxcount; t*=beta/k; terms[k]=t*coeff[k]; end
        terms/sum(terms)
    end
    lo=.01;hi=10.
    while sum((1:maxcount).*law(hi))<target;hi*=2;hi<100 || error("increase count truncation");end
    for _ in 1:60
        mid=(lo+hi)/2
        sum((1:maxcount).*law(mid))<target ? (lo=mid) : (hi=mid)
    end
    beta=(lo+hi)/2
    # a_1=1 and a_k<=1 give a conservative omitted normalising-mass bound.
    tailbound=exp(beta)*ccdf(Poisson(beta),maxcount)/beta
    tailbound<1e-8 || error("Strauss count truncation not certified")
    (beta=beta,coeff=coeff,coeff_se=se,probabilities=law(beta),tailbound=tailbound)
end

"Independent exact rejection draws from the nonempty Strauss prior."
function validate_strauss(m,R,logalpha,beta;draws=50000,seed=93001)
    rng=Xoshiro(seed); counts=Int[]; tries=0; paircounts=Int[]
    while length(counts)<draws
        tries+=1;tries<1000draws || error("rejection acceptance too low")
        k=rand(rng,Poisson(beta));k==0 && continue
        pairs=conflicts([point(rng,m) for _ in 1:k],R)
        if log(rand(rng))<=logalpha*pairs
            push!(counts,k);push!(paircounts,pairs)
        end
    end
    (mean=mean(counts),se=std(counts)/sqrt(draws),acceptance=draws/tries,
     pair_mean=mean(paircounts),counts=counts)
end
end
