module MahalanobisStraussCalibration

using GeneralDPPMixtures,LinearAlgebra,Random,Statistics,Distributions,SpecialFunctions
import ..RepulsionCalibration as RC
const G=GeneralDPPMixtures
export survival_coefficients, count_bounds, calibrate, validate_prior

function geometry(C,widths,radius)
    d=length(widths)
    size(C)==(d,d) && all(>(0),widths) && isfinite(radius) && radius>0 || throw(ArgumentError("invalid Strauss metric/domain/radius"))
    all(isfinite,C) && issymmetric(C) || throw(ArgumentError("invalid covariance metric"))
    Matrix(cholesky(Symmetric(C)).L)\Matrix(Diagonal(widths))
end

function binomial_interval(successes,N,delta)
    lower=successes==0 ? 0. : quantile(Beta(successes,N-successes+1),delta/2)
    upper=successes==N ? 1. : quantile(Beta(successes+1,N-successes),1-delta/2)
    lower,upper
end

"""Independent uniform configurations estimate P(no interaction pair) for
each prefix size. Once a prefix has a conflict all longer prefixes fail, so
the remaining locations need not be drawn. Correlation between prefix sizes
is allowed by the Bonferroni simultaneous binomial intervals.

For finite logalpha<=0, the EXACT integral a_m=E[exp(logalpha*s)] lies between
P(s=0) and P(s=0)+exp(logalpha). No zero success is a mathematical support cap.
"""
function survival_coefficients(C,widths,radius,logalpha;draws=50_000,maxM=160,confidence=.99,seed=28101)
    draws>=2 && maxM>=2 && 0<confidence<1 && isfinite(logalpha) && logalpha<=0 || throw(ArgumentError("invalid Strauss integration settings"))
    T=geometry(C,widths,radius);d=length(widths);hist=zeros(Int,maxM+1);rng=Xoshiro(seed)
    points=zeros(d,maxM);work=zeros(d);unit=zeros(d);R2=radius^2;comparisons=0
    if logalpha==0
        return (estimate=ones(maxM),lower=ones(maxM),upper=ones(maxM),successes=fill(draws,maxM),
            draws=draws,maxM=maxM,confidence=confidence,soft_remainder_bound=0.,comparisons=0,seed=seed,exact_poisson=true)
    end
    for _ in 1:draws
        safe=0
        for m in 1:maxM
            rand!(rng,unit);mul!(work,T,unit);conflict=false
            for j in 1:m-1
                comparisons+=1;distance=0.
                for e in 1:d;distance+=(work[e]-points[e,j])^2;end
                if distance<=R2;conflict=true;break;end
            end
            conflict && break
            points[:,m].=work;safe=m
        end
        hist[safe+1]+=1
    end
    successes=[sum(hist[m+1:end]) for m in 1:maxM]
    estimate=successes./draws;lower=zeros(maxM);upper=zeros(maxM);lower[1]=upper[1]=1.
    remainder=exp(logalpha);delta=(1-confidence)/(maxM-1)
    for m in 2:maxM
        lo,hi=binomial_interval(successes[m],draws,delta)
        lower[m]=lo;upper[m]=min(1.,hi+remainder)
    end
    (estimate=estimate,lower=lower,upper=upper,successes=successes,draws=draws,maxM=maxM,
        confidence=confidence,soft_remainder_bound=remainder,comparisons=comparisons,seed=seed,exact_poisson=false)
end

function extreme_ratio(weights,h,lower,upper,minimise)
    lo=minimum(h);hi=maximum(h)
    for _ in 1:65
        mid=(lo+hi)/2
        value=sum(weights[m]*(h[m]-mid)*
            (minimise ? (h[m]>=mid ? lower[m] : upper[m]) : (h[m]>=mid ? upper[m] : lower[m])) for m in eachindex(h))
        value>0 ? (lo=mid) : (hi=mid)
    end
    (lo+hi)/2
end

"""Propagate simultaneous interaction-integral intervals and an explicit
infinite count tail. beta is activity times the RAW rectangular volume.
The interaction radius is a potential cutoff, not a claim about Strauss g=.99.
"""
function count_bounds(coeff,n::Integer,beta::Real;a=.1)
    n>=1 && a>0 && isfinite(beta) && beta>0 || throw(ArgumentError("invalid count-law arguments"))
    M=length(coeff.estimate);h=[RC.occupied_given_total(n,m,a) for m in 1:M]
    logs=[m*log(beta)-loggamma(m+1.) for m in 1:M];shift=maximum(logs);weights=exp.(logs.-shift)
    Z=sum(weights.*coeff.estimate);Zlo=sum(weights.*coeff.lower);Zhi=sum(weights.*coeff.upper)
    Zlo>0 || error("the numerical lower normalizer vanished")
    # sum_{m>T} beta^m/m! <= exp(beta)*P(Pois(beta)>T)
    # sum_{m>T} m beta^m/m! <= beta*exp(beta)*P(Pois(beta)>=T).
    tailZ=exp(beta+logccdf(Poisson(beta),M)-shift)
    tailM=exp(log(beta)+beta+logccdf(Poisson(beta),M-1)-shift)
    lo=extreme_ratio(weights,h,coeff.lower,coeff.upper,true)
    hi=extreme_ratio(weights,h,coeff.lower,coeff.upper,false)
    mean=sum(weights.*coeff.estimate.*h)/Z
    # Every omitted count has h_m>=h_(T+1). Adding its unknown mass cannot
    # lower the finite-support mean below min(lo,h_(T+1)).
    lower=min(lo,RC.occupied_given_total(n,M+1,a))
    upper=min(Float64(n),hi+tailM/Zlo)
    logacceptlo=log(Zlo)+shift-beta
    logaccepthi=min(0.,log(Zhi+tailZ)+shift-beta)
    (mean_K_estimate=mean,mean_K_lower=lower,mean_K_upper=upper,
        mean_M_estimate=sum(weights.*coeff.estimate.*collect(1:M))/Z,
        count_tail_occupied_bound=tailM/Zlo,count_tail_normalizer_bound=tailZ/Zlo,
        log_rejection_acceptance_lower=logacceptlo,log_rejection_acceptance_upper=logaccepthi,
        rejection_acceptance_estimate=exp(log(Z)+shift-beta),
        probabilities_estimate=weights.*coeff.estimate./Z)
end

"""Calibrate a fixed prior only. A failed confidence/cost check is recorded,
never replaced by a posterior-directed activity or a count truncation.
"""
function calibrate(C,widths,n::Integer;radius,logalpha,target=RC.mfm_occupied_mean(n).mean,a=.1,
        draws=50_000,maxM=160,confidence=.99,seed=28101,max_beta=200.,tolerance=.05,
        minimum_rejection_acceptance=1e-5)
    1<target<n && max_beta>0 && tolerance>0 || throw(ArgumentError("invalid calibration goal"))
    coeff=survival_coefficients(C,widths,radius,logalpha;draws,maxM,confidence,seed)
    lo=.001;hi=min(max_beta,20.)
    for _ in 1:30
        count_bounds(coeff,n,hi;a).mean_K_estimate>=target && break
        hi=min(max_beta,hi*1.4);hi==max_beta && break
    end
    top=count_bounds(coeff,n,hi;a)
    if top.mean_K_estimate<target
        return (status=:insufficient_coefficient_support,beta=hi,target=target,coefficients=coeff,bounds=top,
            radius=Float64(radius),logalpha=Float64(logalpha),tolerance=Float64(tolerance),
            message="The finite Monte Carlo coefficient estimates do not reach the target before the activity work limit; this is not a mathematical count cap.")
    end
    for _ in 1:60
        mid=(lo+hi)/2;count_bounds(coeff,n,mid;a).mean_K_estimate>target ? (hi=mid) : (lo=mid)
    end
    beta=(lo+hi)/2;bounds=count_bounds(coeff,n,beta;a)
    certified=bounds.mean_K_lower>=target-tolerance && bounds.mean_K_upper<=target+tolerance && bounds.count_tail_occupied_bound<1e-6
    cost_ok=bounds.log_rejection_acceptance_lower>=log(minimum_rejection_acceptance)
    status=!certified ? :uncertified_integration : !cost_ok ? :certified_but_rejection_expensive : :calibrated
    (status=status,beta=beta,target=target,coefficients=coeff,bounds=bounds,radius=Float64(radius),logalpha=Float64(logalpha),
        tolerance=Float64(tolerance),message=certified ? "Occupied-mean target lies within the stated simultaneous integration/tail error bounds." :
        "Monte Carlo interaction-integral uncertainty is too large to certify the requested tolerance.")
end

"""Independent exact nonempty Strauss rejection draws, with a hard work
budget. Early rejection after a conflict is exact because future log factors
are nonpositive. No posterior or Markov-chain transition is run.
"""
function validate_prior(C,widths,n::Integer;radius,logalpha,beta,a=.1,draws=2000,max_proposals=2_000_000,seed=39101)
    draws>=2 && max_proposals>=1 && isfinite(logalpha) && logalpha<=0 && beta>0 || throw(ArgumentError("invalid rejection settings"))
    T=geometry(C,widths,radius);d=length(widths);rng=Xoshiro(seed);counts=Int[];values=Float64[];proposals=0;comparisons=0
    unit=zeros(d);work=zeros(d);points=zeros(d,0)
    while length(counts)<draws && proposals<max_proposals
        proposals+=1;M=rand(rng,Poisson(beta));M==0 && continue
        size(points,2)>=M || (points=zeros(d,M))
        threshold=log(rand(rng));logweight=0.;accepted=true
        for m in 1:M
            rand!(rng,unit);mul!(work,T,unit)
            for j in 1:m-1
                comparisons+=1;distance=0.
                for e in 1:d;distance+=(work[e]-points[e,j])^2;end
                distance<=radius^2 && (logweight+=logalpha)
                if logweight<threshold;accepted=false;break;end
            end
            accepted || break
            points[:,m].=work
        end
        if accepted
            push!(counts,M);push!(values,RC.occupied_given_total(n,M,a))
        end
    end
    N=length(counts)
    (complete=N==draws,draws=N,requested_draws=draws,proposals=proposals,max_proposals=max_proposals,
        acceptance=N/proposals,mean_K=N==0 ? NaN : mean(values),mcse_K=N<2 ? NaN : std(values)/sqrt(N),
        mean_M=N==0 ? NaN : mean(counts),counts=counts,comparisons=comparisons,seed=seed)
end

end
