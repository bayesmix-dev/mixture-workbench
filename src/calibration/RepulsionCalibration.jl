module RepulsionCalibration

using GeneralDPPMixtures, LinearAlgebra, Statistics, SpecialFunctions, QuadGK
import ..MahalanobisRepulsion as MR
const G=GeneralDPPMixtures

export occupied_given_total, mfm_occupied_mean, poisson_binomial,
       occupied_moments, sphere_directions, pair_values, pair_profile,
       calibrate, feasibility_cap

"""Exact mean occupied count under symmetric Dirichlet(a,...,a) weights."""
function occupied_given_total(n::Integer,M::Integer,a::Real)
    n>=1 && M>=1 && isfinite(a) && a>0 || throw(ArgumentError("invalid occupancy arguments"))
    M==1 && return 1.
    logempty=logbeta(M*a,Float64(n))-logbeta((M-1)*a,Float64(n))
    -M*expm1(logempty)
end

"""MFM prior mean: M-1|Lambda~Poisson(Lambda), gamma,Lambda~Exp(rate .1).
The omitted-M bound is rigorous; QuadGK's error is a numerical error estimate.
"""
function mfm_occupied_mean(n::Integer;maxM=500,rtol=3e-12)
    n>=1 && maxM>=1 || throw(ArgumentError("invalid MFM mean arguments"))
    q=10/11;p=1/11;weights=[p*q^(M-1) for M in 1:maxM]
    f(x)=exp(-x)*sum(weights[M]*occupied_given_total(n,M,10x) for M in 1:maxM)
    value,error=quadgk(f,0.,.001,.01,.1,1.,10.,Inf;rtol,atol=1e-12)
    (mean=value,quadrature_error=error,count_tail_bound=q^maxM*(maxM+1/p),maxM=maxM)
end

"""Finite Poisson-binomial probabilities through maxM, without renormalising
away the omitted positive tail. The constant term remains included at index1.
"""
function poisson_binomial(lambda;maxM=160)
    maxM>=1 && all(l->isfinite(l)&&0<=l<1,lambda) || throw(ArgumentError("invalid Bernoulli spectrum"))
    T=min(length(lambda),Int(maxM));prob=zeros(T+1);prob[1]=1.
    for (j,l) in enumerate(lambda)
        for M in min(T,j):-1:1;prob[M+1]=(1-l)*prob[M+1]+l*prob[M];end
        prob[1]*=1-l
    end
    prob
end

function occupied_moments(lambda,n::Integer;a=.1,maxM=160)
    n>=1 && a>0 || throw(ArgumentError("invalid allocation prior"))
    probabilities=poisson_binomial(lambda;maxM)
    mu=sum(lambda);mu>0 || throw(ArgumentError("empty spectral mass"))
    logp0=sum(log1p(-l) for l in lambda);p=-expm1(logp0)
    T=length(probabilities)-1
    mean=sum(probabilities[M+1]*occupied_given_total(n,M,a) for M in 1:T)/p
    # For t=log(T/mu)>0, E[N 1(N>T)] <= exp(-t*T)*E[N exp(t*N)]
    # <= T*exp(T-mu-T*log(T/mu)). K<=N bounds its omitted contribution.
    tail=T==length(lambda) ? 0. : T>mu ? T*exp(T-mu-T*log(T/mu))/p : Inf
    retained=sum(probabilities[2:end])/p
    omitted_probability=T==length(lambda) ? 0. : min(max(0.,1-retained),isfinite(tail) ? tail/(T+1) : 1.)
    # These bounds allow root bracketing even when a trial spectrum has most
    # of its mass above M160. No large-M trial is called infeasible merely
    # because the deliberately truncated probability recursion misses it.
    lower=mean+(T==length(lambda) ? 0. : omitted_probability*occupied_given_total(n,T+1,a))
    upper=mean+min(n*omitted_probability,tail)
    (mean_K=mean,mean_K_lower=lower,mean_K_upper=upper,mean_M=mu/p,variance_M=sum(l*(1-l) for l in lambda)/p-(mu/p)^2*exp(logp0),
        nonempty_probability=p,empty_probability=exp(logp0),occupied_tail_bound=tail,
        retained_count_mass=retained,probabilities=probabilities./p)
end
occupied_moments(s::G.Spectrum,n::Integer;kwargs...)=occupied_moments(s.lambda[s.shell],n;kwargs...)

"""Deterministic Fibonacci sphere. Verification uses a different phase and a
fixed orthogonal rotation, rather than a nested subset of calibration nodes.
"""
function sphere_directions(n::Integer;verification=false)
    n>=2 || throw(ArgumentError("at least two sphere directions are needed"))
    dirs=zeros(3,n);golden=pi*(3-sqrt(5.));phase=verification ? sqrt(2.) : 0.
    for j in 1:n
        z=1-2*(j-.5)/n;r=sqrt(max(0.,1-z*z));angle=golden*(j-1)+phase
        dirs[:,j]=[r*cos(angle),r*sin(angle),z]
    end
    if verification
        u=normalize([1.,2.,3.]);theta=.731
        X=[0. -u[3] u[2];u[3] 0. -u[1];-u[2] u[1] 0.]
        rotation=cos(theta)*Matrix{Float64}(I,3,3)+(1-cos(theta))*(u*u')+sin(theta)*X
        dirs=rotation*dirs
    end
    dirs
end

"""Pair correlation conditional on a nonempty finite DPP, evaluated at raw
displacements h=chol(C)*u*r. Equal sine/cosine pairs supply stationarity.
The nonempty correction is p*g_unconditioned, not g_unconditioned/p.
"""
function pair_values(s::G.Spectrum,C,widths,r::Real,directions;nonempty=true)
    d=length(widths);size(C)==(d,d) && size(directions,1)==d || throw(ArgumentError("metric/direction dimension differs"))
    isfinite(r) && r>=0 && all(>(0),widths) || throw(ArgumentError("invalid pair distance"))
    L=cholesky(Symmetric(C)).L;ls=s.lambda[s.shell];mass=sum(ls)
    mass>0 || throw(ArgumentError("empty spectrum"))
    p=nonempty ? -expm1(s.logp0) : 1.
    out=zeros(size(directions,2));crossed=0
    for j in axes(directions,2)
        h=L*directions[:,j]*r
        any(abs.(h).>widths./2) && (crossed+=1)
        phase=2pi.*(transpose(s.frequencies)*(h./widths))
        correlation=sum(ls.*cos.(phase))/mass
        out[j]=p*(1-correlation^2)
    end
    (values=out,mean=mean(out),minimum=minimum(out),maximum=maximum(out),
        half_box_cross_fraction=crossed/size(directions,2))
end

function pair_profile(s,C,widths,radii,directions)
    [merge((radius=Float64(r),),pair_values(s,C,widths,r,directions)) for r in radii]
end

build(B,alpha,c0,nu,cutoff,maxnodes)=MR.build_spectrum(Val(size(B,1)),alpha,c0,nu,cutoff,B;max_box_nodes=maxnodes)

function solve_range(B,C,widths,r,nu,c0,cutoff,dirs;maxnodes,iterations=42,tolerance=2e-9)
    evaluate(alpha)=begin
        s=build(B,alpha,c0,nu,cutoff,maxnodes)
        pair_values(s,C,widths,r,dirs).mean-.99
    end
    lo=Float64(r);hi=4Float64(r);fl=evaluate(lo);fh=evaluate(hi)
    for _ in 1:12
        fl>=0 && break
        lo/=1.3;fl=evaluate(lo)
    end
    for _ in 1:12
        fh<=0 && break
        hi*=1.3;fh=evaluate(hi)
    end
    fl>=0 && fh<=0 || error("could not bracket the .99 MH recovery constraint")
    for _ in 1:iterations
        mid=(lo+hi)/2;fm=evaluate(mid)
        abs(fm)<tolerance && return mid
        fm>0 ? (lo=mid) : (hi=mid)
    end
    (lo+hi)/2
end

function solve_count(reference,n,target,a,cmax;maxM=160,iterations=48)
    base=reference.lambda[reference.shell]./cmax
    maxvalue=occupied_moments(base.*cmax,n;a,maxM)
    maxvalue.mean_K_upper>=target || return nothing,maxvalue
    maxvalue.mean_K_lower>=target || error("count truncation cannot resolve the capacity bracket")
    lo=1e-12;hi=cmax
    for _ in 1:iterations
        mid=(lo+hi)/2
        v=occupied_moments(base.*mid,n;a,maxM)
        if v.mean_K_lower>target
            hi=mid
        elseif v.mean_K_upper<target
            lo=mid
        elseif v.occupied_tail_bound<1e-9
            v.mean_K>target ? (hi=mid) : (lo=mid)
        else
            error("count truncation cannot resolve the occupied-mean root")
        end
    end
    (lo+hi)/2,maxvalue
end

"""Maximal-range boundary along the c0=cmax occupied-mean branch. The range
is its first .99 recovery on the supplied finite search interval; missing
crossings are reported as unavailable, never inferred past the search interval.
"""
function feasibility_cap(B,C,widths,n,target,nu,dirs;alpha_hint=2.,cmax=.99,cutoff=1e-40,a=.1,maxM=160,maxnodes=10_000_000,max_radius=8.)
    function score(alpha)
        v=occupied_moments(build(B,alpha,cmax,nu,cutoff,maxnodes),n;a,maxM)
        v.mean_K_lower>target && return 1.
        v.mean_K_upper<target && return -1.
        v.occupied_tail_bound<1e-9 && return v.mean_K-target
        error("count truncation cannot resolve the feasibility boundary")
    end
    lo=alpha_hint;hi=alpha_hint
    for _ in 1:20;score(lo)>=0 && break;lo/=1.3;end
    for _ in 1:20;score(hi)<=0 && break;hi*=1.3;end
    score(lo)>=0 && score(hi)<=0 || error("could not bracket occupied-mean feasibility boundary")
    for _ in 1:40
        mid=(lo+hi)/2;score(mid)>0 ? (lo=mid) : (hi=mid)
    end
    alpha=(lo+hi)/2;s=build(B,alpha,cmax,nu,cutoff,maxnodes)
    radii=range(0.,Float64(max_radius);length=81)
    index=findfirst(r->pair_values(s,C,widths,r,dirs).mean>=.99,radii)
    radius=if index===nothing
        NaN
    else
        rl=radii[max(1,index-1)];rh=radii[index]
        for _ in 1:35
            rm=(rl+rh)/2;pair_values(s,C,widths,rm,dirs).mean>=.99 ? (rh=rm) : (rl=rm)
        end
        (rl+rh)/2
    end
    (alpha=alpha,c0=cmax,first_recovery_radius=radius,range_search_maximum=Float64(max_radius),
        mean_K=occupied_moments(s,n;a,maxM).mean_K,modes=length(s.kind))
end

"""Calibrate the finite anisotropic DPP before posterior fitting. A requested
radius is never changed when count capacity or verification fails.
"""
function calibrate(C,widths,n::Integer;radius,nu=8.,target=mfm_occupied_mean(n).mean,a=.1,
        cmax=.99,cutoff=1e-40,directions=128,verification_directions=256,
        maxM=160,maxnodes=10_000_000,verification_tolerance=5e-4,profile_points=41)
    length(widths)==3 || throw(ArgumentError("this calibration uses a three-dimensional sphere"))
    radius>0 && 1<target<=n && a>0 && 0<cmax<1 && 0<cutoff<cmax || throw(ArgumentError("invalid calibration target"))
    B=MR.metric_from_covariance(C,widths)
    dirs=sphere_directions(directions);verify=sphere_directions(verification_directions;verification=true)
    c0=cmax;alpha=NaN;maxvalue=nothing;s=nothing
    for _ in 1:4
        alpha=solve_range(B,C,widths,radius,nu,c0,cutoff,dirs;maxnodes)
        reference=build(B,alpha,cmax,nu,cutoff,maxnodes)
        nextc,maxvalue=solve_count(reference,n,target,a,cmax;maxM)
        if nextc===nothing
            cap=feasibility_cap(B,C,widths,n,target,nu,dirs;alpha_hint=alpha,cmax,cutoff,a,maxM,maxnodes,max_radius=max(8.,4radius))
            return (status=:infeasible_count_capacity,radius=Float64(radius),nu=Float64(nu),target=Float64(target),
                alpha=alpha,c0=cmax,maximum_mean_K=maxvalue.mean_K_upper,
                cmax_occupied_mean_bounds=(lower=maxvalue.mean_K_lower,upper=maxvalue.mean_K_upper),feasibility_cap=cap,spectrum=nothing)
        end
        c0=nextc;s=build(B,alpha,c0,nu,cutoff,maxnodes)
        moments=occupied_moments(s,n;a,maxM)
        abs(moments.mean_K-target)<1e-7 && abs(pair_values(s,C,widths,radius,dirs).mean-.99)<2e-8 && break
    end
    moments=occupied_moments(s,n;a,maxM)
    fit=pair_values(s,C,widths,radius,dirs);check=pair_values(s,C,widths,radius,verify)
    numerical_pass=abs(moments.mean_K-target)<1e-7 && moments.occupied_tail_bound<1e-10 &&
        abs(fit.mean-.99)<2e-8 && abs(check.mean-fit.mean)<=verification_tolerance
    radii=collect(range(0.,max(3.,1.5radius);length=profile_points))
    profile128=pair_profile(s,C,widths,radii,dirs);profile256=pair_profile(s,C,widths,radii,verify)
    ls=s.lambda[s.shell]
    (status=numerical_pass ? :feasible : :verification_failed,radius=Float64(radius),nu=Float64(nu),
        target=Float64(target),alpha=alpha,c0=c0,spectrum=s,moments=moments,
        target_profile=fit,verification_profile=check,verification_tolerance=verification_tolerance,
        profile128=profile128,profile256=profile256,maximum_mean_K=maxvalue.mean_K_lower,
        cmax_occupied_mean_bounds=(lower=maxvalue.mean_K_lower,upper=maxvalue.mean_K_upper),
        integrated_unconditioned_deficit=sum(abs2,ls)/sum(ls)^2,
        metric=B,raw_covariance=Matrix(C),widths=collect(widths))
end

end
