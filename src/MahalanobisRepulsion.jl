module MahalanobisRepulsion

using GeneralDPPMixtures, LinearAlgebra, Statistics, SpecialFunctions
const G=GeneralDPPMixtures

export build_spectrum, spectrum_from_prior, metric_from_covariance,
    domain_bounds, omitted_tail_bound

function positive_metric(A,d)
    size(A)==(d,d) || throw(ArgumentError("The spectral metric must be $d by $d"))
    matrix=Matrix{Float64}(A)
    all(isfinite,matrix) && issymmetric(matrix) ||
        throw(ArgumentError("The spectral metric must be finite and symmetric"))
    factor=try
        cholesky(Symmetric(matrix))
    catch e
        e isa PosDefException || rethrow()
        throw(ArgumentError("The spectral metric must be positive definite"))
    end
    matrix,factor
end

"""Use the same raw rectangular domain as Benchmark.model."""
function domain_bounds(y,domain)
    isempty(y) && throw(ArgumentError("Cannot construct a domain from empty data"))
    d=length(first(y));all(x->length(x)==d && all(isfinite,x),y) ||
        throw(ArgumentError("Data must have a common finite dimension"))
    if domain=="padded"
        center=mean(y)
        zeta=map(e->maximum(abs(x[e]-center[e]) for x in y),1:d)
        return center-2.5zeta,center+2.5zeta
    elseif domain=="data_rectangle"
        return [minimum(x[j] for x in y) for j in 1:d],
            [maximum(x[j] for x in y) for j in 1:d]
    end
    throw(ArgumentError("Unknown location domain: $domain"))
end

"""Transform a raw covariance metric C to A = D^(-1) C D^(-1)."""
function metric_from_covariance(C,widths)
    d=length(widths);all(x->isfinite(x) && x>0,widths) ||
        throw(ArgumentError("Domain widths must be finite and positive"))
    matrix,_=positive_metric(C,d)
    A=[matrix[i,j]/widths[i]/widths[j] for i in 1:d,j in 1:d]
    # Divide the symmetric entries identically; do not let a change in division
    # order introduce an artificial nonsymmetric floating-point matrix.
    for i in 1:d, j in 1:i-1
        A[j,i]=A[i,j]
    end
    positive_metric(A,d)[1]
end

function log_upper_gamma(a,x)
    x==0 && return loggamma(a)
    q=gamma_inc(a,x)[2]
    q>0 && return loggamma(a)+log(q)
    # An upper bound remains representable in log space if Q underflows.
    b=max(a-1,0.)
    x>b || throw(ArgumentError("Could not bound the omitted spectral tail"))
    (a-1)*log(x)-x-log1p(-b/x)
end

"""
Conservative upper bound on the infinite eigenvalue mass omitted by the cutoff.

Unit cubes centred at omitted integer nodes are disjoint. Their metric-radius
deviation is bounded by delta=.5*sqrt(sum(abs,A)). Integrating the decreasing
radial envelope outside R-delta bounds the omitted sum. The resulting incomplete
gamma expression may be loose; it is never silently replaced by an unknown zero.
The actual statistical target is the explicitly retained finite spectrum.
"""
function omitted_tail_bound(A,alpha,c0,nu,cutoff)
    d=size(A,1);matrix,factor=positive_metric(A,d)
    a,c,v,epscut=Float64.((alpha,c0,nu,cutoff))
    isfinite(a) && a>0 && isfinite(c) && 0<c<1 && isfinite(v) && v>0 &&
        isfinite(epscut) && 0<epscut<c || throw(ArgumentError("Invalid spectral parameters"))
    radius=(log(c)-log(epscut))^(1/v)/a
    # Inflate the cube bound and deflate R to keep the bound conservative at
    # floating-point boundaries, including a cutoff exactly on a lattice node.
    delta=nextfloat(.5sqrt(sum(abs,matrix))*(1+32eps(Float64)))
    lower=max(0.,radius*(1-32eps(Float64))-delta)
    u=max(0.,lower-delta);x=(a*u)^v
    terms=Float64[]
    lower<delta && push!(terms,log((delta^d-lower^d)/d))
    for j in 0:d-1
        logterm=log(binomial(d-1,j))+(d-1-j)*log(delta)-log(v)-
            (j+1)*log(a)+log_upper_gamma((j+1)/v,x)
        push!(terms,logterm)
    end
    mx=maximum(terms)
    logr=mx+log(sum(exp(t-mx) for t in terms))
    logarea=log(2.)+(d/2)*log(pi)-loggamma(d/2)
    logbound=log(c)+logarea-sum(log,diag(factor.L))+logr
    logbound<=log(floatmax(Float64))-1 ||
        throw(ArgumentError("The omitted-tail upper bound is not representable"))
    nextfloat(max(floatmin(Float64),exp(logbound)*(1+512eps(Float64))))
end

"""
Construct the finite real Fourier spectrum with eigenvalues
lambda_w = c0*exp(-(alpha*sqrt(w' A w))^nu).

The existing raw-box Fourier functions are unchanged. Each nonzero +/- frequency
pair has a cosine and sine with the same eigenvalue and occupies one shell.
`norms` retains true integer Euclidean squared norms as descriptive metadata;
equal norms need not have equal eigenvalues. No omitted nodes are approximated.
"""
function build_spectrum(::Val{D},alpha,c0,nu,cutoff,A;
        max_box_nodes::Integer=10_000_000) where D
    D>=1 || throw(ArgumentError("The spectral dimension must be positive"))
    matrix,factor=positive_metric(A,D)
    a,c,v,epscut=Float64.((alpha,c0,nu,cutoff))
    isfinite(a) && a>0 && isfinite(c) && 0<c<1 && isfinite(v) && v>0 &&
        isfinite(epscut) && 0<epscut<c || throw(ArgumentError("Invalid spectral parameters"))
    max_box_nodes>=1 || throw(ArgumentError("max_box_nodes must be positive"))
    radius=(log(c)-log(epscut))^(1/v)/a
    isfinite(radius) || throw(ArgumentError("The spectral cutoff radius is not finite"))
    inverse=inv(factor)
    # Bound error in the computed inverse by its residual. If B approximates
    # A^-1 and eta >= ||I-A*B||_inf < 1, then
    # ||A^-1-B||_inf <= ||B||_inf*eta/(1-eta). Include a conservative rounding
    # allowance for the residual/product calculation before using its diagonal.
    # An uncertifiable metric fails instead of silently missing lattice nodes.
    allowance=64D*eps(Float64)
    residual=opnorm(Matrix{Float64}(I,D,D)-matrix*inverse,Inf)
    eta=nextfloat((residual+allowance*opnorm(abs.(matrix)*abs.(inverse),Inf))*(1+allowance))
    eta<1 || throw(ArgumentError("Cannot certify inverse-metric enumeration bounds at Float64 precision"))
    inverse_error=nextfloat(opnorm(inverse,Inf)*eta/(1-eta)*(1+allowance))
    # Cauchy-Schwarz in the A inner product gives |w_i| <= R sqrt((A^-1)_ii).
    # Ceil plus one extra integer layer guards the numerical boundary. We still
    # check the actual eigenvalue at every candidate node in this larger box.
    bounds=ntuple(i->ceil(Int,nextfloat(radius*(1+allowance)*sqrt(inverse[i,i]+inverse_error)))+1,D)
    nodes=prod(big(2)*b+1 for b in bounds)
    nodes<=max_box_nodes || throw(ArgumentError(
        "The exact ellipsoid enumeration needs a $nodes-node bounding box; increase max_box_nodes explicitly or revise the calibration bracket"))
    representatives=Tuple{NTuple{D,Int},Int,Float64}[]
    ranges=ntuple(i->(-bounds[i]):bounds[i],D)
    for w in Iterators.product(ranges...)
        firstnonzero=findfirst(!iszero,w)
        (isnothing(firstnonzero) || w[firstnonzero]>0) || continue
        q=dot(collect(w),matrix*collect(w))
        (isnothing(firstnonzero) ? q==0 : q>0) ||
            throw(ArgumentError("Unresolved numerical loss of positive metric norm"))
        lambda=c*exp(-(a*sqrt(q))^v)
        lambda>=epscut || continue
        push!(representatives,(w,sum(abs2,w),lambda))
    end
    sort!(representatives;by=x->(x[2],x[1]))
    isempty(representatives) && error("The constant Fourier mode is missing")
    frequencies=NTuple{D,Int}[];kinds=UInt8[];shell=Int[]
    shellranges=UnitRange{Int}[];norms=Int[];lambdas=Float64[]
    for (h,(w,norm2,lambda)) in enumerate(representatives)
        firstmode=length(kinds)+1
        push!(frequencies,w);push!(kinds,norm2==0 ? 0 : 1);push!(shell,h)
        if norm2>0
            push!(frequencies,w);push!(kinds,2);push!(shell,h)
        end
        push!(shellranges,firstmode:length(kinds));push!(norms,norm2);push!(lambdas,lambda)
    end
    freq=Matrix{Int}(undef,D,length(frequencies))
    for j in eachindex(frequencies), i in 1:D;freq[i,j]=frequencies[j][i];end
    mass=sum(length(r)*lambdas[h] for (h,r) in enumerate(shellranges))
    logp0=sum(length(r)*log1p(-lambdas[h]) for (h,r) in enumerate(shellranges))
    tail=omitted_tail_bound(matrix,a,c,v,epscut)
    G.Spectrum{D}(freq,kinds,shell,shellranges,norms,lambdas,
        log.(lambdas).-log1p.(-lambdas),a,v,c,epscut,mass,mass/(-expm1(logp0)),tail,logp0)
end

function covariance_rows(rows,d)
    rows isa AbstractMatrix && return positive_metric(rows,d)[1]
    rows isa AbstractVector && length(rows)==d &&
        all(row->row isa AbstractVector && length(row)==d,rows) ||
        throw(ArgumentError("dpp_metric_matrix must contain $d raw covariance rows"))
    matrix=[Float64(rows[i][j]) for i in 1:d,j in 1:d]
    positive_metric(matrix,d)[1]
end

"""Optional metric factory; an absent/euclidean metric uses the original path."""
function spectrum_from_prior(y,p,domain)
    d=length(first(y));metric=get(p,"dpp_metric","euclidean")
    if metric=="euclidean"
        # Preserve the original constructor call exactly, including its finite
        # shell grouping and metadata. No metric-only validation runs here.
        return G.build_spectrum(Val(d),p["dpp_alpha"],p["dpp_c0"],p["dpp_nu"],p["dpp_cutoff"])
    end
    metric=="mahalanobis" || throw(ArgumentError("Unknown DPP metric: $metric"))
    haskey(p,"dpp_metric_matrix") || throw(ArgumentError("Missing dpp_metric_matrix"))
    C=covariance_rows(p["dpp_metric_matrix"],d)
    lower,upper=domain_bounds(y,domain)
    A=metric_from_covariance(C,upper-lower)
    build_spectrum(Val(d),p["dpp_alpha"],p["dpp_c0"],p["dpp_nu"],p["dpp_cutoff"],A)
end

end
