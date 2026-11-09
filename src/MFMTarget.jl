module MFMTarget

using GeneralDPPMixtures,LinearAlgebra,SpecialFunctions
import ..MFMBaseline as MF
const G=GeneralDPPMixtures

export log_marginal,likelihood,partition_logtarget,logtarget,logfallingfactorial

"""Stable log of the falling factorial (M)_k; impossible counts have zero mass."""
function logfallingfactorial(M::Integer,k::Integer)
    0<=k<=M || return -Inf
    sum(log(Float64(M-j)) for j in 0:k-1;init=0.)
end

"""
NIW cluster evidence, integrating BOTH mean and covariance:
Sigma~IW(tau,Omega), mu|Sigma~N(mu0,Sigma/kappa).
Cluster.x and its fixed-location covariance cache are irrelevant to this target.
"""
function log_marginal(s::MF.State,c::G.Cluster)
    n=c.n;n==0 && return 0.
    m=s.model;d=length(s.mu0);kap=s.kappa+n
    delta=c.mean-s.mu0
    scale=m.omega+c.scatter+(s.kappa*n/kap)*(delta*delta')
    ld=2sum(log,diag(cholesky(Symmetric(scale)).L))
    -n*d/2*log(pi)+d/2*(log(s.kappa)-log(kap))+m.tau/2*m.logdetomega-
        (m.tau+n)/2*ld+G.logmultigamma((m.tau+n)/2,d)-G.logmultigamma(m.tau/2,d)
end
function member_cluster(s,members::AbstractVector{<:Integer})
    length(unique(members))==length(members) && all(i->1<=i<=length(s.c),members) ||
        throw(ArgumentError("invalid cluster members"))
    c=G.Cluster(s.model,zero(s.mu0))
    for i in sort(members);G.add!(c,s.model.y[i]);end
    c
end
log_marginal(s::MF.State,members::AbstractVector{<:Integer})=log_marginal(s,member_cluster(s,members))
likelihood(s::MF.State)=sum(log_marginal(s,s.clusters[h]) for h in s.active)

"""
Collapsed canonical-partition target conditional on M and gamma. The falling
factorial counts injections of k occupied blocks into the M finite-mixture labels.
`groups`, when supplied, must be a complete nonempty-block partition of the data.
"""
function partition_logtarget(s::MF.State;groups=nothing,M=s.M,gamma=s.gamma)
    M isa Integer && M>=1 && isfinite(gamma) && gamma>0 || return -Inf
    clusters=if groups===nothing
        [s.clusters[h] for h in s.active]
    else
        all(!isempty,groups) && sort(vcat(groups...))==collect(eachindex(s.c)) ||
            throw(ArgumentError("groups must partition every observation exactly once"))
        [member_cluster(s,g) for g in groups]
    end
    k=length(clusters);k<=M || return -Inf
    n=length(s.c)
    logfallingfactorial(M,k)+loggamma(M*gamma)-loggamma(M*gamma+n)+
        sum(loggamma(c.n+gamma)-loggamma(gamma)+log_marginal(s,c) for c in clusters)
end

"""
Full collapsed joint density of the canonical partition, M, gamma and Lambda,
with M-1|Lambda~Poisson(Lambda) and independent Exp(rate=.1) hyperpriors.
Density is with respect to dgamma dLambda and counting measure on partitions/M.
"""
function logtarget(s::MF.State;groups=nothing,M=s.M,gamma=s.gamma,lambda=s.lambda)
    M isa Integer && M>=1 && isfinite(lambda) && lambda>0 || return -Inf
    part=partition_logtarget(s;groups,M,gamma);isfinite(part) || return part
    part-lambda+(M-1)*log(lambda)-loggamma(M)+2log(.1)-.1*(gamma+lambda)
end

end
