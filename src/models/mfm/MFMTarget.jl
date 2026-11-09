module MFMTarget

using GeneralDPPMixtures,LinearAlgebra,SpecialFunctions
import ..MFMBaseline as MF
const G=GeneralDPPMixtures

export log_marginal,likelihood,partition_logtarget,logtarget,logfallingfactorial

include("../../priors/finite_mixture/counts.jl")
include("../../components/gaussian_niw.jl")

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
