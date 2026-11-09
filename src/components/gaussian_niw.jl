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
